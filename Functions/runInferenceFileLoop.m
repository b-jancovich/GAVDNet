function [results, deviceState] = runInferenceFileLoop(filePaths, fileNames, ...
        startIdx, results, model, opts, deviceState, cachePath)
% RUNINFERENCEFILELOOP Run the GAVDNet detector over a list of audio files.
%
% This is the per-file inference loop extracted from the production run
% script so it can be shared by the single-GPU serial path AND by each
% worker of the dual-GPU path. It is INDEX-AGNOSTIC: it processes
% filePaths(startIdx:end) into results(startIdx:end) using local indices, so
% the caller can pass either the whole year's file list (serial, local ==
% global) or one worker's contiguous sub-range (dual-GPU) and merge later.
%
% Inputs:
%   filePaths    - cellstr of full audio file paths for THIS loop
%   fileNames    - cellstr of the corresponding base names (for cache
%                  validation on resume; = {chFiles(range).name})
%   startIdx     - local index of the first file to process (resume point)
%   results      - existing results struct array (preloaded from this loop's
%                  cache on resume; struct([]) for a fresh start)
%   model        - trained GAVDNet model struct
%   opts         - struct of options with fields: featureFraming,
%                  frameStandardization, minSilenceDuration, plotting,
%                  activationThreshold, windowDur, maxInferenceRetries,
%                  maxConsecGpuFailures, gpuResetEveryN, enableShortFileSkip,
%                  shortFileSkipThreshSec, partialCacheEveryN, cacheShardSize,
%                  progressLabel (a short string prefixed to progress lines,
%                  e.g. '[GPU A] ', or '' for the serial run)
%   deviceState  - struct of GPU/device state with fields: useGPU,
%                  gpuDeviceID, bytesAvailable, gpuChain, currentGpuChainIdx,
%                  cpuMemoryBytes, consecGpuFailures
%   cachePath    - full path to this loop's partial cache (a legacy-style
%                  .mat name; shard/manifest names are derived from it)
%
% Outputs:
%   results      - the grown results struct array (local indexing)
%   deviceState  - the updated device state (device may have fallen back)
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Unpack device state into locals (mutated through the loop, repacked on exit).
useGPU             = deviceState.useGPU;
gpuDeviceID        = deviceState.gpuDeviceID;
bytesAvailable     = deviceState.bytesAvailable;
gpuChain           = deviceState.gpuChain;
currentGpuChainIdx = deviceState.currentGpuChainIdx;
cpuMemoryBytes     = deviceState.cpuMemoryBytes;
consecGpuFailures  = deviceState.consecGpuFailures;

label = opts.progressLabel;
nFiles = numel(filePaths);

for fileIdx = startIdx:nFiles
    filePath = filePaths{fileIdx};
    [~, fileName, fileExt] = fileparts(filePath);

    % Announce start
    fprintf('%sRunning inference on file %d of %d...\n', label, fileIdx, nFiles)

    % Periodically clear GPU memory and refresh the free-memory snapshot. A
    % full wait+reset(gpuDevice) rebuilds the CUDA context to clear
    % accumulated VRAM fragmentation / leaks, but over the Thunderbolt eGPU
    % link it is expensive, so it is throttled to every gpuResetEveryN files
    % plus the first file of this loop. The failure paths already reset (the
    % inference retry wrapper and stepGpuFallback both reset the device), so
    % on-error resets are covered without doing one every file. The cheap
    % AvailableMemory read below still runs every file so
    % estimateInferenceMinibatchSize gets a current value and leak /
    % fragmentation trends stay visible in the diary. If the reset itself
    % errors, treat the device as failed and step the fallback chain.
    if useGPU
        if fileIdx == startIdx || mod(fileIdx, opts.gpuResetEveryN) == 0
            try
                wait(gpuDevice(gpuDeviceID));
                reset(gpuDevice(gpuDeviceID));
            catch ME
                warning(['%sPeriodic GPU reset failed: %s. Stepping ' ...
                    'fallback chain.'], label, ME.message)
                [useGPU, gpuDeviceID, bytesAvailable, currentGpuChainIdx, switchedTo] = ...
                    stepGpuFallback(currentGpuChainIdx, gpuChain, cpuMemoryBytes);
                fprintf('\t%sNow using: %s\n', label, switchedTo)
                consecGpuFailures = 0;
            end
        end
        % Cheap per-file free-memory read (property access, not a context
        % reset). Skipped when a failed reset above dropped us to CPU.
        if useGPU
            bytesAvailable = gpuDevice(gpuDeviceID).AvailableMemory;
            fprintf('\t%sGPU available memory: %.2f GB\n', label, bytesAvailable / 1e9)
        end
    end

    % Short-file skip (A3, optional). Before the expensive audio read, cheaply
    % check the file duration via audioinfo (header only, no sample decode). A
    % file shorter than shortFileSkipThreshSec cannot produce a detection that
    % survives the postprocessing length threshold, so it is marked skipped
    % and left un-read. If the header cannot be read, DO NOT skip - fall
    % through to the normal read path, which owns retry / failure handling.
    if opts.enableShortFileSkip
        headerReadOK = true;
        try
            audioMeta = audioinfo(filePath);
        catch
            headerReadOK = false;
        end
        if headerReadOK && audioMeta.Duration < opts.shortFileSkipThreshSec
            results(fileIdx).fileName = [fileName, fileExt];
            results(fileIdx).fileFs = audioMeta.SampleRate;
            results(fileIdx).fileSamps = audioMeta.TotalSamples;
            results(fileIdx).fileDuration = audioMeta.Duration;
            results(fileIdx).probabilities = [];
            results(fileIdx).failComment = sprintf(...
                'Skipped: duration %.3f s < short-file threshold %.3f s', ...
                audioMeta.Duration, opts.shortFileSkipThreshSec);
            fprintf('\t%sSkipped short file (%.3f s < %.3f s threshold).\n', ...
                label, audioMeta.Duration, opts.shortFileSkipThreshSec)
            if mod(fileIdx, opts.partialCacheEveryN) == 0
                saveResultsToPartialCache(cachePath, results, fileIdx, ...
                    opts.featureFraming, opts.frameStandardization, fileNames, opts.cacheShardSize);
            end
            continue
        end
    end

    % Read audio file (retries with 60/180/300 s waits, then 300 s cooldown).
    fprintf('\t%sReading audio...\n', label)
    [audioIn, sampleRate, readErr] = audioReadWithRetry(filePath);
    results(fileIdx).fileName = [fileName, fileExt];
    if isempty(audioIn)
        warning('%s\tCould not read file %d (%s%s) after retries: %s. Skipping...', ...
            label, fileIdx, fileName, fileExt, readErr)
        results(fileIdx).failComment = sprintf('Read failed after retries: %s', readErr);
        if mod(fileIdx, opts.partialCacheEveryN) == 0
            saveResultsToPartialCache(cachePath, results, fileIdx, ...
                opts.featureFraming, opts.frameStandardization, fileNames, opts.cacheShardSize);
        end
        continue
    end

    % Write filename, Fs and file info to detections struct
    results(fileIdx).fileFs = sampleRate;
    results(fileIdx).fileSamps = length(audioIn);
    results(fileIdx).fileDuration = results(fileIdx).fileSamps / results(fileIdx).fileFs;
    results(fileIdx).probabilities = [];

    % Extract datetime from filename
    fprintf('\t%sExtracting datetime stamp from audio filename...\n', label)
    results(fileIdx).fileStartDateTime = extractDatetimeFromFilename(filePath, 'datetime');

    % Skip this file if its name doesn't contain a valid start date
    if isempty(results(fileIdx).fileStartDateTime) || ...
            isnat(results(fileIdx).fileStartDateTime)
        warning('%s\tCould not extract datetime from filename: %s. Skipping...', ...
            label, results(fileIdx).fileName)
        results(fileIdx).failComment = 'Could not read valid recording start data-time from filename';
        if mod(fileIdx, opts.partialCacheEveryN) == 0
            saveResultsToPartialCache(cachePath, results, fileIdx, ...
                opts.featureFraming, opts.frameStandardization, fileNames, opts.cacheShardSize);
        end
        continue
    end

    % Skip this file if it doesn't contain valid audio
    if isValidAudio(audioIn) == false
        warning('%s\tFile %s did not contain valid audio. Skipping...', ...
            label, results(fileIdx).fileName)
        results(fileIdx).failComment = 'Could not read valid audio from file';
        if mod(fileIdx, opts.partialCacheEveryN) == 0
            saveResultsToPartialCache(cachePath, results, fileIdx, ...
                opts.featureFraming, opts.frameStandardization, fileNames, opts.cacheShardSize);
        end
        continue
    end

    % Run preprocessing and inference with retry on transient GPU errors.
    % Each retry resets the GPU and pauses briefly before the next attempt.
    % Files that fail all retries are marked with failComment; the
    % consecutive-failure tracker below decides whether to fall back.
    inferenceSucceeded = false;
    lastInferenceErr = '';
    execTime = NaN;
    numAudioSegments = NaN;
    for attempt = 1:opts.maxInferenceRetries
        try
            [results(fileIdx).probabilities, ~, execTime, ...
                results(fileIdx).silenceMask, numAudioSegments] = gavdNetInference(...
                audioIn, sampleRate, model, bytesAvailable, ...
                opts.featureFraming, opts.frameStandardization, opts.minSilenceDuration, opts.plotting);
            inferenceSucceeded = true;
            consecGpuFailures = 0;
            break
        catch ME
            lastInferenceErr = ME.message;
            warning('%sInference attempt %d/%d failed for %s: %s', ...
                label, attempt, opts.maxInferenceRetries, [fileName, fileExt], ME.message)
            if useGPU
                try
                    wait(gpuDevice(gpuDeviceID));
                    reset(gpuDevice(gpuDeviceID));
                    bytesAvailable = gpuDevice(gpuDeviceID).AvailableMemory;
                catch resetME
                    warning('\t%sGPU reset between inference retries also failed: %s', ...
                        label, resetME.message)
                end
            end
            if attempt < opts.maxInferenceRetries
                pause(10);
            end
        end
    end

    if ~inferenceSucceeded
        fprintf('\t%sInference FAILED after %d attempts. Marking file as failed.\n', ...
            label, opts.maxInferenceRetries)
        results(fileIdx).failComment = sprintf('Inference failed after %d retries: %s', ...
            opts.maxInferenceRetries, lastInferenceErr);
        results(fileIdx).probabilities = [];

        if useGPU
            consecGpuFailures = consecGpuFailures + 1;
            threshold = currentFailureThreshold(currentGpuChainIdx, opts.maxConsecGpuFailures);
            if consecGpuFailures >= threshold
                warning(['%sInference failed for %d consecutive file(s) on the ' ...
                    'current device. Stepping fallback chain.'], label, consecGpuFailures)
                [useGPU, gpuDeviceID, bytesAvailable, currentGpuChainIdx, switchedTo] = ...
                    stepGpuFallback(currentGpuChainIdx, gpuChain, cpuMemoryBytes);
                fprintf('\t%sNow using: %s\n', label, switchedTo)
                consecGpuFailures = 0;
            end
        end

        if mod(fileIdx, opts.partialCacheEveryN) == 0
            saveResultsToPartialCache(cachePath, results, fileIdx, ...
                opts.featureFraming, opts.frameStandardization, fileNames, opts.cacheShardSize);
        end
        continue
    end

    % Report execution time and seconds of audio with high probability
    numTimeBinsProbHigh = sum(results(fileIdx).probabilities > opts.activationThreshold);
    secondsBinsProbHigh = numTimeBinsProbHigh * opts.windowDur;
    fprintf('\t%sInference Completed in %.2f seconds\n', label, execTime)
    fprintf('\t%sTotal audio duration: %.2f seconds\n', label, results(fileIdx).fileDuration)
    fprintf('\t%sDuration with raw detection probability > Activation Threshold%%: %.2f seconds.\n\n', ...
        label, secondsBinsProbHigh)

    % Write diagnostic information to the results struct
    results(fileIdx).probsAllNan = all(isnan(results(fileIdx).probabilities));
    results(fileIdx).probsAnyNan = any(isnan(results(fileIdx).probabilities));
    results(fileIdx).audioAllSilence = all(results(fileIdx).silenceMask == true);
    results(fileIdx).audioAnySilence = any(results(fileIdx).silenceMask == true);
    results(fileIdx).numSplitAudioSegments = numAudioSegments;

    % Per-file partial cache, throttled to every partialCacheEveryN files.
    if mod(fileIdx, opts.partialCacheEveryN) == 0
        saveResultsToPartialCache(cachePath, results, fileIdx, ...
            opts.featureFraming, opts.frameStandardization, fileNames, opts.cacheShardSize);
    end
end

% Repack the (possibly changed) device state for the caller.
deviceState.useGPU             = useGPU;
deviceState.gpuDeviceID        = gpuDeviceID;
deviceState.bytesAvailable     = bytesAvailable;
deviceState.currentGpuChainIdx = currentGpuChainIdx;
deviceState.consecGpuFailures  = consecGpuFailures;
end
