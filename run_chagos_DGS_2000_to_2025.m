% GAVDNet Inference System - Diego Garcia 2000 to 2025 Production Run
%
% Run audio files throught the trained model to detect the target animal
% call.
%
% This is a real-world production run (not a model-comparison evaluation):
% it uses a single fixed model and steps through years of Diego Garcia
% hydrophone data, producing one detections .mat file per year, all stored
% in the same output directory.
%
% Audio is expected to be laid out as:
%       <audioRoot>\<YEAR>\H08S<channel>_<timestamp>_cal<...>.wav
% Each year folder holds three synchronous channels (H08S1, H08S2, H08S3);
% detection runs on channel 1 only (channelPrefix = "H08S1").
%
% Year folders are auto-discovered by scanning <audioRoot> for 4-digit
% numeric subfolders. Year folders that contain no channel-1 files are
% skipped with a warning.
%
% Model inference runs on audio files in a loop, then saves raw results to
% disk (raw results = probabilities for target call presence per STFT time
% bin). The post processing procedure that converts these raw probabilities
% into discrete detection boundaries is run in a second loop after
% reloading the raw data. The script is built this way so that post
% processing parameters can be iteratively fine tuned without having to run
% all the audio through the model again, which is computationally
% expensive, time consuming and energy inefficient.
%
% Per-year output files (all in inferenceOutputPath except as noted):
%   detector_raw_results_<year>.mat                 (raw probabilities, cached)
%   detector_results_postprocessed_<year>.mat       (final detections, first run)
%   detector_results_postprocessed_<year>_<ts>.mat  (re-runs with different postproc settings)
%   detector_inference_log_<year>_<ts>.txt          (diary log)
%   %TEMP%\GAVDNet\detector_raw_partial_<year>.mat  (periodic checkpoint, local SSD,
%                                                    deleted on year completion)
%
% The script is interruptible and restartable in three stages:
%   * On launch, every existing detector_results_postprocessed_<year>*.mat
%     is scanned. If any saved postProcOptions / featureFraming /
%     frameStandardization match the current run's settings, that year is
%     skipped entirely (no diary, no inference, no postprocessing).
%   * Else, if detector_raw_results_<year>.mat exists, model inference is
%     skipped and only the (cheap) postprocessing stage re-runs. The new
%     postproc output is saved to detector_results_postprocessed_<year>.mat
%     if that name is free, otherwise to a _<ts>-suffixed filename so the
%     prior run's output is preserved.
%   * Else, if %TEMP%\GAVDNet\detector_raw_partial_<year>.mat exists AND
%     its saved featureFraming / frameStandardization / channel-1 file
%     list match the current run, inference resumes from the next
%     un-processed file. Mismatched partial caches are discarded with a
%     warning. The partial cache is flushed every partialCacheEveryN
%     files (default 100) and deleted once the full raw .mat is saved.
%     A legacy partial cache at the pre-2026-05-12 on-OneDrive location
%     is auto-migrated to %TEMP% on first encounter.
% To force a year's inference to re-run, delete its raw .mat as well.
%
% eGPU stability mitigations (Thunderbolt RTX 4090 + internal T550 fallback):
%   * Periodic partial cache (every partialCacheEveryN files, default 100)
%     written to local SSD under %TEMP%\GAVDNet\. Worst case crash loss
%     is partialCacheEveryN - 1 files; the cache lives off OneDrive so
%     the per-flush write does not incur sync-agent stalls.
%   * Year-start GPU health check via Functions/gpuHealthCheck: small
%     sentinel computation + 3-iter 200 MB H<->D throughput probe. Logs
%     sentinel status, available memory, mean/std/min throughput. Warns
%     on marginal links; forces a fallback step on a failed sentinel.
%   * try/catch+retry wrapper around gavdNetInference (maxInferenceRetries
%     attempts with a wait+reset(gpuDevice) between each). Audio I/O is
%     unchanged - audioReadWithRetry already handles transient drive blips.
%   * Automatic device fallback chain: after maxConsecGpuFailures
%     consecutive inference failures on the primary GPU the script
%     switches to the next GPU in the chain (T550 internal); after a
%     single failure on any non-primary GPU it switches to CPU. Once a
%     fallback has occurred the script stays on the new device for the
%     remainder of the run - re-enabling the primary GPU requires
%     restarting MATLAB.
%   * Per-file GPU AvailableMemory is logged before each inference so
%     leaks / fragmentation trends are visible in the diary.
%
% This script assumes a single MATLAB instance owns the eGPU at any one
% time. Launching a second MATLAB process that also creates a CUDA context
% on the same GPU has been observed to destabilise the device.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%% Init

clear
close all
clc
clear persistent

%% **** USER INPUT ****

plotting = false;

% Path to the config file:
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos_exclude_chorus.m";

% Trained model Data Path (single fixed model for production run):
gavdNetDataPath = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar Exclude Chorus";

% Root folder containing per-year subfolders of audio:
audioRoot = "E:\Diego Garcia South 3Ch";

% Filename prefix that selects channel 1 only. The hydrophone files in
% each year folder are named H08S1_*, H08S2_*, H08S3_* for channels 1, 2,
% 3 respectively; we only run detection on Ch1.
channelPrefix = "H08S1";

% Output path - all per-year results files go here:
inferenceOutputPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_DGS_Detections_2000_to_2025\-10 to 10 single exemplar exclude chorus";

% --- eGPU stability mitigations ---
% Max attempts at gavdNetInference per file before marking the file as
% failed. Each retry resets the GPU and pauses briefly.
maxInferenceRetries = 3;

% After this many CONSECUTIVE failed files on the primary GPU, step down
% the fallback chain (primary GPU -> next GPU -> CPU). Non-primary GPUs
% step after a single failure regardless of this setting.
maxConsecGpuFailures = 3;

% Minimum acceptable mean host<->device throughput (GB/s) for the
% year-start health check. Below this the link is logged as 'marginal'
% (warning only - does not trigger fallback on its own).
healthCheckThroughputMin = 1.5;

% Per-file partial cache (detector_raw_partial_<year>.mat) is flushed to
% disk every this many files. The cache lives in %TEMP%\GAVDNet\ (local
% SSD) rather than inferenceOutputPath, so OneDrive sync never sees it.
% Worst case crash loss = partialCacheEveryN - 1 files. The full raw
% cache is always written at year completion regardless of this setting.
partialCacheEveryN = 250;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.

%% One-time setup (model + GPU loaded ONCE, outside the year loop)

% NOTE: GAVDNet_config_DGS_chagos.m is a plain script (not a function),
% so `run(configPath)` dumps every variable in the config straight into
% this workspace - including its own values for `gavdNetDataPath` and
% `inferenceOutputPath`, which clobber the USER INPUT values set above.
% Snapshot the user inputs before run() so we can re-apply them after.
userGavdNetDataPath     = gavdNetDataPath;
userInferenceOutputPath = inferenceOutputPath;

% Add dependencies to path & load config
run(configPath) % Load config file
projectRoot = pwd;
[gitRoot, ~, ~] = fileparts(projectRoot);
addpath(fullfile(projectRoot, "Functions"))

% Re-apply USER INPUT values that were just overwritten by run(configPath):
gavdNetDataPath     = userGavdNetDataPath;
inferenceOutputPath = userInferenceOutputPath;
clear userGavdNetDataPath userInferenceOutputPath

fprintf('\tGAVDNet model data path: %s\n', gavdNetDataPath)
fprintf('\tAudio root: %s\n', audioRoot)
fprintf('\tChannel prefix: %s\n', channelPrefix)
fprintf('\tInference output path: %s\n', inferenceOutputPath)

% Load the trained model. Handle multiple model files with a UI dialog:
modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
if isscalar(modelList)
    load(fullfile(modelList.folder, modelList.name))
    fprintf('Loading model: %s\n', modelList.name)
else
    [file, location] = uigetfile(gavdNetDataPath, 'Select a model to load:');
    load(fullfile(location, file))
end

% Set the length threshold parameter for the post-processing.
% Use the mean training call length, multiplied by the scaling
% factor set in config:
postProcOptions.LT = model.dataSynthesisParams.meanTargetCallDuration .* ...
    postProcOptions.LT_scaler;

% Set maximum expected call duration from the longest signal in the
% training dataset, with a +20% tolerance.
postProcOptions.maxTargetCallDuration = model.dataSynthesisParams.maxTargetCallDuration * 1.2;

% Set up for GPU or CPU processing (one device for the whole run):
[useGPU, gpuDeviceID, ~, bytesAvailable] = gpuConfig();

% Build the GPU fallback chain. Entries are sorted by TotalMemory
% descending so chain(1) is the largest / primary GPU (RTX 4090 eGPU on
% this machine) and chain(end) is the smallest (T550 internal). The
% stepGpuFallback helper at the bottom of this script walks this chain
% on consecutive inference failures, eventually dropping to CPU.
numGPUs = gpuDeviceCount("available");
if numGPUs > 0
    gpuChain = struct('deviceID', cell(1, numGPUs), ...
                      'Name', cell(1, numGPUs), ...
                      'TotalMemory', cell(1, numGPUs));
    for k = 1:numGPUs
        info = gpuDevice(k);
        gpuChain(k).deviceID = k;
        gpuChain(k).Name = char(info.Name);
        gpuChain(k).TotalMemory = info.TotalMemory;
    end
    [~, sortIdx] = sort([gpuChain.TotalMemory], 'descend');
    gpuChain = gpuChain(sortIdx);

    fprintf('GPU fallback chain (largest -> smallest):\n')
    for k = 1:numel(gpuChain)
        fprintf('\t[%d] device %d: %s (%.1f GB total)\n', ...
            k, gpuChain(k).deviceID, gpuChain(k).Name, ...
            gpuChain(k).TotalMemory / 1e9)
    end

    % Initial chain index = wherever gpuConfig's chosen device sits in
    % the chain. (gpuConfig picks by AvailableMemory, this chain is
    % sorted by TotalMemory - they usually agree but not always.)
    [~, currentGpuChainIdx] = ismember(gpuDeviceID, [gpuChain.deviceID]);
    if currentGpuChainIdx == 0
        currentGpuChainIdx = 1;
    end

    % Restore active device to gpuConfig's selection - the enumeration
    % loop above left whichever GPU was indexed last as the active one.
    gpuDevice(gpuDeviceID);
else
    gpuChain = struct('deviceID', {}, 'Name', {}, 'TotalMemory', {});
    currentGpuChainIdx = 0;  % CPU
end

% Snapshot available CPU memory in bytes (used as bytesAvailable when the
% fallback chain reaches CPU). Mirrors gpuConfig's getSystemMemoryGB logic.
try
    if ispc
        memInfo = memory;
        cpuMemoryBytes = memInfo.MemAvailableAllArrays;
    else
        cpuMemoryBytes = 16e9;
    end
catch
    cpuMemoryBytes = 16e9;
end
fprintf('CPU available memory (for fallback): %.2f GB\n', cpuMemoryBytes / 1e9)

% Local-SSD directory for the per-file partial caches. Kept off OneDrive
% so the checkpoint writes do not trigger sync agent stalls; the partial
% cache only needs to survive process crashes, not multi-machine sharing,
% so a local path is appropriate. Windows %TEMP% survives reboots.
partialCacheDir = fullfile(tempdir, 'GAVDNet');
if ~exist(partialCacheDir, 'dir')
    [ok, msg] = mkdir(partialCacheDir);
    if ~ok
        error('Could not create partial cache directory %s: %s', ...
            partialCacheDir, msg)
    end
end
fprintf('Partial cache directory: %s\n', partialCacheDir)

% NOTE: model is loaded once outside the year loop because fsTarget is
% constant across years. This means the persistent Mel filterbank cached
% inside gavdNetPreprocess (keyed to fsTarget) and the persistent filter
% coefficients cached inside eventSplitter remain valid for the entire
% run, so we do NOT call `clear gavdNetPreprocess` or `clear persistent`
% between years - that would force unnecessary recomputation.

%% Discover year subfolders under audioRoot

% A "year subfolder" is a 4-digit numeric directory name (e.g. "2007").
dirEntries = dir(audioRoot);
isYearFolder = false(numel(dirEntries), 1);
for k = 1:numel(dirEntries)
    name = dirEntries(k).name;
    isYearFolder(k) = dirEntries(k).isdir && ...
                      strlength(name) == 4 && ...
                      ~isnan(str2double(name));
end
yearFolderEntries = dirEntries(isYearFolder);
years = sort(arrayfun(@(e) str2double(e.name), yearFolderEntries));

if isempty(years)
    error('No 4-digit year subfolders found under %s', audioRoot)
end

fprintf('Found %d year subfolders: %s\n', numel(years), ...
    strjoin(string(years), ', '))

%% Main loop: per year




years = 2019:2025;


for yearIdx = 1:numel(years)
    year = years(yearIdx);
    yearDir = fullfile(audioRoot, num2str(year));

    %% Per-year output paths
    saveNamePathRaw = fullfile(inferenceOutputPath, ...
        sprintf('detector_raw_results_%d.mat', year));
    saveNamePathPartial = fullfile(partialCacheDir, ...
        sprintf('detector_raw_partial_%d.mat', year));
    saveNamePathPostUnsuffixed = fullfile(inferenceOutputPath, ...
        sprintf('detector_results_postprocessed_%d.mat', year));

    % Backward-compat: if a legacy partial cache exists at the old
    % on-OneDrive location (from before the local-SSD move), migrate it
    % to the new local path so resume still works for in-flight years.
    legacyPartialPath = fullfile(inferenceOutputPath, ...
        sprintf('detector_raw_partial_%d.mat', year));
    if ~exist(saveNamePathPartial, 'file') && exist(legacyPartialPath, 'file')
        try
            movefile(legacyPartialPath, saveNamePathPartial)
            fprintf('Migrated legacy partial cache: %s -> %s\n', ...
                legacyPartialPath, saveNamePathPartial)
        catch ME
            warning('Could not migrate legacy partial cache %s: %s', ...
                legacyPartialPath, ME.message)
        end
    end

    %% Skip check: any existing postproc file matching current settings?
    % Glob all postproc files for this year (unsuffixed + any timestamped
    % re-runs) and load only the three settings structs from each. If any
    % file's saved settings match the current run's settings exactly, the
    % year is already done and we skip it entirely - no diary, no work.
    matchedPostprocFile = '';
    existingPostprocFiles = dir(fullfile(inferenceOutputPath, ...
        sprintf('detector_results_postprocessed_%d*.mat', year)));
    for k = 1:numel(existingPostprocFiles)
        candidatePath = fullfile(existingPostprocFiles(k).folder, ...
            existingPostprocFiles(k).name);
        try
            saved = load(candidatePath, 'postProcOptions', ...
                'featureFraming', 'frameStandardization');
        catch ME
            warning('Could not inspect %s: %s. Treating as non-match.', ...
                existingPostprocFiles(k).name, ME.message)
            continue
        end
        if all(isfield(saved, {'postProcOptions','featureFraming','frameStandardization'})) && ...
                isequaln(saved.postProcOptions,      postProcOptions)      && ...
                isequaln(saved.featureFraming,       featureFraming)       && ...
                isequaln(saved.frameStandardization, frameStandardization)
            matchedPostprocFile = existingPostprocFiles(k).name;
            break
        end
    end

    if ~isempty(matchedPostprocFile)
        fprintf(['=== Year %d: postprocessed results matching current ' ...
            'settings already exist (%s), skipping. ===\n'], ...
            year, matchedPostprocFile)
        continue
    end

    %% Per-year run timestamp (shared between diary log and any new postproc file)
    ts = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));

    %% Pick a non-clobbering postproc output filename.
    % If the unsuffixed slot is free, take it (matches legacy file naming).
    % Otherwise append the run timestamp so prior runs are preserved.
    if exist(saveNamePathPostUnsuffixed, 'file')
        saveNamePathPost = fullfile(inferenceOutputPath, ...
            sprintf('detector_results_postprocessed_%d_%s.mat', year, ts));
    else
        saveNamePathPost = saveNamePathPostUnsuffixed;
    end

    %% Start logging (per-year diary)
    logname = sprintf('detector_inference_log_%d_%s.txt', year, ts);
    diary(fullfile(inferenceOutputPath, logname));
    fprintf('=== Year %d ===\n', year)

    %% Build the channel-1 file list for this year
    chFiles = dir(fullfile(yearDir, sprintf('%s_*.wav', channelPrefix)));
    if isempty(chFiles)
        warning('No %s_* files in %s. Skipping year %d.', ...
            channelPrefix, yearDir, year)
        diary off
        continue
    end
    chFilePaths = fullfile({chFiles.folder}, {chFiles.name});
    fprintf('Found %d %s_* files for %d.\n', numel(chFilePaths), ...
        channelPrefix, year)

    % Per-year audio path used by the postprocessing loop's audioread()
    % (results(i).fileName is just the basename).
    inferenceAudioPath = yearDir;

    %% Run Model

    % Files are read with audioReadWithRetry rather than via audioDatastore.
    % Empirically, transient I/O failures on external drives (USB blips,
    % OneDrive sync races, antivirus locks) caused ~4% of files to be
    % silently dropped from a 2019 run because the single try/catch had no
    % retry logic and audioDatastore's internal-pointer behaviour on a
    % failed read() is not contractually defined. A direct for-loop over
    % chFilePaths with audioReadWithRetry sidesteps both issues. Audio is
    % loaded on the CPU and only moved to the GPU later, inside
    % gavdNetInference, in minibatch-sized chunks - this keeps VRAM bounded
    % to the active minibatch on small GPUs (e.g. T550, 4 GB).

    % If raw results have been saved, load them and skip inference. Else
    % run (or resume) inference now. If a partial raw cache exists from a
    % crashed prior run AND its featureFraming / frameStandardization /
    % channel-1 file list match the current run, resume from the next
    % un-processed file.
    if exist(saveNamePathRaw, 'file')
        fprintf('Found cached raw results for %d. Loading... \n', year)
        load(saveNamePathRaw)
    else
        % Initialise inference state - results array and starting file index.
        results = struct([]);
        startFileIdx = 1;

        % Attempt to resume from a partial cache if one exists.
        if exist(saveNamePathPartial, 'file')
            try
                partial = load(saveNamePathPartial);
                currentChFileNames = {chFiles.name};
                settingsMatch = isequaln(partial.featureFraming, featureFraming) && ...
                    isequaln(partial.frameStandardization, frameStandardization);
                filesMatch = isequal(numel(partial.chFileNames), numel(currentChFileNames)) && ...
                    all(strcmp(partial.chFileNames(:), currentChFileNames(:)));
                if settingsMatch && filesMatch
                    results = partial.results;
                    startFileIdx = partial.nextFileIdx;
                    fprintf(['Resuming year %d from file %d/%d (loaded %d ' ...
                        'cached entries from partial cache).\n'], ...
                        year, startFileIdx, numel(chFilePaths), numel(results))
                else
                    warning(['Partial cache %s does not match current ' ...
                        'settings / file list. Starting year from file 1.'], ...
                        saveNamePathPartial)
                end
                clear partial
            catch ME
                warning(['Could not load partial cache %s: %s. ' ...
                    'Starting year from file 1.'], ...
                    saveNamePathPartial, ME.message)
            end
        end

        % Year-start GPU health check (sentinel + small throughput probe).
        % Skipped if we are already on CPU (no GPU to probe). A failed
        % sentinel forces an immediate fallback step; a marginal link is
        % logged as a warning but does not trigger fallback on its own.
        if useGPU
            try
                [hcStatus, hcMetrics] = gpuHealthCheck(gpuDeviceID, healthCheckThroughputMin);
            catch ME
                warning('gpuHealthCheck threw: %s. Treating GPU as failed.', ME.message)
                hcStatus = 'failed';
                hcMetrics = struct( ...
                    'deviceName', sprintf('device %d', gpuDeviceID), ...
                    'availableMemoryBytes', NaN, ...
                    'sentinelOK', false, 'sentinelErr', ME.message, ...
                    'h2dMean', NaN, 'h2dStd', NaN, 'h2dMin', NaN, ...
                    'd2hMean', NaN, 'd2hStd', NaN, 'd2hMin', NaN);
            end

            fprintf('GPU health (device %d "%s"): %s\n', ...
                gpuDeviceID, hcMetrics.deviceName, hcStatus)
            if ~isnan(hcMetrics.availableMemoryBytes)
                fprintf('\tAvailable memory: %.2f GB\n', hcMetrics.availableMemoryBytes / 1e9)
            end
            if hcMetrics.sentinelOK
                fprintf('\tSentinel: OK\n')
            else
                fprintf('\tSentinel: FAILED (%s)\n', hcMetrics.sentinelErr)
            end
            if ~isnan(hcMetrics.h2dMean)
                fprintf('\tH->D throughput: mean %.2f GB/s, std %.2f, min %.2f\n', ...
                    hcMetrics.h2dMean, hcMetrics.h2dStd, hcMetrics.h2dMin)
                fprintf('\tD->H throughput: mean %.2f GB/s, std %.2f, min %.2f\n', ...
                    hcMetrics.d2hMean, hcMetrics.d2hStd, hcMetrics.d2hMin)
            end

            switch hcStatus
                case 'healthy'
                    % no action required
                case 'marginal'
                    warning(['GPU link is marginal (mean throughput < %.2f ' ...
                        'GB/s). Inference may experience link drops.'], ...
                        healthCheckThroughputMin)
                case 'failed'
                    warning(['GPU health check FAILED for device %d at ' ...
                        'start of year %d. Stepping fallback chain.'], ...
                        gpuDeviceID, year)
                    [useGPU, gpuDeviceID, bytesAvailable, currentGpuChainIdx, switchedTo] = ...
                        stepGpuFallback(currentGpuChainIdx, gpuChain, cpuMemoryBytes);
                    fprintf('Now using: %s\n', switchedTo)
            end
        else
            fprintf('Running year %d on CPU (no GPU health check).\n', year)
        end

        % Per-year consecutive-failure counter for fallback decision.
        % Resets at year start so a fallback from a prior year does not
        % carry over to the new device.
        consecGpuFailures = 0;

        % Run inference on each remaining un-processed file.
        for fileIdx = startFileIdx:numel(chFilePaths)
            filePath = chFilePaths{fileIdx};
            [~, fileName, fileExt] = fileparts(filePath);

            % Announce start
            fprintf('Running inference on file %d of %d...\n', fileIdx, numel(chFilePaths))

            % Clear GPU memory from previous iteration and refresh the
            % free-memory snapshot. The initial bytesAvailable from
            % gpuConfig() can drift over a long run as other processes
            % consume VRAM (compositor, browser, etc.); on small GPUs an
            % out-of-date estimate can cause estimateInferenceMinibatchSize
            % to pick a batch that no longer fits. Per-file VRAM is logged
            % so leaks / fragmentation trends are visible in the diary.
            % If the reset itself errors, treat the device as failed and
            % step the fallback chain immediately.
            if useGPU
                try
                    wait(gpuDevice(gpuDeviceID));
                    reset(gpuDevice(gpuDeviceID));
                    bytesAvailable = gpuDevice(gpuDeviceID).AvailableMemory;
                    fprintf('\tGPU available memory: %.2f GB\n', bytesAvailable / 1e9)
                catch ME
                    warning(['Per-file GPU reset failed: %s. Stepping ' ...
                        'fallback chain.'], ME.message)
                    [useGPU, gpuDeviceID, bytesAvailable, currentGpuChainIdx, switchedTo] = ...
                        stepGpuFallback(currentGpuChainIdx, gpuChain, cpuMemoryBytes);
                    fprintf('\tNow using: %s\n', switchedTo)
                    consecGpuFailures = 0;
                end
            end

            % Read audio file (retries with 60/180/300 s waits, then 300 s cooldown).
            fprintf('\tReading audio...\n')
            [audioIn, sampleRate, readErr] = audioReadWithRetry(filePath);
            results(fileIdx).fileName = [fileName, fileExt];
            if isempty(audioIn)
                warning('\tCould not read file %d (%s%s) after retries: %s. Skipping...\n', ...
                    fileIdx, fileName, fileExt, readErr)
                results(fileIdx).failComment = sprintf( ...
                    'Read failed after retries: %s', readErr);
                if mod(fileIdx, partialCacheEveryN) == 0
                    saveResultsToPartialCache(saveNamePathPartial, results, fileIdx, ...
                        featureFraming, frameStandardization, {chFiles.name});
                end
                continue
            end

            % Write filename, Fs and file info to detections struct
            results(fileIdx).fileFs = sampleRate;
            results(fileIdx).fileSamps = length(audioIn);
            results(fileIdx).fileDuration = results(fileIdx).fileSamps / results(fileIdx).fileFs;
            results(fileIdx).probabilities = [];

            % Extract datetime from filename
            fprintf('\tExtracting datetime stamp from audio filename...\n')
            results(fileIdx).fileStartDateTime = extractDatetimeFromFilename(filePath, 'datetime');

            % Skip this file if it's name doesn't contain a valid start date
            if isempty(results(fileIdx).fileStartDateTime) ||...
                    isnat(results(fileIdx).fileStartDateTime)
                warning('\tCould not extract datetime from filename: %s. Skipping...\n',...
                    results(fileIdx).fileName)
                results(fileIdx).failComment = 'Could not read valid recording start data-time from filename';
                if mod(fileIdx, partialCacheEveryN) == 0
                    saveResultsToPartialCache(saveNamePathPartial, results, fileIdx, ...
                        featureFraming, frameStandardization, {chFiles.name});
                end
                continue
            end

            % Skip this file if it doesn't contain valid audio
            if isValidAudio(audioIn) == false
                warning('\tFile %s did not contain valid audio. Skipping...\n', results(fileIdx).fileName)
                results(fileIdx).failComment = 'Could not read valid audio from file';
                if mod(fileIdx, partialCacheEveryN) == 0
                    saveResultsToPartialCache(saveNamePathPartial, results, fileIdx, ...
                        featureFraming, frameStandardization, {chFiles.name});
                end
                continue
            end

            % NOTE: previously a sample-domain datetime vector was
            % constructed and stored on every results entry; for 4-hour
            % files at 250 Hz that was ~58 MB per file, which accumulated
            % to >100 GB across a year and caused host-RAM OOMs in long
            % runs. Event datetimes are now derived in postprocessing
            % directly from fileStartDateTime + sample index / fileFs,
            % which only needs the three scalars already stored above.

            % Run preprocessing and inference with retry on transient GPU
            % errors. Each retry resets the GPU and pauses briefly before
            % the next attempt. Files that fail all retries are marked
            % with failComment; the consecutive-failure tracker below
            % decides whether to fall back to the next device.
            inferenceSucceeded = false;
            lastInferenceErr = '';
            execTime = NaN;
            numAudioSegments = NaN;
            for attempt = 1:maxInferenceRetries
                try
                    [results(fileIdx).probabilities, ~, execTime, ...
                        results(fileIdx).silenceMask, numAudioSegments] = gavdNetInference(...
                        audioIn, sampleRate, model, bytesAvailable, ...
                        featureFraming, frameStandardization, minSilenceDuration, plotting);
                    inferenceSucceeded = true;
                    consecGpuFailures = 0;
                    break
                catch ME
                    lastInferenceErr = ME.message;
                    warning('Inference attempt %d/%d failed for %s: %s', ...
                        attempt, maxInferenceRetries, [fileName, fileExt], ME.message)
                    if useGPU
                        try
                            wait(gpuDevice(gpuDeviceID));
                            reset(gpuDevice(gpuDeviceID));
                            bytesAvailable = gpuDevice(gpuDeviceID).AvailableMemory;
                        catch resetME
                            warning('\tGPU reset between inference retries also failed: %s', ...
                                resetME.message)
                        end
                    end
                    if attempt < maxInferenceRetries
                        pause(10);
                    end
                end
            end

            if ~inferenceSucceeded
                fprintf(['\tInference FAILED after %d attempts. ' ...
                    'Marking file as failed.\n'], maxInferenceRetries)
                results(fileIdx).failComment = sprintf( ...
                    'Inference failed after %d retries: %s', ...
                    maxInferenceRetries, lastInferenceErr);
                results(fileIdx).probabilities = [];

                if useGPU
                    consecGpuFailures = consecGpuFailures + 1;
                    threshold = currentFailureThreshold(currentGpuChainIdx, maxConsecGpuFailures);
                    if consecGpuFailures >= threshold
                        warning(['Inference failed for %d consecutive ' ...
                            'file(s) on the current device. Stepping ' ...
                            'fallback chain.'], consecGpuFailures)
                        [useGPU, gpuDeviceID, bytesAvailable, currentGpuChainIdx, switchedTo] = ...
                            stepGpuFallback(currentGpuChainIdx, gpuChain, cpuMemoryBytes);
                        fprintf('\tNow using: %s\n', switchedTo)
                        consecGpuFailures = 0;
                    end
                end

                if mod(fileIdx, partialCacheEveryN) == 0
                    saveResultsToPartialCache(saveNamePathPartial, results, fileIdx, ...
                        featureFraming, frameStandardization, {chFiles.name});
                end
                continue
            end

            % Report execution time and seconds of audio with high probability
            numTimeBinsProbHigh = sum(results(fileIdx).probabilities > postProcOptions.AT);
            secondsBinsProbHigh = numTimeBinsProbHigh * windowDur;
            fprintf('\tInference Completed in %.2f seconds\n', execTime)
            fprintf('\tTotal audio duration: %.2f seconds\n', results(fileIdx).fileDuration)
            fprintf('\tDuration with raw detection probability > Activation Threshold%%: %.2f seconds.\n\n', secondsBinsProbHigh)

            % Write diagnostic information to the results struct
            results(fileIdx).probsAllNan = all(isnan(results(fileIdx).probabilities));
            results(fileIdx).probsAnyNan = any(isnan(results(fileIdx).probabilities));
            results(fileIdx).audioAllSilence = all(results(fileIdx).silenceMask == true);
            results(fileIdx).audioAnySilence = any(results(fileIdx).silenceMask == true);
            results(fileIdx).numSplitAudioSegments = numAudioSegments;

            % Per-file partial cache, throttled to every partialCacheEveryN
            % files (see USER INPUT). Writing every file is too expensive
            % once the results array grows past a few tens of MB; throttling
            % bounds worst-case I/O without giving up crash recovery entirely.
            if mod(fileIdx, partialCacheEveryN) == 0
                saveResultsToPartialCache(saveNamePathPartial, results, fileIdx, ...
                    featureFraming, frameStandardization, {chFiles.name});
            end
        end

        % Save the output
        save(saveNamePathRaw, 'results', '-v7.3')
        fprintf('Year %d: saved %d raw results to %s\n', year, length(results), saveNamePathRaw)

        % Delete the partial cache now that the full raw cache is safely on disk.
        if exist(saveNamePathPartial, 'file')
            try
                delete(saveNamePathPartial)
            catch ME
                warning('Could not delete partial cache %s: %s', ...
                    saveNamePathPartial, ME.message)
            end
        end
    end

    %% Reload and post-process the raw predictions to get detections and confidence scores.

    fprintf('Postprocesing model outputs...\n')
    for i = 1:length(results)

        % Defensive skip for entries that the inference loop could not
        % fully process. Possible failure modes:
        %   (a) audioReadWithRetry returned empty -> fileName set, no probabilities
        %   (b) datetime parse failed -> fileName set, probabilities = []
        %   (c) isValidAudio rejected the file -> fileName set, probabilities = []
        %   (d) legacy raw .mat from before the audioReadWithRetry refactor
        %       may also have entries with no fileName at all
        % In any of these cases there is nothing to postprocess; mark zero
        % detections so flattenDetections handles the entry gracefully.
        if (~isfield(results, 'fileName') || isempty(results(i).fileName)) || ...
                (~isfield(results, 'probabilities') || isempty(results(i).probabilities))
            results(i).eventSampleBoundaries = [];
            results(i).confidence = [];
            results(i).nDetections = 0;
            fprintf('File %g: skipped (no valid inference output)\n', i)
            continue
        end

        % Get audio for this file. Wrap audioread to handle the case where
        % the source file has moved or been deleted between the inference
        % run and a later postprocessing-only re-run.
        try
            [audioIn, fileFs] = audioread(fullfile(inferenceAudioPath, results(i).fileName));
        catch ME
            warning('File %g (%s): cannot read audio (%s). Skipping.', ...
                i, results(i).fileName, ME.message)
            results(i).eventSampleBoundaries = [];
            results(i).confidence = [];
            results(i).nDetections = 0;
            continue
        end

        % Run postprocessing to determine decision boundaries.
        [results(i).eventSampleBoundaries, ~, ...
            results(i).confidence] = gavdNetPostprocess(...
            audioIn, fileFs, results(i).probabilities, model.preprocParams, ...
            postProcOptions);

        % Get number of detections
        results(i).nDetections = size(results(i).eventSampleBoundaries, 1);

        % Get the datetime start and end times for each detected event using
        if ~isempty(results(i).eventSampleBoundaries)
            for detIdx = 1:results(i).nDetections

                % Get event boundaries (as sample indices)
                eventStart = results(i).eventSampleBoundaries(detIdx, 1);
                eventEnd = results(i).eventSampleBoundaries(detIdx, 2);

                % Convert sample indices to datetime relative to file start.
                % Computed on-the-fly from the file's start datetime and
                % sample rate (no per-sample datetime vector materialised).
                results(i).eventTimesDT(detIdx, 1) = results(i).fileStartDateTime + ...
                    seconds((eventStart - 1) / results(i).fileFs);
                results(i).eventTimesDT(detIdx, 2) = results(i).fileStartDateTime + ...
                    seconds((eventEnd - 1) / results(i).fileFs);
            end
        end
         fprintf('File %g: %g events detected\n', i, results(i).nDetections)
    end

    % Detections are one row per audio file, potentially multiple detections per row.
    % Flatten detections to one row per detection.
    results = flattenDetections(results, model.preprocParams);

    %% Save the output
    save(saveNamePathPost, 'results', 'featureFraming', 'frameStandardization', 'postProcOptions', '-v7.3')

    fprintf('Year %d: saved %d post processed detections to %s\n', year, length(results), saveNamePathPost)
    diary off

    % Per-year cleanup. CRITICAL: keep model, postProcOptions, gpuConfig
    % outputs, eGPU-mitigation state (gpuChain, currentGpuChainIdx,
    % cpuMemoryBytes, retry/fallback thresholds) and config-file variables
    % loaded across years - do NOT add `clear gavdNetPreprocess` or
    % `clear persistent` here (see comment above the year loop for the
    % rationale).
    clearvars -except plotting configPath gavdNetDataPath audioRoot ...
        channelPrefix inferenceOutputPath projectRoot gitRoot ...
        model postProcOptions useGPU gpuDeviceID bytesAvailable ...
        gpuChain currentGpuChainIdx cpuMemoryBytes ...
        maxInferenceRetries maxConsecGpuFailures healthCheckThroughputMin ...
        partialCacheDir partialCacheEveryN ...
        years yearIdx ...
        featureFraming frameStandardization minSilenceDuration windowDur

end

%% Local functions (eGPU stability mitigations)

function saveResultsToPartialCache(savePath, results, lastCompletedIdx, ...
        featureFraming, frameStandardization, chFileNames)
% Write the per-file partial cache used by the resume-on-restart logic.
%
% Inputs (mirroring the script's loop-local variables):
%   savePath            - Full path to detector_raw_partial_<year>.mat
%   results             - Current results struct array
%   lastCompletedIdx    - Index of the file whose iteration just ended.
%                         nextFileIdx in the cache is set to this + 1.
%   featureFraming      - To verify settings on resume
%   frameStandardization - To verify settings on resume
%   chFileNames         - cellstr {chFiles.name} for this year, used on
%                         resume to ensure the file list hasn't changed
%
% The partial cache is rewritten after every iteration (including ones
% that ended in `continue` due to read / datetime / inference failure)
% so a crash never loses more than the current file's work.
    partialCache.results = results;
    partialCache.nextFileIdx = lastCompletedIdx + 1;
    partialCache.featureFraming = featureFraming;
    partialCache.frameStandardization = frameStandardization;
    partialCache.chFileNames = chFileNames;
    try
        save(savePath, '-struct', 'partialCache', '-v7.3');
    catch ME
        warning('Could not write partial cache to %s: %s', savePath, ME.message);
    end
end


function [useGPU, gpuDeviceID, bytesAvailable, newChainIdx, switchedTo] = ...
        stepGpuFallback(currentChainIdx, gpuChain, cpuMemoryBytes)
% Advance one step down the GPU fallback chain after the current device
% is judged to be failing (e.g. consecGpuFailures >= threshold, or the
% year-start health check returned 'failed').
%
% currentChainIdx convention:
%   1..N  - index into gpuChain (entries sorted by TotalMemory desc, so 1
%           is the primary / most-capable GPU)
%   0     - currently on CPU; no further fallback is possible
%
% Transitions:
%   1..N-1  -> next GPU in chain (try to reset and activate it; if that
%              also throws, skip straight to CPU)
%   N       -> CPU (chain exhausted)
%   0       -> remains CPU
%
% Returns the new (useGPU, gpuDeviceID, bytesAvailable, newChainIdx) plus
% a human-readable description of where we ended up, for logging.

    if currentChainIdx == 0
        useGPU = false;
        gpuDeviceID = 0;
        bytesAvailable = cpuMemoryBytes;
        newChainIdx = 0;
        switchedTo = 'CPU (already on CPU; no further fallback)';
        return
    end

    % Best-effort release of the failing device's context. The device may
    % already be unresponsive, which is exactly the situation that
    % triggered the fallback, so swallow any error here.
    try
        wait(gpuDevice(gpuChain(currentChainIdx).deviceID));
        reset(gpuDevice(gpuChain(currentChainIdx).deviceID));
    catch
        % Ignore - device unresponsive
    end

    if currentChainIdx < numel(gpuChain)
        % Step to next GPU in chain
        newChainIdx = currentChainIdx + 1;
        candidateID = gpuChain(newChainIdx).deviceID;
        try
            g = gpuDevice(candidateID);
            reset(g);
            useGPU = true;
            gpuDeviceID = candidateID;
            bytesAvailable = g.AvailableMemory;
            switchedTo = sprintf('GPU %d ("%s", %.1f GB free)', ...
                gpuDeviceID, char(g.Name), bytesAvailable / 1e9);
        catch ME
            warning(['Could not activate fallback GPU %d: %s. ' ...
                'Skipping to CPU.'], candidateID, ME.message)
            useGPU = false;
            gpuDeviceID = 0;
            bytesAvailable = cpuMemoryBytes;
            newChainIdx = 0;
            switchedTo = 'CPU (fallback GPU also unavailable)';
        end
    else
        % Chain exhausted - drop to CPU
        useGPU = false;
        gpuDeviceID = 0;
        bytesAvailable = cpuMemoryBytes;
        newChainIdx = 0;
        switchedTo = sprintf('CPU (%.1f GB available)', cpuMemoryBytes / 1e9);
    end
end


function threshold = currentFailureThreshold(currentChainIdx, primaryThreshold)
% Per-device consecutive-failure threshold before triggering fallback.
% The primary GPU (chain idx 1) gets the user-configurable threshold; any
% non-primary GPU gets a single shot before being dropped, per the
% report's "RTX 4090 -> T550 -> CPU, T550 fails on first file -> CPU"
% fallback policy.
    if currentChainIdx == 1
        threshold = primaryThreshold;
    else
        threshold = 1;
    end
end
