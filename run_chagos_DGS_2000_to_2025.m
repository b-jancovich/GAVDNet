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
%   %TEMP%\GAVDNet\detector_raw_partial_<year>_part<K>.mat + _manifest.mat
%                                                   (sharded periodic checkpoint
%                                                    on local SSD, deleted on
%                                                    year completion; _gpuA /
%                                                    _gpuB variants in dual-GPU
%                                                    mode)
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
%   * Else, if a partial cache for the year exists in %TEMP%\GAVDNet\ AND its
%     saved featureFraming / frameStandardization / channel-1 file list match
%     the current run, inference resumes from the next un-processed file. The
%     cache is sharded (detector_raw_partial_<year>_part<K>.mat plus a small
%     _manifest.mat) and flushed every partialCacheEveryN files (default 250);
%     only the current shard is rewritten each flush. A legacy single-file
%     detector_raw_partial_<year>.mat (e.g. an in-flight pre-sharding cache) is
%     read and migrated to the sharded format on first resume. Mismatched or
%     incomplete caches are discarded with a warning. In dual-GPU mode each
%     worker keeps its own _gpuA / _gpuB cache and resumes independently. All
%     caches are deleted once the full raw .mat is saved.
% To force a year's inference to re-run, delete its raw .mat as well.
%
% eGPU stability mitigations (Thunderbolt RTX 4090 + internal T550 fallback):
%   * Sharded periodic partial cache (every partialCacheEveryN files,
%     default 250) written to local SSD under %TEMP%\GAVDNet\. Only the
%     current shard is rewritten per flush, so checkpoint cost stays bounded
%     no matter how far into the year the run is. Worst case crash loss is
%     partialCacheEveryN - 1 files; the cache lives off OneDrive so the
%     per-flush write does not incur sync-agent stalls.
%   * The GPU context reset (wait+reset) is throttled to every gpuResetEveryN
%     files (default 50) rather than every file - a full context reset over
%     the Thunderbolt link is expensive. The cheap per-file AvailableMemory
%     read still runs every file, and the failure paths still reset on error.
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
% Optional speed features (see the USER INPUT block and DUALGPU_NOTES.md):
%   * enableShortFileSkip - skip, before reading, files too short to contain a
%     detection that survives postprocessing (default off; a scientifically-
%     neutral cleanup for truncated / degenerate files).
%   * enableDualGpu - process each year's remaining files concurrently on two
%     GPUs, routed by length (largest files to the primary, smallest to the
%     secondary; see gpuPrimaryFileFraction), each worker pinned to a distinct
%     GPU with its own cache and merged at the end (default off; the serial
%     single-GPU path is unchanged and is the tested fallback).
%
% This script assumes a single MATLAB instance owns each GPU at any one time.
% A second CUDA context on the SAME GPU has been observed to destabilise the
% eGPU, so the dual-GPU mode pins each worker to a DISTINCT GPU and releases
% the client's GPU context first. Do not launch a second MATLAB that also
% uses these GPUs.
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

% Partial cache is flushed to disk every this many files. The cache lives
% in %TEMP%\GAVDNet\ (local SSD) rather than inferenceOutputPath, so
% OneDrive sync never sees it. Worst case crash loss = partialCacheEveryN
% - 1 files. The full raw cache is always written at year completion
% regardless of this setting.
partialCacheEveryN = 250;

% The partial cache is stored as fixed-size shards
% (detector_raw_partial_<year>_part<K>.mat) plus a small manifest, so each
% checkpoint only rewrites the current shard (<= cacheShardSize entries)
% instead of the whole, ever-growing results array. This bounds checkpoint
% write cost to O(cacheShardSize) rather than O(files processed so far).
% cacheShardSize is snapped up to a whole multiple of partialCacheEveryN at
% setup so every shard boundary coincides with a checkpoint. A legacy
% single-file detector_raw_partial_<year>.mat (e.g. an in-flight 2006
% cache) is still read on resume for backward compatibility.
cacheShardSize = 2000;

% A full wait+reset(gpuDevice) tears down and rebuilds the CUDA context to
% clear accumulated VRAM fragmentation / leaks. Over the Thunderbolt eGPU
% link this is expensive, so it is throttled to run once every this many
% files (a count of files, not a duration), plus on the first file of the
% run and immediately after any inference failure. The cheap per-file
% AvailableMemory read is unaffected and still runs every file. Set to 1 to
% restore the original "reset before every file" behaviour. Set to 50 (from
% an earlier 250) to refresh the Thunderbolt eGPU link more often, hedging
% the observed upward drift in per-file inference time on long sustained
% runs while still saving ~98% of the old every-file reset cost.
gpuResetEveryN = 50;

% --- Short-file skip (optional, default OFF) ---
% When enabled, an audio file whose duration is below the skip threshold is
% marked as skipped and NOT read or run through the model. This avoids the
% per-file overhead on files that are provably too short to yield any
% detection. It is a robustness / cleanup feature for truncated or
% degenerate files - it does NOT meaningfully speed up years made of many
% ordinary short recordings, whose durations exceed the threshold.
% DEFAULT OFF so production behaviour is unchanged unless explicitly opted
% into. ENABLED for this production run: it is scientifically neutral (a file
% shorter than LT cannot yield a detection that survives postprocessing) and
% it also avoids the sub-second files that otherwise fail inference 3x with
% two 10 s retry pauses each. Set back to false to restore the old behaviour.
enableShortFileSkip = true;

% Skip threshold in seconds. Leave EMPTY ([]) to auto-derive the
% scientifically-safe value = postProcOptions.LT (the postprocessing length
% threshold, = meanTargetCallDuration * LT_scaler). A file shorter than LT
% cannot produce a detection that survives postprocessing, regardless of
% content, so skipping it changes no scientific result. WARNING: setting an
% explicit value GREATER than LT risks silently discarding files that could
% contain a real, detectable call - keep any override <= LT.
shortFileSkipDurationSec = [];

% --- Dual-GPU concurrent inference (optional; default OFF) ---
% When enabled AND at least two GPUs are available, a year's remaining files
% are split into two contiguous ranges processed CONCURRENTLY by two parallel
% workers, each pinned to a distinct GPU (the two largest in the fallback
% chain, e.g. RTX 4090 for the primary share and T550 for the secondary).
% Each worker runs the normal per-file pipeline over its range into its own
% worker-scoped partial cache (detector_raw_partial_<year>_gpuA/_gpuB), and
% the two result ranges are merged with the preloaded/cached results before
% postprocessing. The single-GPU serial path is used (unchanged) when this is
% false or fewer than two GPUs are present.
% IMPORTANT (eGPU): each worker holds a CUDA context on its OWN GPU. Validate
% stability on a small run first - a second context on the Thunderbolt 4090
% has previously destabilised the link. If unstable, set this false to use
% the tested serial path.
enableDualGpu = true;

% Fraction of a year's REMAINING files assigned to the primary GPU
% (gpuChain(1), the largest / fastest). The remaining files are routed by
% LENGTH: sorted by size (bytes, a proxy for duration), the primary takes the
% largest primaryFileFraction of them and the secondary takes the smallest
% rest. This keeps long single-segment files on the high-memory primary (full
% batch size) and gives the low-memory secondary the short files it processes
% efficiently, while still handing it some files in a mostly-long year. A
% scalar in (0,1); e.g. 0.6 sends the largest 60% of the remaining files to
% the primary and the smallest 40% to the secondary. The split is clamped so
% the secondary always gets at least one (the shortest) file. Because the
% split is derived only from the fixed file sizes and this fraction, the two
% workers' ranges are identical across restarts, so each worker resumes its
% own cache correctly. Tune from the per-worker throughput logged at year end.
% Only used when enableDualGpu is true.
gpuPrimaryFileFraction = 0.6;

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

% Resolve the short-file skip threshold (A3, see USER INPUT). Auto value is
% postProcOptions.LT: a file shorter than LT cannot yield a detection that
% survives postprocessing. Computed once here because model.LT_scaler and
% meanTargetCallDuration are constant for the whole run. Held across years
% by the clearvars -except allowlist at the end of the year loop.
if isempty(shortFileSkipDurationSec)
    shortFileSkipThreshSec = postProcOptions.LT;
else
    shortFileSkipThreshSec = shortFileSkipDurationSec;
end
if enableShortFileSkip
    fprintf('Short-file skip ENABLED: files shorter than %.3f s will be skipped before read.\n', ...
        shortFileSkipThreshSec)
else
    fprintf('Short-file skip disabled (all files processed regardless of duration).\n')
end

% Snap cacheShardSize up to a whole multiple of partialCacheEveryN so every
% shard boundary lands on a checkpoint (the sharded-cache writer relies on
% this to finalise each completed shard exactly once). Held across years by
% the clearvars -except allowlist.
cacheShardSize = max(partialCacheEveryN, ...
    ceil(cacheShardSize / partialCacheEveryN) * partialCacheEveryN);
fprintf('Partial cache shard size: %d files (checkpoint every %d files).\n', ...
    cacheShardSize, partialCacheEveryN)

% Set up for GPU or CPU processing (one device for the whole run):
[useGPU, gpuDeviceID, ~, bytesAvailable] = gpuConfig();

% Build the GPU fallback chain. Entries are sorted by TotalMemory
% descending so chain(1) is the largest / primary GPU (RTX 4090 eGPU on
% this machine) and chain(end) is the smallest (T550 internal). The
% Functions/stepGpuFallback helper walks this chain on consecutive inference
% failures, eventually dropping to CPU.
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

years = 2001:2018;

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

        % Attempt to resume from a partial cache if one exists. Uses the
        % sharded format (manifest + part files); falls back to a legacy
        % single-file cache (e.g. an in-flight 2006
        % detector_raw_partial_2006.mat) for backward compatibility. Any
        % settings / file-list mismatch, missing shard, or unreadable file is
        % reported inside the loader, which then returns isValidResume =
        % false so the year restarts from file 1.
        [resumedResults, resumedStartIdx, isValidResume] = ...
            loadResultsFromShardedCache(saveNamePathPartial, cacheShardSize, ...
            featureFraming, frameStandardization, {chFiles.name});
        if isValidResume
            results = resumedResults;
            startFileIdx = resumedStartIdx;
            fprintf(['Resuming year %d from file %d/%d (loaded %d cached ' ...
                'entries from partial cache).\n'], ...
                year, startFileIdx, numel(chFilePaths), numel(results))
        end
        clear resumedResults

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

        % Assemble the option and device-state structs passed to the shared
        % per-file inference loop (Functions/runInferenceFileLoop), used by
        % both the serial path and each dual-GPU worker. This replaces the
        % previously-inline per-file loop; the logic is unchanged.
        opts = struct( ...
            'featureFraming', featureFraming, ...
            'frameStandardization', frameStandardization, ...
            'minSilenceDuration', minSilenceDuration, ...
            'plotting', plotting, ...
            'activationThreshold', postProcOptions.AT, ...
            'windowDur', windowDur, ...
            'maxInferenceRetries', maxInferenceRetries, ...
            'maxConsecGpuFailures', maxConsecGpuFailures, ...
            'gpuResetEveryN', gpuResetEveryN, ...
            'enableShortFileSkip', enableShortFileSkip, ...
            'shortFileSkipThreshSec', shortFileSkipThreshSec, ...
            'partialCacheEveryN', partialCacheEveryN, ...
            'cacheShardSize', cacheShardSize, ...
            'progressLabel', '');

        % Device state (fallback chain, current device, counters). The
        % per-year consecutive-failure counter resets at year start so a
        % fallback from a prior year does not carry over to the new device.
        deviceState = struct( ...
            'useGPU', useGPU, ...
            'gpuDeviceID', gpuDeviceID, ...
            'bytesAvailable', bytesAvailable, ...
            'gpuChain', gpuChain, ...
            'currentGpuChainIdx', currentGpuChainIdx, ...
            'cpuMemoryBytes', cpuMemoryBytes, ...
            'consecGpuFailures', 0);

        % Choose serial (single GPU / CPU) or dual-GPU concurrent inference.
        numFilesThisYear = numel(chFilePaths);
        numRemaining = numFilesThisYear - startFileIdx + 1;
        useDualGpu = enableDualGpu && useGPU && numel(gpuChain) >= 2 && numRemaining >= 2;

        if useDualGpu
            % Route the REMAINING files between the two GPUs by LENGTH: sorted
            % by size (bytes, a proxy for duration), the largest
            % gpuPrimaryFileFraction of the files go to the primary GPU and the
            % smallest rest to the secondary. This keeps long single-segment
            % files on the high-memory primary (full batch size) and gives the
            % low-memory secondary the short files, while still handing it some
            % files to do in a mostly-long year. The split depends only on the
            % fixed file sizes and the fraction, so the two workers' ranges are
            % identical across restarts and each worker resumes its own
            % worker-scoped cache. runYearDualGpu runs the two workers
            % concurrently (each pinned to a distinct GPU) and returns the
            % merged global results.
            remainingSizes = [chFiles(startFileIdx:numFilesThisYear).bytes];
            remainingGlobalIdx = startFileIdx:numFilesThisYear;
            [aRange, bRange, splitInfo] = planLengthRoutedSplit(...
                remainingSizes, remainingGlobalIdx, gpuPrimaryFileFraction);
            fprintf(['Length-routed split: primary (%s) gets %d files / %.2f GB ' ...
                '(largest files); secondary (%s) gets %d files / %.2f GB ' ...
                '(smallest files).\n'], ...
                gpuChain(1).Name, splitInfo.nPrimary, splitInfo.bytesPrimary / 1e9, ...
                gpuChain(2).Name, splitInfo.nSecondary, splitInfo.bytesSecondary / 1e9)

            results = runYearDualGpu(chFilePaths, {chFiles.name}, aRange, bRange, ...
                results, model, opts, gpuChain, cpuMemoryBytes, saveNamePathPartial);
        else
            % Serial single-device path: process the whole remaining list on
            % the current device, resuming from startFileIdx.
            [results, deviceState] = runInferenceFileLoop(chFilePaths, ...
                {chFiles.name}, startFileIdx, results, model, opts, ...
                deviceState, saveNamePathPartial);
            % Persist any device fallback that happened this year into the
            % run-level device variables (kept across years by clearvars).
            useGPU             = deviceState.useGPU;
            gpuDeviceID        = deviceState.gpuDeviceID;
            bytesAvailable     = deviceState.bytesAvailable;
            currentGpuChainIdx = deviceState.currentGpuChainIdx;
        end

        % Save the output
        save(saveNamePathRaw, 'results', '-v7.3')
        fprintf('Year %d: saved %d raw results to %s\n', year, length(results), saveNamePathRaw)

        % Delete the partial cache(s) now that the full raw cache is safely
        % on disk - the serial cache (manifest, every shard, and any legacy
        % single file) plus the two dual-GPU worker caches if they exist.
        deletePartialCache(saveNamePathPartial)
        [pcDir, pcName, ~] = fileparts(saveNamePathPartial);
        deletePartialCache(fullfile(pcDir, [pcName, '_gpuA.mat']))
        deletePartialCache(fullfile(pcDir, [pcName, '_gpuB.mat']))
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
        partialCacheDir partialCacheEveryN gpuResetEveryN cacheShardSize ...
        enableShortFileSkip shortFileSkipDurationSec shortFileSkipThreshSec ...
        enableDualGpu gpuPrimaryFileFraction ...
        years yearIdx ...
        featureFraming frameStandardization minSilenceDuration windowDur

end
