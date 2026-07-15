function profile_inference_stages(numFilesToProfile)
% PROFILE_INFERENCE_STAGES  Per-file timing breakdown for the Chagos DGS
% production inference pipeline, used to (a) confirm the A1 per-file
% GPU-reset saving and (b) decide whether dual-GPU inference (Phase B) is
% worth building or the run is I/O-bound.
%
% Runs the REAL model over the first numFilesToProfile channel-1 files of a
% target year and measures, per file:
%   tReset - wait+reset(gpuDevice)          (the cost A1 stops paying every file)
%   tRead  - audioReadWithRetry             (external-drive I/O)
%   tGPU   - gavdNetInference's execTime    (minibatchpredict only)
%   tCPU   - inference wall time minus tGPU (silence + event-split + mel STFT)
%
% Usage:
%   profile_inference_stages          % default 300 files
%   profile_inference_stages(150)
%
% NOTE: run this in the SAME MATLAB instance that owns the eGPU (do not open
% a second MATLAB that also creates a CUDA context on the RTX 4090 - the
% run-script header warns this destabilises the link). It uses the GPU and
% the E: drive and will take several minutes.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

if nargin < 1 || isempty(numFilesToProfile)
    numFilesToProfile = 300;
end

% ---- Configuration (mirrors run_chagos_DGS_2000_to_2025.m USER INPUT) ----
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos_exclude_chorus.m";
gavdNetDataPath = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar Exclude Chorus";
audioRoot = "E:\Diego Garcia South 3Ch";
channelPrefix = "H08S1";
year = 2006;   % the year that stalled - most fragmented

% ---- One-time setup (mirror the production script) ----
userGavdNetDataPath = gavdNetDataPath;
run(configPath)
% Repo root is the parent of this script's folder ("Utilitiy Scripts"), NOT
% pwd (run_matlab_file sets pwd to the script's own folder).
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(fullfile(projectRoot, "Functions"))
gavdNetDataPath = userGavdNetDataPath;

modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
if isempty(modelList)
    error('No GAVDNet_trained_* model found in %s', gavdNetDataPath)
end
load(fullfile(modelList(1).folder, modelList(1).name))  % loads `model`
fprintf('Loaded model: %s\n', modelList(1).name)

% GPU device (whichever is currently active; report which).
if gpuDeviceCount("available") < 1
    error('No GPU available - this profiler targets the GPU inference path.')
end
g = gpuDevice;
fprintf('Profiling on GPU: %s (%.1f GB total)\n', char(g.Name), g.TotalMemory / 1e9)

% ---- Build the file list ----
yearDir = fullfile(audioRoot, num2str(year));
chFiles = dir(fullfile(yearDir, sprintf('%s_*.wav', channelPrefix)));
if isempty(chFiles)
    error('No %s_* files in %s', channelPrefix, yearDir)
end
chFilePaths = fullfile({chFiles.folder}, {chFiles.name});
nProfile = min(numFilesToProfile, numel(chFilePaths));
fprintf('Profiling first %d of %d files for %d.\n\n', nProfile, numel(chFilePaths), year)

% ---- Warm-up (discard timing on the first 2 files: JIT + GPU init) ----
for w = 1:min(2, nProfile)
    [a, fsw, ~] = audioReadWithRetry(chFilePaths{w});
    if ~isempty(a) && isValidAudio(a)
        try
            gavdNetInference(a, fsw, model, g.AvailableMemory, ...
                featureFraming, frameStandardization, minSilenceDuration, false);
        catch
            % Ignore warm-up failures (e.g. a too-short file).
        end
    end
end

% ---- Timed loop ----
tReset = nan(nProfile, 1);
tRead  = nan(nProfile, 1);
tGPU   = nan(nProfile, 1);
tCPU   = nan(nProfile, 1);
fileDurations = nan(nProfile, 1);
nInferErrors = 0;   % files that errored in inference (e.g. too short)

for i = 1:nProfile
    fp = chFilePaths{i};

    % Reset cost (what A1 removes on 249 of every 250 files)
    tr = tic;
    wait(g);
    reset(g);
    tReset(i) = toc(tr);
    bytesAvailable = g.AvailableMemory;

    % Read cost
    tr = tic;
    [audioIn, sampleRate, ~] = audioReadWithRetry(fp);
    tRead(i) = toc(tr);
    if isempty(audioIn) || ~isValidAudio(audioIn)
        continue
    end
    fileDurations(i) = numel(audioIn) / sampleRate;

    % Inference wall time and GPU-only execTime. Wrap in try/catch so a
    % too-short file (which the real script handles via its retry wrapper)
    % does not abort profiling - just record it as skipped and continue.
    try
        tr = tic;
        [~, ~, execTime] = gavdNetInference(audioIn, sampleRate, model, ...
            bytesAvailable, featureFraming, frameStandardization, ...
            minSilenceDuration, false);
        tInfWall = toc(tr);
        tGPU(i) = execTime;
        tCPU(i) = max(tInfWall - execTime, 0);
    catch
        nInferErrors = nInferErrors + 1;
    end

    if mod(i, 25) == 0
        fprintf('  ... %d/%d files profiled\n', i, nProfile)
    end
end

% ---- Aggregate ----
mReset = mean(tReset, 'omitnan');
mRead  = mean(tRead,  'omitnan');
mGPU   = mean(tGPU,   'omitnan');
mCPU   = mean(tCPU,   'omitnan');
mDur   = mean(fileDurations, 'omitnan');

% Current per-file wall (with the OLD every-file reset)
perFileOld = mReset + mRead + mGPU + mCPU;
% Projected per-file wall under A1 (reset amortised over gpuResetEveryN=250)
perFileNew = mReset / 250 + mRead + mGPU + mCPU;

fprintf('\n===================== PER-FILE TIMING (mean) =====================\n');
fprintf('  files profiled      : %d   (inference errors / too-short skipped: %d)\n', ...
    nProfile, nInferErrors);
fprintf('  file audio duration : %8.2f s\n', mDur);
fprintf('  tReset (per file)   : %8.3f s   %5.1f%% of old per-file wall\n', mReset, 100*mReset/perFileOld);
fprintf('  tRead  (I/O)        : %8.3f s   %5.1f%%\n', mRead, 100*mRead/perFileOld);
fprintf('  tGPU   (execTime)   : %8.3f s   %5.1f%%\n', mGPU,  100*mGPU/perFileOld);
fprintf('  tCPU   (preproc)    : %8.3f s   %5.1f%%\n', mCPU,  100*mCPU/perFileOld);
fprintf('  ---------------------------------------------\n');
fprintf('  per-file wall (OLD, reset every file) : %8.3f s\n', perFileOld);
fprintf('  per-file wall (A1, reset every 250)   : %8.3f s   (%.1fx faster)\n', ...
    perFileNew, perFileOld / perFileNew);

% ---- Projections ----
filesRemaining2006 = 258017 - 90250;
fprintf('\n===================== A1 PROJECTION ==============================\n');
fprintf('  Remaining 2006 files (~%d):\n', filesRemaining2006);
fprintf('    OLD: %6.1f h     A1: %6.1f h     saved: %6.1f h\n', ...
    perFileOld * filesRemaining2006 / 3600, ...
    perFileNew * filesRemaining2006 / 3600, ...
    (perFileOld - perFileNew) * filesRemaining2006 / 3600);

% ---- Phase B guidance ----
gpuFrac = mGPU / perFileNew;   % GPU share of the A1-optimised per-file wall
ioFrac  = mRead / perFileNew;
cpuFrac = mCPU / perFileNew;
fprintf('\n===================== PHASE B (dual-GPU) GUIDANCE ================\n');
fprintf('  After A1, per-file breakdown is ~ GPU %.0f%%, I/O %.0f%%, CPU-preproc %.0f%%.\n', ...
    100*gpuFrac, 100*ioFrac, 100*cpuFrac);
if gpuFrac < 0.15
    fprintf('  -> GPU compute is a SMALL fraction. A second GPU (T550) will help little.\n');
    if ioFrac >= cpuFrac
        fprintf('     Bottleneck looks I/O-bound (external E: drive). Two workers reading\n');
        fprintf('     E: at once may THRASH - Phase B is NOT recommended; consider faster\n');
        fprintf('     storage / staging files to local SSD instead.\n');
    else
        fprintf('     Bottleneck looks CPU-preproc-bound. More CPU-parallel workers feeding\n');
        fprintf('     one GPU would help more than a second GPU - but still hits the E: I/O\n');
        fprintf('     ceiling. Weigh Phase B against simply staging audio to local SSD.\n');
    end
else
    fprintf('  -> GPU compute is a MEANINGFUL fraction. Dual-GPU (Phase B) could give up to\n');
    fprintf('     ~ 1 + (T550/4090 throughput). Proceed to build Phase B (static split).\n');
end
fprintf('==================================================================\n');
end
