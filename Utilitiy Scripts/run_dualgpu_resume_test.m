function run_dualgpu_resume_test(numFiles)
% RUN_DUALGPU_RESUME_TEST Self-contained dual-GPU resume test over the files
% in E:\dual_gpu_resume_test. Does NOT touch run_chagos or any production
% config; it drives the same dual-GPU code path (runYearDualGpu, the shared
% per-file loop, and the sharded worker caches) directly, keeping all caches
% in %TEMP%.
%
% Phases (all visible in the console):
%   1. REFERENCE  - a full, uninterrupted dual-GPU run over the test files.
%   2. SIMULATED CRASH - each worker's partial cache is rolled back to the
%      midpoint of its range, using the SAME saveResultsToPartialCache the
%      workers use, i.e. exactly the on-disk state a crash midway would leave.
%   3. RESUMED    - a second dual-GPU run that must pick up from each worker's
%      midpoint (watch for "resume local <k>", k > 1, and workers starting at
%      "file k of M"), then complete.
%
% Checks: the resumed result is bit-identical to the reference (correctness of
% resume + merge) and each worker resumed FROM THE MIDDLE, not from file 1.
%
% Usage:
%   run_dualgpu_resume_test        % all H08S1_* files in the test folder
%   run_dualgpu_resume_test(20)    % cap to the first 20 files for a quicker run
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% ---- Test folder + production config/model (read-only use of the config) ----
audioFolder = "E:\dual_gpu_resume_test";
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos_exclude_chorus.m";
gavdNetDataPath = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar Exclude Chorus";
channelPrefix = "H08S1";

% ---- Setup (loads config vars + model; does not modify anything) ----
userGavdNetDataPath = gavdNetDataPath;
run(configPath)
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);            % parent of "Utilitiy Scripts"
addpath(fullfile(projectRoot, "Functions"))
gavdNetDataPath = userGavdNetDataPath;

modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
if isempty(modelList)
    error('No GAVDNet_trained_* model found in %s', gavdNetDataPath)
end
load(fullfile(modelList(1).folder, modelList(1).name))  % loads `model`
postProcOptions.LT = model.dataSynthesisParams.meanTargetCallDuration .* postProcOptions.LT_scaler;

% ---- GPUs (need two) ----
numGPUs = gpuDeviceCount("available");
if numGPUs < 2
    error('This test needs at least two GPUs (found %d).', numGPUs)
end
gpuChain = struct('deviceID', {}, 'Name', {}, 'TotalMemory', {});
for k = 1:numGPUs
    info = gpuDevice(k);
    gpuChain(k).deviceID = k;
    gpuChain(k).Name = char(info.Name);
    gpuChain(k).TotalMemory = info.TotalMemory;
end
[~, si] = sort([gpuChain.TotalMemory], 'descend');
gpuChain = gpuChain(si);
cpuMemoryBytes = 16e9;
fprintf('Primary GPU: %s | Secondary GPU: %s\n', gpuChain(1).Name, gpuChain(2).Name)

% ---- File list from the test folder (channel-1 first, else any wav) ----
chFiles = dir(fullfile(audioFolder, sprintf('%s_*.wav', channelPrefix)));
if isempty(chFiles)
    chFiles = dir(fullfile(audioFolder, '*.wav'));
end
if numel(chFiles) < 2
    error('Need >= 2 wav files in %s (found %d).', audioFolder, numel(chFiles))
end
if nargin >= 1 && ~isempty(numFiles)
    chFiles = chFiles(1:min(numFiles, numel(chFiles)));
end
paths = fullfile({chFiles.folder}, {chFiles.name});
names = {chFiles.name};
N = numel(paths);
fprintf('Resume-testing on %d files from %s\n', N, audioFolder)

opts = struct( ...
    'featureFraming', featureFraming, 'frameStandardization', frameStandardization, ...
    'minSilenceDuration', minSilenceDuration, 'plotting', false, ...
    'activationThreshold', postProcOptions.AT, 'windowDur', windowDur, ...
    'maxInferenceRetries', 3, 'maxConsecGpuFailures', 3, 'gpuResetEveryN', 50, ...
    'enableShortFileSkip', true, 'shortFileSkipThreshSec', postProcOptions.LT, ...
    'partialCacheEveryN', 5, 'cacheShardSize', 10, 'progressLabel', '');

tmp = fullfile(tempdir, 'GAVDNet_dualgpu_resume_test');
if exist(tmp, 'dir'); rmdir(tmp, 's'); end
mkdir(tmp);
cachePath = fullfile(tmp, 'resume.mat');
cacheA = fullfile(tmp, 'resume_gpuA.mat');
cacheB = fullfile(tmp, 'resume_gpuB.mat');

splitCount = min(max(round(0.5 * N), 1), N - 1);
aRange = 1:splitCount;
bRange = (splitCount + 1):N;
namesA = names(aRange);
namesB = names(bRange);

% ---- Phase 1: REFERENCE (uninterrupted) dual-GPU run ----
fprintf('\n========== PHASE 1: REFERENCE dual-GPU run (%d files) ==========\n', N)
resultsFull = runYearDualGpu(paths, names, aRange, bRange, struct([]), ...
    model, opts, gpuChain, cpuMemoryBytes, cachePath);
deletePartialCache(cacheA)   % discard the reference run's own (complete) caches
deletePartialCache(cacheB)

% ---- Phase 2: SIMULATE a crash - roll each worker's cache to its midpoint ----
fprintf('\n========== PHASE 2: SIMULATED CRASH (roll caches to midpoint) ==========\n')
rA = resultsFull(aRange);
rB = resultsFull(bRange);
kA = max(1, floor(numel(rA) / 2));
kB = max(1, floor(numel(rB) / 2));
for j = 1:kA; rA(j).resumeTag = j; end   % tag cached entries to verify resume
for j = 1:kB; rB(j).resumeTag = j; end
saveResultsToPartialCache(cacheA, rA(1:kA), kA, opts.featureFraming, ...
    opts.frameStandardization, namesA, opts.cacheShardSize);
saveResultsToPartialCache(cacheB, rB(1:kB), kB, opts.featureFraming, ...
    opts.frameStandardization, namesB, opts.cacheShardSize);
fprintf('Cache state after "crash": worker A done %d/%d, worker B done %d/%d.\n', ...
    kA, numel(rA), kB, numel(rB))

% ---- Phase 3: RESUMED dual-GPU run ----
fprintf('\n========== PHASE 3: RESUMED dual-GPU run (should pick up mid-range) ==========\n')
resultsResumed = runYearDualGpu(paths, names, aRange, bRange, struct([]), ...
    model, opts, gpuChain, cpuMemoryBytes, cachePath);

% ---- Checks ----
fprintf('\n========== CHECKS ==========\n')
allPass = true;
allPass = check('count preserved', numel(resultsResumed) == N) && allPass;
allPass = check('file order preserved', ...
    isequal({resultsResumed.fileName}, {resultsFull.fileName})) && allPass;

probsOK = true;
for i = 1:N
    if ~isequaln(resultsResumed(i).probabilities, resultsFull(i).probabilities)
        probsOK = false;
    end
end
allPass = check('all probabilities identical to reference', probsOK) && allPass;
allPass = check(sprintf('worker A resumed from middle (file %d, not 1)', kA + 1), ...
    tagPattern(resultsResumed, aRange, kA)) && allPass;
allPass = check(sprintf('worker B resumed from middle (file %d, not 1)', kB + 1), ...
    tagPattern(resultsResumed, bRange, kB)) && allPass;

% ---- Cleanup ----
deletePartialCache(cacheA)
deletePartialCache(cacheB)
deletePartialCache(cachePath)
rmdir(tmp, 's');
delete(gcp('nocreate'))

if allPass
    fprintf(['\nALL DUAL-GPU RESUME CHECKS PASSED. The Phase-3 log above should ' ...
        'show each worker starting at "file %d of %d" / "file %d of %d" ' ...
        '(mid-range), not file 1.\n'], kA + 1, numel(rA), kB + 1, numel(rB));
else
    error('Dual-GPU resume test FAILED - see checks above.');
end
end

% ------------------------------------------------------------------------
function ok = check(name, cond)
ok = logical(cond);
if ok
    fprintf('PASS  %s\n', name);
else
    fprintf('FAIL  %s\n', name);
end
end

% ------------------------------------------------------------------------
function ok = tagPattern(results, range, k)
% True if entries range(1:k) carry their resumeTag (came from the cache) and
% entries range(k+1:end) have an empty resumeTag (were re-processed).
ok = true;
for i = 1:numel(range)
    tag = results(range(i)).resumeTag;
    if i <= k
        if ~isequal(tag, i); ok = false; end
    else
        if ~isempty(tag); ok = false; end
    end
end
end
