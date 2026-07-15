function test_dualGpu_resume(numFiles)
% TEST_DUALGPU_RESUME Automated end-to-end resume test for the dual-GPU path.
%
% 1. Runs a small set of real 2006 files dual-GPU (the uninterrupted
%    REFERENCE) -> resultsFull.
% 2. Rewrites each worker's partial cache to a MID-RANGE state built from the
%    reference results, tagging the cached entries. This reproduces the exact
%    on-disk state a crash midway through each worker's range would leave
%    (the cache is written by the same saveResultsToPartialCache the workers
%    use).
% 3. Reruns dual-GPU (the RESUMED run) and checks:
%      - file order preserved and every file's probabilities are identical to
%        the reference (correctness of resume + merge);
%      - each worker resumed FROM THE MIDDLE - the tagged cached entries
%        survive, and the entries after the resume point were re-processed
%        (no tag), proving it did not restart from file 1.
%
% This complements the manual Ctrl-C procedure in DUALGPU_NOTES.md and needs
% no interactive interruption. Uses temp caches under %TEMP% only.
%
% Usage:
%   test_dualGpu_resume        % default 16 files (8 per worker)
%   test_dualGpu_resume(24)
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

if nargin < 1 || isempty(numFiles)
    numFiles = 16;
end

% ---- Configuration (mirrors run_chagos_DGS_2000_to_2025.m) ----
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos_exclude_chorus.m";
gavdNetDataPath = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar Exclude Chorus";
audioRoot = "E:\Diego Garcia South 3Ch";
channelPrefix = "H08S1";
year = 2006;

% ---- Setup ----
userGavdNetDataPath = gavdNetDataPath;
run(configPath)
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(fullfile(projectRoot, "Functions"))
gavdNetDataPath = userGavdNetDataPath;

modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
if isempty(modelList)
    error('No GAVDNet_trained_* model found in %s', gavdNetDataPath)
end
load(fullfile(modelList(1).folder, modelList(1).name))  % loads `model`
postProcOptions.LT = model.dataSynthesisParams.meanTargetCallDuration .* postProcOptions.LT_scaler;

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

yearDir = fullfile(audioRoot, num2str(year));
chFiles = dir(fullfile(yearDir, sprintf('%s_*.wav', channelPrefix)));
chFiles = chFiles(1:min(numFiles, numel(chFiles)));
paths = fullfile({chFiles.folder}, {chFiles.name});
names = {chFiles.name};
N = numel(paths);

opts = struct( ...
    'featureFraming', featureFraming, 'frameStandardization', frameStandardization, ...
    'minSilenceDuration', minSilenceDuration, 'plotting', false, ...
    'activationThreshold', postProcOptions.AT, 'windowDur', windowDur, ...
    'maxInferenceRetries', 3, 'maxConsecGpuFailures', 3, 'gpuResetEveryN', 50, ...
    'enableShortFileSkip', true, 'shortFileSkipThreshSec', postProcOptions.LT, ...
    'partialCacheEveryN', 3, 'cacheShardSize', 6, 'progressLabel', '');

tmp = fullfile(tempdir, 'GAVDNet_dualgpu_resume');
if exist(tmp, 'dir'); rmdir(tmp, 's'); end
mkdir(tmp);
cachePath = fullfile(tmp, 'resume.mat');
cacheA = fullfile(tmp, 'resume_gpuA.mat');
cacheB = fullfile(tmp, 'resume_gpuB.mat');

% Interleaved split (odd -> worker A, even -> worker B) so the non-contiguous
% merge AND per-worker resume paths are exercised in the live orchestrator on
% real GPUs, independent of file-size uniformity. Production routes by length
% (planLengthRoutedSplit, unit-tested in test_planLengthRoutedSplit); here we
% only need runYearDualGpu to resume + merge correctly on non-contiguous ranges.
aRange = 1:2:N;
bRange = 2:2:N;
namesA = names(aRange);
namesB = names(bRange);

% ---- (1) Uninterrupted REFERENCE dual-GPU run ----
fprintf('\n===== REFERENCE dual-GPU run (%d files) =====\n', N)
resultsFull = runYearDualGpu(paths, names, aRange, bRange, struct([]), ...
    model, opts, gpuChain, cpuMemoryBytes, cachePath);
deletePartialCache(cacheA)   % discard the reference run's (complete) caches
deletePartialCache(cacheB)

% ---- (2) Build MID-RANGE partial worker caches from the reference, tagged ----
rA = resultsFull(aRange);
rB = resultsFull(bRange);
kA = max(1, floor(numel(rA) / 2));
kB = max(1, floor(numel(rB) / 2));
for j = 1:kA; rA(j).resumeTag = j; end
for j = 1:kB; rB(j).resumeTag = j; end
saveResultsToPartialCache(cacheA, rA(1:kA), kA, opts.featureFraming, ...
    opts.frameStandardization, namesA, opts.cacheShardSize);
saveResultsToPartialCache(cacheB, rB(1:kB), kB, opts.featureFraming, ...
    opts.frameStandardization, namesB, opts.cacheShardSize);
fprintf('Seeded partial caches: worker A done %d/%d, worker B done %d/%d.\n', ...
    kA, numel(rA), kB, numel(rB))

% ---- (3) RESUMED dual-GPU run ----
fprintf('\n===== RESUMED dual-GPU run (should resume mid-range) =====\n')
resultsResumed = runYearDualGpu(paths, names, aRange, bRange, struct([]), ...
    model, opts, gpuChain, cpuMemoryBytes, cachePath);

% ---- Checks ----
fprintf('\n===== CHECKS =====\n')
allPass = true;
allPass = check('count preserved', numel(resultsResumed) == N) && allPass;
allPass = check('file order preserved', ...
    isequal({resultsResumed.fileName}, {resultsFull.fileName})) && allPass;

% Correctness: every file's probabilities identical to the reference (each
% worker's GPU is deterministic, so re-processed tail == reference tail; the
% cached head == reference head by construction).
probsOK = true;
for i = 1:N
    if ~isequaln(resultsResumed(i).probabilities, resultsFull(i).probabilities)
        probsOK = false;
    end
end
allPass = check('all probabilities identical to reference', probsOK) && allPass;

% Resume-from-middle: cached entries kept their tag; tail entries did not.
allPass = check('worker A resumed from middle (not from file 1)', ...
    tagPattern(resultsResumed, aRange, kA)) && allPass;
allPass = check('worker B resumed from middle (not from file 1)', ...
    tagPattern(resultsResumed, bRange, kB)) && allPass;

% ---- Cleanup ----
deletePartialCache(cacheA)
deletePartialCache(cacheB)
deletePartialCache(cachePath)
rmdir(tmp, 's');
delete(gcp('nocreate'))

if allPass
    fprintf('\nALL DUAL-GPU RESUME CHECKS PASSED.\n');
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
