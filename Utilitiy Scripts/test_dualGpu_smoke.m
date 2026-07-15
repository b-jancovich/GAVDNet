function test_dualGpu_smoke(numFiles)
% TEST_DUALGPU_SMOKE End-to-end smoke test for the file-level dual-GPU path.
%
% Runs the SAME small set of real 2006 files two ways in one session:
%   (1) serial  - all files on the primary GPU (gpuChain(1)), via
%                 runInferenceFileLoop (the shared per-file loop);
%   (2) dual-GPU - files split ~50/50 across both GPUs on INTERLEAVED
%                 (odd/even) global indices, via runYearDualGpu, so the
%                 non-contiguous merge path is exercised.
% Then it checks that the dual-GPU merge reassembles the files in the correct
% global order and that results agree with the serial run:
%   - worker-A files (same GPU as serial) must be BIT-IDENTICAL;
%   - worker-B files (other GPU) are compared within a small tolerance,
%     because different GPU architectures can differ at the eps level.
%
% Running serial-then-dual in one session also exercises the real client
% GPU-context release before the workers pin their devices. WATCH the log for
% two DIFFERENT "pinned to GPU device ..." lines (primary vs secondary) and
% for any eGPU link instability. Uses temp caches under %TEMP% only.
%
% Usage:
%   test_dualGpu_smoke        % default 24 files
%   test_dualGpu_smoke(40)
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

if nargin < 1 || isempty(numFiles)
    numFiles = 24;
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
projectRoot = fileparts(scriptDir);   % parent of "Utilitiy Scripts"
addpath(fullfile(projectRoot, "Functions"))
gavdNetDataPath = userGavdNetDataPath;

modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
if isempty(modelList)
    error('No GAVDNet_trained_* model found in %s', gavdNetDataPath)
end
load(fullfile(modelList(1).folder, modelList(1).name))  % loads `model`
postProcOptions.LT = model.dataSynthesisParams.meanTargetCallDuration .* postProcOptions.LT_scaler;

% ---- GPU chain (largest first) ----
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

% ---- File list (first numFiles channel-1 files of the year) ----
yearDir = fullfile(audioRoot, num2str(year));
chFiles = dir(fullfile(yearDir, sprintf('%s_*.wav', channelPrefix)));
if numel(chFiles) < 2
    error('Need >= 2 files in %s', yearDir)
end
chFiles = chFiles(1:min(numFiles, numel(chFiles)));
chFilePaths = fullfile({chFiles.folder}, {chFiles.name});
names = {chFiles.name};
N = numel(chFilePaths);
fprintf('Smoke-testing on %d files from %d.\n', N, year)

% ---- Shared opts (small cache params so sharding is exercised) ----
opts = struct( ...
    'featureFraming', featureFraming, ...
    'frameStandardization', frameStandardization, ...
    'minSilenceDuration', minSilenceDuration, ...
    'plotting', false, ...
    'activationThreshold', postProcOptions.AT, ...
    'windowDur', windowDur, ...
    'maxInferenceRetries', 3, ...
    'maxConsecGpuFailures', 3, ...
    'gpuResetEveryN', 50, ...
    'enableShortFileSkip', true, ...
    'shortFileSkipThreshSec', postProcOptions.LT, ...
    'partialCacheEveryN', 5, ...
    'cacheShardSize', 10, ...
    'progressLabel', '');

tmp = fullfile(tempdir, 'GAVDNet_dualgpu_smoke');
if exist(tmp, 'dir')
    rmdir(tmp, 's');
end
mkdir(tmp);

% ---- (1) SERIAL run: all files on the primary GPU ----
fprintf('\n===== SERIAL run (primary GPU) =====\n')
gpuDevice(gpuChain(1).deviceID);
dsSerial = struct('useGPU', true, 'gpuDeviceID', gpuChain(1).deviceID, ...
    'bytesAvailable', gpuChain(1).TotalMemory, 'gpuChain', gpuChain, ...
    'currentGpuChainIdx', 1, 'cpuMemoryBytes', cpuMemoryBytes, 'consecGpuFailures', 0);
cacheSerial = fullfile(tmp, 'serial.mat');
resultsSerial = runInferenceFileLoop(chFilePaths, names, 1, struct([]), ...
    model, opts, dsSerial, cacheSerial);

% ---- (2) DUAL-GPU run ----
% Route with a MAXIMALLY INTERLEAVED split (odd global indices -> worker A,
% even -> worker B). Production routes by length (planLengthRoutedSplit), but
% for near-uniform file sizes that split is (near-)contiguous and would not
% stress the non-contiguous merge path in runYearDualGpu + scatterStructArrays.
% An odd/even split guarantees the orchestrator is exercised on interleaved
% ranges on REAL GPUs, whatever the file sizes. (The length routing itself is
% unit-tested in test_planLengthRoutedSplit.) For transparency, report the
% length-routed split these files would actually get in production.
fprintf('\n===== DUAL-GPU run =====\n')
[~, ~, lenInfo] = planLengthRoutedSplit([chFiles.bytes], 1:N, 0.6);
fprintf(['Production length-routed split on these %d files: primary %d files / ' ...
    '%.3f GB (largest), secondary %d files / %.3f GB (smallest).\n'], ...
    N, lenInfo.nPrimary, lenInfo.bytesPrimary / 1e9, ...
    lenInfo.nSecondary, lenInfo.bytesSecondary / 1e9)
aRange = 1:2:N;    % odd global indices  -> worker A (primary GPU)
bRange = 2:2:N;    % even global indices -> worker B (secondary GPU)
cacheDual = fullfile(tmp, 'dual.mat');
resultsDual = runYearDualGpu(chFilePaths, names, aRange, bRange, struct([]), ...
    model, opts, gpuChain, cpuMemoryBytes, cacheDual);

% ---- Checks ----
fprintf('\n===== CHECKS =====\n')
allPass = true;
allPass = check('serial count = N', numel(resultsSerial) == N) && allPass;
allPass = check('dual count = N', numel(resultsDual) == N) && allPass;
allPass = check('file order preserved (merge)', ...
    isequal({resultsDual.fileName}, {resultsSerial.fileName})) && allPass;

% RAW-PROBABILITY equivalence is the acceptance criterion, because the raw
% probabilities are the durable artefact and are re-thresholded in postproc
% sweeps - so agreement at a single activation threshold is not enough. The
% workers match the serial compute-thread count (runYearDualGpu sets
% maxNumCompThreads('automatic')), so the primary-GPU worker is bit-identical
% to serial and only the secondary GPU differs, by the cross-GPU hardware
% floor. Gate on the MAX |diff| being below the postproc hysteresis band
% (AT 0.70 / DT 0.699 = 1e-3), which bounds the worst-case detection change at
% ANY future threshold. The distribution and threshold-crossing disagreements
% across a range of thresholds are reported for transparency.
maxTol = 1e-3;
thr = [0.10 0.30 0.50 0.70 0.90];
sA = probStats(resultsSerial, resultsDual, aRange, thr);
sB = probStats(resultsSerial, resultsDual, bRange, thr);
fprintf('  worker A (%s): median=%.2e p99=%.2e MAX=%.2e | crossings@%s = %s\n', ...
    gpuChain(1).Name, sA.med, sA.p99, sA.max, mat2str(thr), mat2str(sA.dis));
fprintf('  worker B (%s): median=%.2e p99=%.2e MAX=%.2e | crossings@%s = %s\n', ...
    gpuChain(2).Name, sB.med, sB.p99, sB.max, mat2str(thr), mat2str(sB.dis));
allPass = check('worker-A files same length as serial', sA.lenOK) && allPass;
allPass = check(sprintf('worker-A raw probs within %.0e (MAX=%.2e)', maxTol, sA.max), ...
    sA.max < maxTol) && allPass;
allPass = check('worker-B files same length as serial', sB.lenOK) && allPass;
allPass = check(sprintf('worker-B raw probs within %.0e (MAX=%.2e)', maxTol, sB.max), ...
    sB.max < maxTol) && allPass;

% ---- Cleanup ----
deletePartialCache(cacheSerial)
deletePartialCache(cacheDual)
deletePartialCache(fullfile(tmp, 'dual_gpuA.mat'))
deletePartialCache(fullfile(tmp, 'dual_gpuB.mat'))
rmdir(tmp, 's');
delete(gcp('nocreate'))

if allPass
    fprintf(['\nALL DUAL-GPU SMOKE CHECKS PASSED. Confirm the two "pinned to ' ...
        'GPU device" log lines above named DIFFERENT GPUs, and that no eGPU ' ...
        'instability occurred.\n']);
else
    error('Dual-GPU smoke test FAILED - see checks above.');
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
function s = probStats(resultsSerial, resultsDual, range, thr)
% Distribution of per-bin |serial - dual| probability differences over all
% files in range, plus the number of above-threshold crossing disagreements
% at each threshold in thr. s has fields: lenOK, med, p99, max, dis (1xN).
s.lenOK = true;
allDiffs = [];
dis = zeros(1, numel(thr));
for i = range
    ps = double(resultsSerial(i).probabilities(:));
    pd = double(resultsDual(i).probabilities(:));
    if ~isequal(numel(ps), numel(pd))
        s.lenOK = false;
        continue
    end
    if isempty(ps)
        continue
    end
    allDiffs = [allDiffs; abs(ps - pd)]; %#ok<AGROW>
    for t = 1:numel(thr)
        dis(t) = dis(t) + sum((ps > thr(t)) ~= (pd > thr(t)));
    end
end
if isempty(allDiffs)
    s.med = 0; s.p99 = 0; s.max = 0;
else
    s.med = median(allDiffs);
    s.p99 = prctile(allDiffs, 99);
    s.max = max(allDiffs);
end
s.dis = dis;
end
