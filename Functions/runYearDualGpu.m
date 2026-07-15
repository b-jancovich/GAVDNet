function results = runYearDualGpu(chFilePaths, chFileNames, aRange, bRange, ...
        preloadResults, model, opts, gpuChain, cpuMemoryBytes, saveNamePathPartial)
% RUNYEARDUALGPU Run one year's remaining inference concurrently on two GPUs.
%
% The remaining files have already been split by the caller into two
% global-index ranges: aRange (primary GPU, gpuChain(1)) and bRange (secondary
% GPU, gpuChain(2)). The split is length-routed (planLengthRoutedSplit): the
% largest files go to the primary and the smallest to the secondary, so the
% ranges are in general NON-contiguous subsets of the remaining files. Each
% range is processed by a parallel worker pinned to its GPU, running the shared
% per-file loop (runInferenceFileLoop) with LOCAL indexing over its own file
% list and its own worker-scoped partial cache
% (detector_raw_partial_<year>_gpuA/_gpuB). Each worker resumes independently
% from its own cache; because the split is deterministic (derived from the
% fixed file sizes and the primary work fraction), the ranges are identical
% across restarts. The two workers' results are SCATTERED back to their global
% indices and merged with the preloaded results into one global results array
% in original file order.
%
% eGPU NOTE: each worker holds a CUDA context on its OWN GPU only (its
% fallback chain is a single device, so it drops to CPU rather than onto the
% other worker's GPU). The client releases its GPU context before the workers
% start. Validate stability on a small run - a second CUDA context on the
% Thunderbolt 4090 has previously destabilised the link.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

    functionsDir = fileparts(mfilename('fullpath'));

    % Worker-scoped cache paths (off the serial cache base).
    [pcDir, pcName, ~] = fileparts(saveNamePathPartial);
    cacheA = fullfile(pcDir, [pcName, '_gpuA.mat']);
    cacheB = fullfile(pcDir, [pcName, '_gpuB.mat']);

    % Per-worker file lists / names (sliced to each worker's global range).
    pathsA = chFilePaths(aRange);   namesA = chFileNames(aRange);
    pathsB = chFilePaths(bRange);   namesB = chFileNames(bRange);

    % Per-worker resume from each worker's own cache (LOCAL indexing).
    [preA, startA, okA] = loadResultsFromShardedCache(cacheA, opts.cacheShardSize, ...
        opts.featureFraming, opts.frameStandardization, namesA);
    if ~okA
        preA = struct([]); startA = 1;
    end
    [preB, startB, okB] = loadResultsFromShardedCache(cacheB, opts.cacheShardSize, ...
        opts.featureFraming, opts.frameStandardization, namesB);
    if ~okB
        preB = struct([]); startB = 1;
    end

    % Per-worker device state: a SINGLE-device fallback chain (own GPU only).
    dsA = makeWorkerDeviceState(gpuChain(1), cpuMemoryBytes);
    dsB = makeWorkerDeviceState(gpuChain(2), cpuMemoryBytes);

    % Per-worker opts (distinct progress labels).
    optsA = opts; optsA.progressLabel = '[GPU A] ';
    optsB = opts; optsB.progressLabel = '[GPU B] ';

    fprintf(['Dual-GPU: worker A -> GPU %d (%s), %d files (largest; resume local %d); ' ...
        'worker B -> GPU %d (%s), %d files (smallest; resume local %d).\n'], ...
        gpuChain(1).deviceID, gpuChain(1).Name, numel(aRange), startA, ...
        gpuChain(2).deviceID, gpuChain(2).Name, numel(bRange), startB)

    % Ensure a fresh 2-worker pool and release the client's GPU context so the
    % workers own the GPUs.
    p = gcp('nocreate');
    if isempty(p) || p.NumWorkers ~= 2
        delete(p)
        parpool('local', 2);
    end
    try
        gpuDevice([]);
    catch
        % Some MATLAB releases do not support deselecting the device; the
        % client simply does no GPU work during the parallel phase.
    end

    % Send each worker only its own data (Composite avoids broadcasting both
    % workers' file lists / preloaded results to both workers).
    poolPaths = Composite(); poolNames = Composite(); poolStart = Composite();
    poolPre = Composite();   poolOpts = Composite();  poolDs = Composite();
    poolCache = Composite();
    poolPaths{1} = pathsA; poolPaths{2} = pathsB;
    poolNames{1} = namesA; poolNames{2} = namesB;
    poolStart{1} = startA; poolStart{2} = startB;
    poolPre{1}   = preA;   poolPre{2}   = preB;
    poolOpts{1}  = optsA;  poolOpts{2}  = optsB;
    poolDs{1}    = dsA;    poolDs{2}    = dsB;
    poolCache{1} = cacheA; poolCache{2} = cacheB;

    spmd(2)
        addpath(functionsDir)               % workers need Functions/ on the path
        % Match the serial path's compute-thread count. Parallel workers
        % default to a single computational thread; the serial client uses all
        % cores (gpuConfig sets 'automatic'). A thread-count mismatch changes
        % the CPU STFT/mel reduction order and perturbs raw probabilities by up
        % to ~0.1 at near-zero file-tail bins - which matters because raw
        % probabilities are the durable artefact and are re-thresholded in
        % postproc sweeps. With matched thread counts the primary-GPU worker's
        % raw probabilities are bit-identical to the serial run.
        maxNumCompThreads('automatic');
        gpuDevice(poolDs.gpuDeviceID);      % pin this worker to its GPU
        fprintf('%spinned to GPU device %d: %s (%d compute threads)\n', ...
            poolOpts.progressLabel, poolDs.gpuDeviceID, gpuDevice().Name, ...
            maxNumCompThreads);
        resWorker = runInferenceFileLoop(poolPaths, poolNames, poolStart, ...
            poolPre, model, poolOpts, poolDs, poolCache);
    end

    resultsA = resWorker{1};
    resultsB = resWorker{2};

    % Merge by SCATTERING each piece to its global indices. The length-routed
    % split makes aRange / bRange non-contiguous, so a plain concatenation would
    % not preserve global file order. The preload covers 1..firstRemaining-1
    % (contiguous); worker A covers aRange; worker B covers bRange; together
    % they partition 1..N. scatterStructArrays field-unions the three (they may
    % carry different fields) and places each at its true global indices.
    N = numel(chFilePaths);
    firstRemaining = min([aRange(:); bRange(:)]);
    preRange = 1:(firstRemaining - 1);
    results = scatterStructArrays(N, {preloadResults, resultsA, resultsB}, ...
        {preRange, aRange, bRange});
end

% ------------------------------------------------------------------------
function ds = makeWorkerDeviceState(gpuChainEntry, cpuMemoryBytes)
% Single-device state/chain for one dual-GPU worker: it uses only its own GPU
% and falls back to CPU (never the other worker's GPU) on failure.
    ds = struct( ...
        'useGPU', true, ...
        'gpuDeviceID', gpuChainEntry.deviceID, ...
        'bytesAvailable', gpuChainEntry.TotalMemory, ...
        'gpuChain', gpuChainEntry, ...
        'currentGpuChainIdx', 1, ...
        'cpuMemoryBytes', cpuMemoryBytes, ...
        'consecGpuFailures', 0);
end
