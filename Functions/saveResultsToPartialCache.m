function saveResultsToPartialCache(savePath, results, lastCompletedIdx, ...
        featureFraming, frameStandardization, chFileNames, cacheShardSize)
% SAVERESULTSTOPARTIALCACHE Write the sharded partial cache used by the
% resume-on-restart logic.
%
% Instead of rewriting the entire (ever-growing) results array on every
% checkpoint, only the shard containing lastCompletedIdx is written, plus a
% small manifest. Shard K covers indices ((K-1)*cacheShardSize + 1 :
% K*cacheShardSize). Because cacheShardSize is a whole multiple of the
% checkpoint interval, every shard boundary lands on a checkpoint, so each
% completed shard is finalised exactly once and never rewritten; only the
% current (incomplete) shard is rewritten as it fills.
%
% Indices are LOCAL to whatever file list this cache tracks. In the serial
% run that is the whole year's file list (so local == global). Each dual-GPU
% worker passes its own worker-scoped savePath and its own range's results,
% so its cache is self-contained with local indexing; the client maps
% local->global when it merges the workers' results.
%
% Inputs:
%   savePath            - Full path to the (legacy-style) cache name; the
%                         shard and manifest names are derived from it by
%                         stripping the extension.
%   results             - Current results struct array (local indexing)
%   lastCompletedIdx    - Index of the file whose iteration just ended;
%                         nextFileIdx in the manifest is set to this + 1.
%   featureFraming      - To verify settings on resume
%   frameStandardization - To verify settings on resume
%   chFileNames         - cellstr of file names for this cache's list, used
%                         on resume to ensure the file list hasn't changed
%   cacheShardSize      - Number of files per shard
%
% The manifest is written AFTER the shard so it only ever commits a
% nextFileIdx whose data is already durable; a crash between the two writes
% costs at most re-processing the current shard's tail (data is never lost).
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
    [d, n, ~] = fileparts(savePath);
    base = fullfile(d, n);

    shardK = ceil(lastCompletedIdx / cacheShardSize);
    shardStart = (shardK - 1) * cacheShardSize + 1;
    shardEnd = min(shardK * cacheShardSize, lastCompletedIdx);
    shardPath = sprintf('%s_part%d.mat', base, shardK);
    manifestPath = [base, '_manifest.mat'];

    shard.results = results(shardStart:shardEnd);
    shard.globalStart = shardStart;
    shard.globalEnd = shardEnd;

    manifest.nextFileIdx = lastCompletedIdx + 1;
    manifest.featureFraming = featureFraming;
    manifest.frameStandardization = frameStandardization;
    manifest.chFileNames = chFileNames;
    manifest.cacheShardSize = cacheShardSize;
    manifest.lastShard = shardK;

    try
        save(shardPath, '-struct', 'shard', '-v7.3');
        save(manifestPath, '-struct', 'manifest', '-v7.3');
    catch ME
        warning('Could not write partial cache shard to %s: %s', shardPath, ME.message);
    end
end
