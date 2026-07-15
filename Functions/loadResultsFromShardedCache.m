function [results, startFileIdx, isValidResume] = loadResultsFromShardedCache( ...
        savePath, cacheShardSize, currentFeatureFraming, ...
        currentFrameStandardization, currentChFileNames)
% LOADRESULTSFROMSHARDEDCACHE Load a partial cache for resume. Returns the
% reconstructed results struct array, the file index to resume from, and
% whether a valid resume was found. On any mismatch / corruption it returns
% ([], 1, false) so the caller restarts the list from index 1.
%
% Two formats are supported, in priority order:
%   1. Sharded cache (preferred): a manifest (<base>_manifest.mat) plus part
%      files (<base>_part<K>.mat), reassembled by index. Checked FIRST so that
%      once a list has sharded progress, resume uses it rather than an older
%      legacy single-file cache whose stale nextFileIdx would silently discard
%      tail progress.
%   2. Legacy single-file cache at savePath itself
%      (detector_raw_partial_<year>.mat). An in-flight year written by the
%      previous cache scheme (e.g. an existing 2006 cache). On first resume it
%      is rewritten as a complete set of shards + manifest and the legacy file
%      is removed, so all subsequent resumes use the up-to-date sharded cache.
%
% Indices are LOCAL to whatever file list this cache tracks (the whole year
% for the serial run; one worker's range for a dual-GPU worker).
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
    results = struct([]);
    startFileIdx = 1;
    isValidResume = false;

    [d, n, ~] = fileparts(savePath);
    base = fullfile(d, n);
    manifestPath = [base, '_manifest.mat'];

    % --- Format 1 (preferred): sharded cache ---
    if exist(manifestPath, 'file')
        [results, startFileIdx, isValidResume] = tryLoadSharded(base, ...
            manifestPath, currentFeatureFraming, currentFrameStandardization, ...
            currentChFileNames);
        if isValidResume
            return
        end
    end

    % --- Format 2: legacy single-file cache (migrated to shards on resume) ---
    if exist(savePath, 'file')
        try
            legacy = load(savePath);
            hasFields = all(isfield(legacy, {'results', 'nextFileIdx', ...
                'featureFraming', 'frameStandardization', 'chFileNames'}));
            if hasFields && settingsAndFilesMatch(legacy, currentFeatureFraming, ...
                    currentFrameStandardization, currentChFileNames)
                % Migrate to the sharded format so future resumes (including
                % interruptions during the tail) use the up-to-date sharded
                % progress instead of this file's stale nextFileIdx.
                try
                    writeAllShards(base, legacy.results, legacy.nextFileIdx, ...
                        legacy.featureFraming, legacy.frameStandardization, ...
                        legacy.chFileNames, cacheShardSize);
                    try
                        delete(savePath);
                    catch
                        % Best effort: shards are checked first, so a leftover
                        % legacy file no longer shadows them.
                    end
                catch ME
                    warning(['Legacy->sharded migration failed: %s. Resuming ' ...
                        'from the legacy cache this run.'], ME.message);
                end
                results = legacy.results;
                startFileIdx = legacy.nextFileIdx;
                isValidResume = true;
                return
            elseif hasFields
                warning(['Legacy partial cache %s does not match current ' ...
                    'settings / file list. Ignoring it.'], savePath)
            end
        catch ME
            warning('Could not load legacy partial cache %s: %s', savePath, ME.message);
        end
    end
end


function [results, startFileIdx, isValidResume] = tryLoadSharded(base, ...
        manifestPath, currentFeatureFraming, currentFrameStandardization, ...
        currentChFileNames)
% Load and reassemble a sharded cache. Returns ([], 1, false) on any manifest
% mismatch, missing/unreadable shard, or coverage gap so the caller can fall
% back to a legacy cache or restart the list.
    results = struct([]);
    startFileIdx = 1;
    isValidResume = false;

    try
        manifest = load(manifestPath);
    catch ME
        warning(['Could not load partial cache manifest %s: %s. Ignoring ' ...
            'sharded cache.'], manifestPath, ME.message);
        return
    end
    if ~settingsAndFilesMatch(manifest, currentFeatureFraming, ...
            currentFrameStandardization, currentChFileNames)
        warning(['Partial cache manifest %s does not match current settings / ' ...
            'file list. Ignoring sharded cache.'], manifestPath)
        return
    end

    shardFiles = dir(sprintf('%s_part*.mat', base));
    if isempty(shardFiles)
        return
    end

    % Load every shard, trimming any entries beyond nextFileIdx-1 (a shard can
    % briefly hold more than the manifest commits if a crash landed between the
    % shard write and the manifest write). Assemble in ascending start order
    % and require the shards to tile 1..nExpected with no gaps - a gap would
    % silently drop files. Assembly is by horzcat rather than indexed
    % assignment so the struct field set is inherited from the shards.
    nExpected = manifest.nextFileIdx - 1;
    nShards = numel(shardFiles);
    starts = zeros(nShards, 1);
    ends = zeros(nShards, 1);
    parts = cell(nShards, 1);
    for s = 1:nShards
        sp = fullfile(shardFiles(s).folder, shardFiles(s).name);
        try
            shard = load(sp);
        catch ME
            warning(['Could not load partial cache shard %s: %s. Ignoring ' ...
                'sharded cache.'], sp, ME.message);
            return
        end
        gStart = shard.globalStart;
        gEnd = min(shard.globalEnd, nExpected);
        starts(s) = gStart;
        ends(s) = gEnd;
        if gEnd >= gStart
            parts{s} = shard.results(1:(gEnd - gStart + 1));
        else
            parts{s} = [];
        end
    end
    [starts, order] = sort(starts);
    ends = ends(order);
    parts = parts(order);

    assembled = struct([]);
    expectedNext = 1;
    for s = 1:nShards
        if isempty(parts{s})
            continue
        end
        if starts(s) ~= expectedNext
            break
        end
        if isempty(assembled)
            assembled = parts{s};
        else
            assembled = [assembled, parts{s}]; %#ok<AGROW>
        end
        expectedNext = ends(s) + 1;
    end

    if expectedNext ~= nExpected + 1
        warning(['Sharded partial cache under %s is incomplete (covers %d of ' ...
            '%d entries). Ignoring it.'], base, expectedNext - 1, nExpected)
        return
    end

    results = assembled;
    startFileIdx = manifest.nextFileIdx;
    isValidResume = true;
end


function writeAllShards(base, results, nextFileIdx, featureFraming, ...
        frameStandardization, chFileNames, cacheShardSize)
% Write results(1:nextFileIdx-1) out as a COMPLETE set of shards plus a
% manifest. Used to migrate a legacy single-file cache into the sharded
% format on first resume. Shards are written before the manifest so the
% manifest only commits once its shards are durable.
    nDone = nextFileIdx - 1;
    numShards = ceil(nDone / cacheShardSize);
    for k = 1:numShards
        sStart = (k - 1) * cacheShardSize + 1;
        sEnd = min(k * cacheShardSize, nDone);
        shard.results = results(sStart:sEnd);
        shard.globalStart = sStart;
        shard.globalEnd = sEnd;
        save(sprintf('%s_part%d.mat', base, k), '-struct', 'shard', '-v7.3');
        clear shard
    end
    manifest.nextFileIdx = nextFileIdx;
    manifest.featureFraming = featureFraming;
    manifest.frameStandardization = frameStandardization;
    manifest.chFileNames = chFileNames;
    manifest.cacheShardSize = cacheShardSize;
    manifest.lastShard = numShards;
    save([base, '_manifest.mat'], '-struct', 'manifest', '-v7.3');
end


function tf = settingsAndFilesMatch(cache, featureFraming, frameStandardization, chFileNames)
% Shared resume validator: the cached featureFraming / frameStandardization
% and the file list (count + names) must match the current run.
    settingsMatch = isequaln(cache.featureFraming, featureFraming) && ...
        isequaln(cache.frameStandardization, frameStandardization);
    filesMatch = isequal(numel(cache.chFileNames), numel(chFileNames)) && ...
        all(strcmp(cache.chFileNames(:), chFileNames(:)));
    tf = settingsMatch && filesMatch;
end
