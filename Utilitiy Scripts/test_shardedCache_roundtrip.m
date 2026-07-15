function test_shardedCache_roundtrip()
% Round-trip test for the A4 sharded partial-cache logic used in
% run_chagos_DGS_2000_to_2025.m. The cache functions are LOCAL to that
% script and cannot be called externally, so faithful copies are embedded
% below and exercised here against synthetic results in a temp directory.
% Validates: normal write -> crash -> resume; settings / file-list mismatch;
% missing-shard coverage guard; cleanup; the legacy single-file fallback WITH
% migration to shards (the existing-2006-cache safety path); that migration
% deletes the legacy file and future resumes use shards; and self-heal when a
% legacy file coexists with an incomplete sharded cache (the 2006-tail bug).
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

td = fullfile(tempdir, 'GAVDNet_test_shardedCache');
if exist(td, 'dir')
    rmdir(td, 's');
end
mkdir(td);
savePath = fullfile(td, 'detector_raw_partial_9999.mat');  % legacy-style name
base = fullfile(td, 'detector_raw_partial_9999');

ff = 'event-split';
fs = 'true';
cacheShardSize = 2000;
partialCacheEveryN = 250;
nFiles = 5500;
chFileNames = arrayfun(@(k) sprintf('H08S1_file_%05d.wav', k), 1:nFiles, ...
    'UniformOutput', false);

allPass = true;

truth = struct('fileName', {}, 'val', {});
for k = 1:nFiles
    truth(k).fileName = chFileNames{k};
    truth(k).val = k;
end

% ---- T1: checkpoint loop up to a crash at 5300, resume from shards ----
crashAt = 5300;
results = struct('fileName', {}, 'val', {});
for fileIdx = 1:crashAt
    results(fileIdx) = truth(fileIdx);
    if mod(fileIdx, partialCacheEveryN) == 0
        saveResultsToPartialCache(savePath, results, fileIdx, ff, fs, ...
            chFileNames, cacheShardSize);
    end
end
[r, startIdx, ok] = loadResultsFromShardedCache(savePath, cacheShardSize, ff, fs, chFileNames);
lastCheckpoint = floor(crashAt / partialCacheEveryN) * partialCacheEveryN;  % 5250
allPass = check('T1 resume valid',       ok == true) && allPass;
allPass = check('T1 startFileIdx=5251',  isequal(startIdx, lastCheckpoint + 1)) && allPass;
allPass = check('T1 numel results=5250', numel(r) == lastCheckpoint) && allPass;
allPass = check('T1 values intact',      isequal([r.val], 1:lastCheckpoint)) && allPass;
allPass = check('T1 three shard files',  countFiles(base, '_part*.mat') == 3) && allPass;

% ---- T2 / T3: settings and file-list mismatch rejected ----
[~, ~, ok2] = loadResultsFromShardedCache(savePath, cacheShardSize, 'none', fs, chFileNames);
allPass = check('T2 settings mismatch rejected', ok2 == false) && allPass;
[~, ~, ok3] = loadResultsFromShardedCache(savePath, cacheShardSize, ff, fs, chFileNames(1:end-1));
allPass = check('T3 filelist mismatch rejected', ok3 == false) && allPass;

% ---- T4: missing shard -> coverage check invalidates (no legacy present) ----
delete([base, '_part2.mat']);
[~, ~, ok4] = loadResultsFromShardedCache(savePath, cacheShardSize, ff, fs, chFileNames);
allPass = check('T4 missing shard rejected', ok4 == false) && allPass;

% ---- T5: cleanup removes all artefacts ----
for fileIdx = 250:250:1000
    saveResultsToPartialCache(savePath, results(1:fileIdx), fileIdx, ff, fs, ...
        chFileNames, cacheShardSize);
end
deletePartialCache(savePath);
allPass = check('T5 cleanup removes all', countFiles(base, '*') == 0) && allPass;

% ---- T6: legacy single-file cache is read AND migrated to shards ----
writeLegacy(savePath, truth(1:3000), 3001, ff, fs, chFileNames);
[rL, startL, okL] = loadResultsFromShardedCache(savePath, cacheShardSize, ff, fs, chFileNames);
allPass = check('T6 legacy resume valid',   okL == true) && allPass;
allPass = check('T6 legacy startIdx=3001',  isequal(startL, 3001)) && allPass;
allPass = check('T6 legacy values intact',  isequal([rL.val], 1:3000)) && allPass;
allPass = check('T6 legacy file deleted',   ~exist(savePath, 'file')) && allPass;
allPass = check('T6 manifest written',      exist([base, '_manifest.mat'], 'file') > 0) && allPass;
% 3000 files / 2000 shard = 2 shards
allPass = check('T6 two shards written',    countFiles(base, '_part*.mat') == 2) && allPass;

% ---- T7: after migration, second load resumes from shards, same result ----
[rL2, startL2, okL2] = loadResultsFromShardedCache(savePath, cacheShardSize, ff, fs, chFileNames);
allPass = check('T7 shard resume valid',    okL2 == true) && allPass;
allPass = check('T7 shard startIdx=3001',   isequal(startL2, 3001)) && allPass;
allPass = check('T7 shard values intact',   isequal([rL2.val], 1:3000)) && allPass;

deletePartialCache(savePath);

% ---- T8: legacy mismatch rejected (no migration) ----
writeLegacy(savePath, truth(1:3000), 3001, ff, fs, chFileNames);
[~, ~, ok8] = loadResultsFromShardedCache(savePath, cacheShardSize, 'none', fs, chFileNames);
allPass = check('T8 legacy mismatch rejected', ok8 == false) && allPass;
allPass = check('T8 legacy NOT migrated',      countFiles(base, '_part*.mat') == 0) && allPass;
deletePartialCache(savePath);

% ---- T9: SELF-HEAL - legacy + incomplete shards (the 2006-tail bug) ----
% Simulate the buggy state: a full legacy cache (1..4000) PLUS an incomplete
% sharded cache (only a late shard present, manifest claims tail progress).
% The loader must NOT trust the incomplete shards, must fall back to legacy,
% migrate to a COMPLETE shard set, and resume correctly.
writeLegacy(savePath, truth(1:4000), 4001, ff, fs, chFileNames);
% Incomplete sharded cache: only shard 3 (files 4001:4500) + a manifest that
% claims nextFileIdx=4501, but shards 1 & 2 (files 1:4000) are MISSING.
stray.results = truth(4001:4500);
stray.globalStart = 4001;
stray.globalEnd = 4500;
save([base, '_part3.mat'], '-struct', 'stray', '-v7.3');
man.nextFileIdx = 4501; man.featureFraming = ff; man.frameStandardization = fs;
man.chFileNames = chFileNames; man.cacheShardSize = cacheShardSize; man.lastShard = 3;
save([base, '_manifest.mat'], '-struct', 'man', '-v7.3');
[r9, start9, ok9] = loadResultsFromShardedCache(savePath, cacheShardSize, ff, fs, chFileNames);
allPass = check('T9 self-heal resume valid',   ok9 == true) && allPass;
allPass = check('T9 self-heal startIdx=4001',  isequal(start9, 4001)) && allPass;   % from legacy
allPass = check('T9 self-heal values intact',  isequal([r9.val], 1:4000)) && allPass;
allPass = check('T9 legacy file deleted',      ~exist(savePath, 'file')) && allPass;
% A subsequent load must now succeed from the (re-migrated, complete) shards.
[~, start9b, ok9b] = loadResultsFromShardedCache(savePath, cacheShardSize, ff, fs, chFileNames);
allPass = check('T9 reload from shards valid', ok9b == true && isequal(start9b, 4001)) && allPass;

rmdir(td, 's');

if allPass
    fprintf('\nALL SHARDED-CACHE TESTS PASSED. A4 verified.\n');
else
    error('A4 sharded-cache tests FAILED - see cases above.');
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

function nf = countFiles(base, pat)
nf = numel(dir([base, pat]));
end

function writeLegacy(savePath, resultsArr, nextFileIdx, ff, fs, chFileNames)
legacy.results = resultsArr;
legacy.nextFileIdx = nextFileIdx;
legacy.featureFraming = ff;
legacy.frameStandardization = fs;
legacy.chFileNames = chFileNames;
save(savePath, '-struct', 'legacy', '-v7.3');
end

% ======================= VERBATIM COPIES UNDER TEST =====================
% Exact copies of the local functions in run_chagos_DGS_2000_to_2025.m.
% Keep in sync if the originals change.

function saveResultsToPartialCache(savePath, results, lastCompletedIdx, ...
        featureFraming, frameStandardization, chFileNames, cacheShardSize)
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

function [results, startFileIdx, isValidResume] = loadResultsFromShardedCache( ...
        savePath, cacheShardSize, currentFeatureFraming, ...
        currentFrameStandardization, currentChFileNames)
    results = struct([]);
    startFileIdx = 1;
    isValidResume = false;
    [d, n, ~] = fileparts(savePath);
    base = fullfile(d, n);
    manifestPath = [base, '_manifest.mat'];

    if exist(manifestPath, 'file')
        [results, startFileIdx, isValidResume] = tryLoadSharded(base, ...
            manifestPath, currentFeatureFraming, currentFrameStandardization, ...
            currentChFileNames);
        if isValidResume
            return
        end
    end

    if exist(savePath, 'file')
        try
            legacy = load(savePath);
            hasFields = all(isfield(legacy, {'results', 'nextFileIdx', ...
                'featureFraming', 'frameStandardization', 'chFileNames'}));
            if hasFields && settingsAndFilesMatch(legacy, currentFeatureFraming, ...
                    currentFrameStandardization, currentChFileNames)
                try
                    writeAllShards(base, legacy.results, legacy.nextFileIdx, ...
                        legacy.featureFraming, legacy.frameStandardization, ...
                        legacy.chFileNames, cacheShardSize);
                    try
                        delete(savePath);
                    catch
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
    settingsMatch = isequaln(cache.featureFraming, featureFraming) && ...
        isequaln(cache.frameStandardization, frameStandardization);
    filesMatch = isequal(numel(cache.chFileNames), numel(chFileNames)) && ...
        all(strcmp(cache.chFileNames(:), chFileNames(:)));
    tf = settingsMatch && filesMatch;
end

function deletePartialCache(savePath)
    [d, n, ~] = fileparts(savePath);
    base = fullfile(d, n);
    targets = [{savePath}, {[base, '_manifest.mat']}];
    shardFiles = dir(sprintf('%s_part*.mat', base));
    for s = 1:numel(shardFiles)
        targets{end+1} = fullfile(shardFiles(s).folder, shardFiles(s).name); %#ok<AGROW>
    end
    for t = 1:numel(targets)
        if exist(targets{t}, 'file')
            try
                delete(targets{t})
            catch ME
                warning('Could not delete partial cache file %s: %s', ...
                    targets{t}, ME.message)
            end
        end
    end
end
