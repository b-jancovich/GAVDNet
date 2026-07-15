function deletePartialCache(savePath)
% DELETEPARTIALCACHE Remove every partial-cache artefact for a file list once
% its full raw cache is safely written: the manifest, all shard part files,
% and any legacy single-file cache. Given a worker-scoped base name (e.g.
% detector_raw_partial_<year>_gpuA.mat) it removes that worker's cache.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
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
