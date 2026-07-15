function [primaryRange, secondaryRange, info] = planLengthRoutedSplit(...
        fileSizes, globalIndices, primaryFileFraction)
% PLANLENGTHROUTEDSPLIT Route a year's remaining files between two GPUs by length.
%
% Splits a set of remaining audio files across two GPU workers by FILE LENGTH
% rather than by position: the largest files go to the primary GPU and the
% smallest to the secondary. This keeps long (single-segment) files on the
% high-memory primary GPU (which processes them at full batch size) and hands
% the secondary GPU the short files it can process efficiently - while still
% giving the secondary GPU some files to do even in a year where almost every
% file is long.
%
% File size (bytes) is used as a proxy for audio duration. The recordings are
% a fixed sample rate / bit depth / channel count, so bytes are monotonic in
% duration and require no extra I/O (dir() already returns them), unlike an
% audioinfo() header read per file.
%
% The split keeps the original count-fraction meaning of the knob: the primary
% GPU receives primaryFileFraction of the remaining files - but selected by
% size (the largest), not by position. Equivalently: sort the remaining files
% by length, give the top primaryFileFraction to the primary and the bottom
% (smallest) to the secondary. The primary GPU is the faster device, so it
% takes the larger share; tune primaryFileFraction from the per-worker
% throughput logged at year end. The split is clamped so BOTH workers always
% receive at least one file - so the secondary GPU always gets the shortest
% files to work on, even in a mostly-long year.
%
% Because the split is derived only from the (fixed) file sizes and
% primaryFileFraction, with ties broken deterministically by global index, the
% two ranges are identical across restarts - which is what lets each worker
% resume its own cache after an interruption.
%
% Inputs:
%   fileSizes           - vector of per-file sizes in bytes for the REMAINING
%                         files (proxy for audio duration)
%   globalIndices       - vector (same length) of each file's global index into
%                         the year's full file list
%   primaryFileFraction - scalar in the open interval (0,1): fraction of the
%                         remaining FILE COUNT assigned to the primary GPU (the
%                         largest files)
%
% Outputs:
%   primaryRange   - global indices for the primary GPU (largest files),
%                    sorted ascending (row vector)
%   secondaryRange - global indices for the secondary GPU (smallest files),
%                    sorted ascending (row vector)
%   info           - struct: nPrimary, nSecondary, bytesPrimary, bytesSecondary,
%                    totalBytes, primaryFileFractionRealized
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

fileSizes = double(fileSizes(:));
globalIndices = double(globalIndices(:));
M = numel(fileSizes);

if numel(globalIndices) ~= M
    error('planLengthRoutedSplit:sizeMismatch', ...
        'fileSizes (%d) and globalIndices (%d) must have equal length.', ...
        M, numel(globalIndices));
end
if M < 2
    error('planLengthRoutedSplit:tooFewFiles', ...
        'Need at least 2 files to split across two GPUs (got %d).', M);
end
if ~isscalar(primaryFileFraction) || ~(primaryFileFraction > 0 && primaryFileFraction < 1)
    error('planLengthRoutedSplit:badFraction', ...
        'primaryFileFraction must be a scalar in the open interval (0,1); got %g.', ...
        primaryFileFraction);
end

% Sort the remaining files by size DESCENDING (largest first). Ties are broken
% by ascending global index so the ordering - and therefore the split - is
% deterministic across restarts (each worker must see an identical file list
% to resume its own cache).
[~, ord] = sortrows([-fileSizes, globalIndices], [1 2]);
gSorted = globalIndices(ord);
wSorted = fileSizes(ord);

% Take the top k files (the largest) for the primary GPU, the rest (the
% smallest) for the secondary. k is the same count the old position-based split
% used - round(fraction * numFiles) - so the knob keeps its meaning; only the
% SELECTION changed from "first by position" to "largest by size". Clamp so
% each worker keeps at least one file: the secondary GPU always gets the
% shortest files to process, even when nearly every file is large.
k = round(primaryFileFraction * M);
k = min(max(k, 1), M - 1);

primaryRange = sort(gSorted(1:k)).';        % largest files, ascending global order
secondaryRange = sort(gSorted(k+1:end)).';  % smallest files, ascending global order

info = struct();
info.nPrimary = k;
info.nSecondary = M - k;
info.bytesPrimary = sum(wSorted(1:k));
info.bytesSecondary = sum(wSorted(k+1:end));
info.totalBytes = info.bytesPrimary + info.bytesSecondary;
info.primaryFileFractionRealized = k / M;
end
