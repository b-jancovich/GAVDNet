function results = scatterStructArrays(N, pieces, ranges)
% SCATTERSTRUCTARRAYS Assemble a global struct array from index-scattered pieces.
%
% Builds a 1-by-N struct array by placing each pieces{k} at the global indices
% ranges{k}. Unlike a plain concatenation, the ranges need NOT be contiguous or
% in order - this is what lets the dual-GPU merge reassemble worker results
% whose global indices are interleaved (e.g. when files are routed to the two
% GPUs by length rather than by position).
%
% The pieces may have different field sets (e.g. a "skipped" file entry lacks
% silenceMask). All field names are unioned into a common set - missing fields
% padded with [] and reordered consistently - so indexed assignment into the
% preallocated global array succeeds. Field order does not affect downstream
% (name-based) access or isequaln comparison.
%
% The ranges are expected to partition 1:N (disjoint, covering every index).
% Any index not written by a range is left as an all-empty-field entry, which
% the postprocessing loop treats as "no valid inference output" (zero
% detections) - the same defensive behaviour as a failed file.
%
% Inputs:
%   N      - total number of global entries (length of the output array)
%   pieces - cell array of struct arrays, one per source range
%   ranges - cell array of index vectors; ranges{k} are the global indices that
%            pieces{k} occupies (numel(ranges{k}) must equal numel(pieces{k}))
%
% Output:
%   results - 1-by-N struct array with pieces scattered to their global indices
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

if numel(pieces) ~= numel(ranges)
    error('scatterStructArrays:argMismatch', ...
        'pieces (%d) and ranges (%d) must have equal length.', ...
        numel(pieces), numel(ranges));
end

% Drop empty pieces / empty ranges (e.g. an empty preload on a fresh year).
keep = false(1, numel(pieces));
for i = 1:numel(pieces)
    keep(i) = ~isempty(pieces{i}) && ~isempty(ranges{i});
end
pieces = pieces(keep);
ranges = ranges(keep);

if isempty(pieces)
    results = struct([]);
    return
end

% Common field set across all pieces (stable order, first-seen wins).
allFields = {};
for i = 1:numel(pieces)
    allFields = union(allFields, fieldnames(pieces{i}), 'stable');
end
for i = 1:numel(pieces)
    pieces{i} = normalizeFields(pieces{i}, allFields);
end

% Preallocate a 1-by-N array of all-empty-field entries, then scatter each
% piece into its global indices. All pieces now share an identical field set
% and order, so indexed struct assignment is valid.
template = normalizeFields(struct(), allFields);
results = repmat(template, 1, N);
for i = 1:numel(pieces)
    r = ranges{i};
    if numel(r) ~= numel(pieces{i})
        error('scatterStructArrays:rangeSizeMismatch', ...
            'piece %d has %d entries but its range covers %d indices.', ...
            i, numel(pieces{i}), numel(r));
    end
    results(r) = pieces{i};
end
end

% ------------------------------------------------------------------------
function s = normalizeFields(s, allFields)
% Add any fields in allFields that s lacks (as []) and reorder s to allFields.
missing = setdiff(allFields, fieldnames(s));
for m = 1:numel(missing)
    [s.(missing{m})] = deal([]);
end
s = orderfields(s, allFields);
end
