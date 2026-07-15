function test_scatterStructArrays()
% Verify the dual-GPU merge helper (Functions/scatterStructArrays.m) places
% each piece at its true GLOBAL indices, even when the pieces have different
% field sets and - crucially for length-routed splitting - when the workers'
% index ranges are NON-contiguous / interleaved. This is what replaced the
% old contiguous-concatenation merge.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(scriptDir), 'Functions'))

allPass = true;

% --- Case 1: contiguous ranges (the old behaviour) preserved order ----------
preload = mkEntries('f', 1:3, false);          % processed (has silenceMask)
A = mkEntries('f', 4:6, false);
B = mkEntries('f', 7:9, false);
merged = scatterStructArrays(9, {preload, A, B}, {1:3, 4:6, 7:9});
allPass = check('contiguous: count = 9', numel(merged) == 9) && allPass;
allPass = check('contiguous: order preserved (val = 1..9)', ...
    isequal([merged.val], 1:9)) && allPass;

% --- Case 2: NON-contiguous / interleaved worker ranges ---------------------
% Length routing sends the largest files to A and the smallest to B, so their
% global indices interleave. The merge must still land every entry at its own
% global index.
preload = mkEntries('f', 1:3, false);
A = mkEntries('f', [4 6 8], false);            % primary: scattered globals
B = mkEntries('f', [5 7 9], false);            % secondary: scattered globals
merged = scatterStructArrays(9, {preload, A, B}, {1:3, [4 6 8], [5 7 9]});
allPass = check('interleaved: count = 9', numel(merged) == 9) && allPass;
allPass = check('interleaved: every entry at its global index (val = 1..9)', ...
    isequal([merged.val], 1:9)) && allPass;
allPass = check('interleaved: names correct', ...
    isequal({merged.fileName}, arrayfun(@(k) sprintf('f%02d', k), 1:9, 'UniformOutput', false))) && allPass;

% --- Case 3: differing field sets are unioned -------------------------------
preload = mkEntries('f', 1:3, false);          % has silenceMask, no failComment
A = struct('fileName', {}, 'val', {});
gs = [4 6 8];
for j = 1:3
    A(j).fileName = sprintf('f%02d', gs(j));
    A(j).val = gs(j);
    if j == 2
        A(j).failComment = 'Skipped';          % field preload/B lack
    else
        A(j).silenceMask = false(1, 2);
    end
end
B = mkEntries('f', [5 7 9], false);
merged = scatterStructArrays(9, {preload, A, B}, {1:3, gs, [5 7 9]});
allPass = check('fields: has failComment field', ...
    any(strcmp(fieldnames(merged), 'failComment'))) && allPass;
allPass = check('fields: has silenceMask field', ...
    any(strcmp(fieldnames(merged), 'silenceMask'))) && allPass;
allPass = check('fields: skipped entry (global 6) failComment set', ...
    strcmp(merged(6).failComment, 'Skipped')) && allPass;
allPass = check('fields: skipped entry (global 6) silenceMask empty', ...
    isempty(merged(6).silenceMask)) && allPass;
allPass = check('fields: processed entry (global 1) failComment empty', ...
    isempty(merged(1).failComment)) && allPass;

% --- Case 4: empty preload (fresh year) is dropped --------------------------
A = mkEntries('f', 1:3, false);
B = mkEntries('f', 4:6, false);
merged = scatterStructArrays(6, {struct([]), A, B}, {[], 1:3, 4:6});
allPass = check('empty-preload: count = 6', numel(merged) == 6) && allPass;
allPass = check('empty-preload: order (val = 1..6)', isequal([merged.val], 1:6)) && allPass;

% --- Case 5: size mismatch is a loud error ----------------------------------
allPass = check('errors when a range size != its piece size', ...
    throwsError(@() scatterStructArrays(6, {mkEntries('f', 1:3, false)}, {1:2}))) && allPass;

if allPass
    fprintf('\nALL scatterStructArrays TESTS PASSED.\n');
else
    error('scatterStructArrays tests FAILED - see checks above.');
end
end

% ------------------------------------------------------------------------
function s = mkEntries(prefix, globals, ~)
% Build a struct array with fileName/val/silenceMask, val = the global index.
s = struct('fileName', {}, 'val', {}, 'silenceMask', {});
for j = 1:numel(globals)
    g = globals(j);
    s(j).fileName = sprintf('%s%02d', prefix, g);
    s(j).val = g;
    s(j).silenceMask = false(1, 4);
end
end

% ------------------------------------------------------------------------
function ok = throwsError(fn)
ok = false;
try
    fn();
catch
    ok = true;
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
