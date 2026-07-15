function test_planLengthRoutedSplit()
% Verify the length-routed dual-GPU split (Functions/planLengthRoutedSplit.m):
%   - the primary GPU gets the LARGEST files, the secondary the SMALLEST
%     (every primary file is >= every secondary file by size);
%   - the two ranges partition the remaining files (disjoint, complete);
%   - the primary's realized byte share tracks the requested work fraction;
%   - BOTH workers always get at least one file (clamping), even at extreme
%     fractions or when nearly every file is large;
%   - the split is deterministic and independent of input ordering (so each
%     worker's file list is identical across restarts -> cache resume is safe).
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(scriptDir), 'Functions'))

allPass = true;

% --- Case 1: uniform sizes -> count split (0.7 * 10 = 7 to primary) ---------
sizes = 100 * ones(1, 10);
idx = 1:10;
[a, b, info] = planLengthRoutedSplit(sizes, idx, 0.7);
allPass = check('uniform: primary gets 7 files', info.nPrimary == 7) && allPass;
allPass = check('uniform: secondary gets 3 files', info.nSecondary == 3) && allPass;
allPass = check('uniform: partition complete + disjoint', isPartition(a, b, idx)) && allPass;
allPass = check('uniform: realized frac = 0.7', abs(info.primaryFileFractionRealized - 0.7) < 1e-9) && allPass;

% --- Case 2: mixed sizes -> largest 4 to primary, non-contiguous ranges -----
% sizes desc: 200(5) 100(2) 20(3) 10(1) 5(4) 1(6); round(0.6*6)=4 -> primary
% gets the 4 largest {5,2,3,1}, secondary gets {4,6}.
sizes = [10 100 20 5 200 1];
idx = 1:6;
[a, b] = planLengthRoutedSplit(sizes, idx, 0.6);
allPass = check('mixed: 4 largest -> primary', isequal(a, [1 2 3 5])) && allPass;
allPass = check('mixed: 2 smallest -> secondary', isequal(b, [4 6])) && allPass;
allPass = check('mixed: primary range is non-contiguous', ~isequal(a, a(1):a(end))) && allPass;
allPass = check('mixed: every primary file >= every secondary file', ...
    lengthRoutedInvariant(sizes, idx, a, b)) && allPass;
allPass = check('mixed: partition complete + disjoint', isPartition(a, b, idx)) && allPass;

% --- Case 3: length-routing invariant on a random-but-fixed size vector -----
sizes = [37 5 900 900 12 450 3 88 210 6 6 1000 44 44 44];
idx = 1:numel(sizes);
[a, b] = planLengthRoutedSplit(sizes, idx, 0.75);
allPass = check('random: length-routing invariant holds', ...
    lengthRoutedInvariant(sizes, idx, a, b)) && allPass;
allPass = check('random: partition complete + disjoint', isPartition(a, b, idx)) && allPass;

% --- Case 4: clamping -> secondary always keeps >= 1 file -------------------
sizes = 100 * ones(1, 10);
idx = 1:10;
[a, b] = planLengthRoutedSplit(sizes, idx, 0.999);
allPass = check('clamp high frac: secondary keeps >= 1 file', numel(b) >= 1) && allPass;
allPass = check('clamp high frac: primary keeps >= 1 file', numel(a) >= 1) && allPass;
[a, b] = planLengthRoutedSplit(sizes, idx, 1e-4);
allPass = check('clamp low frac: primary keeps >= 1 file', numel(a) >= 1) && allPass;
allPass = check('clamp low frac: secondary keeps >= 1 file', numel(b) >= 1) && allPass;

% --- Case 5: mostly-long year -> secondary still gets the shortest files ----
% 18 four-hour files and 2 short files; the secondary should get the 2 short
% ones (and only those) at a high primary fraction.
sizes = [repmat(1e7, 1, 18), 5e4, 6e4];
idx = 1:20;
[a, b] = planLengthRoutedSplit(sizes, idx, 0.9);
allPass = check('mostly-long: secondary gets only the 2 short files', ...
    isequal(b, [19 20])) && allPass;
allPass = check('mostly-long: length-routing invariant holds', ...
    lengthRoutedInvariant(sizes, idx, a, b)) && allPass;

% --- Case 6: determinism / order-independence ------------------------------
sizes = [37 5 900 900 12 450 3 88 210 6 6 1000 44 44 44];
idx = 1:numel(sizes);
[a1, b1] = planLengthRoutedSplit(sizes, idx, 0.65);
perm = [12 3 7 1 15 9 4 2 11 6 14 5 8 13 10];   % arbitrary fixed permutation
[a2, b2] = planLengthRoutedSplit(sizes(perm), idx(perm), 0.65);
allPass = check('deterministic: primary range order-independent', isequal(a1, a2)) && allPass;
allPass = check('deterministic: secondary range order-independent', isequal(b1, b2)) && allPass;

% --- Case 7: input validation ----------------------------------------------
allPass = check('errors on < 2 files', throwsError(@() planLengthRoutedSplit(100, 1, 0.7))) && allPass;
allPass = check('errors on fraction >= 1', throwsError(@() planLengthRoutedSplit([1 2], [1 2], 1))) && allPass;
allPass = check('errors on fraction <= 0', throwsError(@() planLengthRoutedSplit([1 2], [1 2], 0))) && allPass;
allPass = check('errors on length mismatch', throwsError(@() planLengthRoutedSplit([1 2 3], [1 2], 0.5))) && allPass;

if allPass
    fprintf('\nALL planLengthRoutedSplit TESTS PASSED.\n');
else
    error('planLengthRoutedSplit tests FAILED - see checks above.');
end
end

% ------------------------------------------------------------------------
function ok = isPartition(a, b, idx)
% True if a and b are disjoint and their union equals idx.
ok = isempty(intersect(a, b)) && isequal(sort([a(:); b(:)]).', sort(idx(:)).');
end

% ------------------------------------------------------------------------
function ok = lengthRoutedInvariant(sizes, idx, a, b)
% True if every file assigned to the primary is at least as large as every
% file assigned to the secondary (the defining property of length routing).
sa = sizes(ismember(idx, a));
sb = sizes(ismember(idx, b));
ok = isempty(sa) || isempty(sb) || min(sa) >= max(sb);
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
