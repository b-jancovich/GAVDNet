function test_detectSilentRegions_equivalence()
% Verify the vectorised detectSilentRegions (A2) produces bit-identical
% silence masks to the original per-window for-loop implementation.
%
% Self-contained: embeds the ORIGINAL algorithm as a local function and
% compares against the live Functions\detectSilentRegions.m over a range of
% representative signals. Runs on CPU only; does not touch the base
% workspace (it is a function) and does not use the GPU.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

thisFile = mfilename('fullpath');
% Locate the repo Functions folder relative to a known checkout path.
functionsDir = 'C:\Users\z5439673\Git\GAVDNet\Functions';
addpath(functionsDir)

fs = 250;                 % Hz, matches the Chagos DGS archive
minSilenceDuration = 1;   % s, matches config

cases = struct('name', {}, 'x', {});

% 1. Pure noise (no silence)
rng(1);
cases(end+1) = mkcase('white-noise-600s', single(0.1 * randn(fs*600, 1)));

% 2. All zeros (guarded all-silent path)
cases(end+1) = mkcase('all-zeros-120s', zeros(fs*120, 1, 'single'));

% 3. Silence -> loud -> silence (clear regions)
x = single([0.001*randn(fs*30,1); 0.5*randn(fs*20,1); 0.001*randn(fs*30,1)]);
cases(end+1) = mkcase('sil-loud-sil', x);

% 4. Mixed intermittent activity
x = single(0.002*randn(fs*300,1));
for k = 1:10
    a = randi(fs*300 - fs*5);
    x(a:a+fs*3) = x(a:a+fs*3) + single(0.4*randn(fs*3+1,1));
end
cases(end+1) = mkcase('intermittent', x);

% 5. Near-threshold amplitude (stresses the < comparison)
x = single(0.01 + 0.001*randn(fs*120,1));
x(fs*60:fs*61) = x(fs*60:fs*61) + single(0.05);
cases(end+1) = mkcase('near-threshold', x);

% 6. Long multi-hour file at 250 Hz (2003-style, ~4 h)
rng(2);
cases(end+1) = mkcase('long-4h-250Hz', single(0.05*randn(fs*4*3600, 1)));

% 7. Short 2-minute file at 250 Hz (2006-style)
cases(end+1) = mkcase('short-2min-250Hz', single(0.05*randn(fs*120, 1)));

% 8. Very short (< one window) -> all-silent early-return path
cases(end+1) = mkcase('tiny-3samp', single([0.1; -0.2; 0.05]));

% 9. Odd length, single sample above zero
cases(end+1) = mkcase('odd-len', single(0.03*randn(12345,1)));

allPass = true;
for i = 1:numel(cases)
    maskNew = detectSilentRegions(cases(i).x, fs, minSilenceDuration);
    maskOld = detectSilentRegions_OLD(cases(i).x, fs, minSilenceDuration);
    ok = isequal(maskNew, maskOld);
    nDiff = sum(maskNew(:) ~= maskOld(:));
    if ok
        fprintf('PASS  %-20s  (n=%d, silent=%d)\n', cases(i).name, ...
            numel(cases(i).x), sum(maskNew));
    else
        allPass = false;
        fprintf('FAIL  %-20s  differing samples = %d of %d\n', ...
            cases(i).name, nDiff, numel(cases(i).x));
    end
end

if allPass
    fprintf('\nALL CASES BIT-IDENTICAL. A2 verified.\n');
else
    error('A2 equivalence FAILED - see cases above.');
end
end

% ------------------------------------------------------------------------
function c = mkcase(name, x)
c.name = name;
c.x = x;
end

% ------------------------------------------------------------------------
function silenceMask = detectSilentRegions_OLD(audioIn, fs, minSilenceDuration)
% VERBATIM copy of the pre-A2 implementation (per-window for-loop) for
% equivalence testing only.
if ~isvector(audioIn)
    error('audioIn must be a vector');
end
audioIn = audioIn(:);
if isgpuarray(audioIn)
    audioIn = gather(audioIn);
end
silenceMask = false(size(audioIn));
windowDuration = 0.02;
windowSamples = round(windowDuration * fs);
hopSamples = round(windowSamples / 2);

numWindows = floor((length(audioIn) - windowSamples) / hopSamples) + 1;
rmsValues = zeros(numWindows, 1);
for i = 1:numWindows
    startIdx = (i-1) * hopSamples + 1;
    endIdx = startIdx + windowSamples - 1;
    if endIdx <= length(audioIn)
        windowData = audioIn(startIdx:endIdx);
        rmsValues(i) = sqrt(mean(windowData.^2));
    end
end

nonZeroRMS = rmsValues(rmsValues > 0);
if isempty(nonZeroRMS)
    silenceMask(:) = true;
    return;
end
logRMS = log10(nonZeroRMS + eps);
medianLogRMS = median(logRMS);
madLogRMS = median(abs(logRMS - medianLogRMS));
k = 3;
thresholdLog = medianLogRMS - k * madLogRMS;
threshold = 10^thresholdLog;
noiseFloor = max(nonZeroRMS) * 1e-6;
threshold = max(threshold, noiseFloor);
belowThreshold = rmsValues < threshold;

silentRegions = [];
inSilence = false;
startSilence = 0;
for i = 1:length(belowThreshold)
    if belowThreshold(i) && ~inSilence
        inSilence = true;
        startSilence = (i-1) * hopSamples + 1;
    elseif ~belowThreshold(i) && inSilence
        inSilence = false;
        endSilence = (i-1) * hopSamples + windowSamples - 1;
        endSilence = min(endSilence, length(audioIn));
        duration = (endSilence - startSilence + 1) / fs;
        if duration >= minSilenceDuration
            silentRegions = [silentRegions, [startSilence; endSilence]];
        end
    end
end
if inSilence
    endSilence = length(audioIn);
    duration = (endSilence - startSilence + 1) / fs;
    if duration >= minSilenceDuration
        silentRegions = [silentRegions, [startSilence; endSilence]];
    end
end
if size(silentRegions, 2) > 1
    merged = [];
    currentStart = silentRegions(1, 1);
    currentEnd = silentRegions(2, 1);
    for i = 2:size(silentRegions, 2)
        if silentRegions(1, i) - currentEnd <= windowSamples
            currentEnd = silentRegions(2, i);
        else
            merged = [merged, [currentStart; currentEnd]];
            currentStart = silentRegions(1, i);
            currentEnd = silentRegions(2, i);
        end
    end
    merged = [merged, [currentStart; currentEnd]];
    silentRegions = merged;
end
for i = 1:size(silentRegions, 2)
    startIdx = silentRegions(1, i);
    endIdx = silentRegions(2, i);
    silenceMask(startIdx:endIdx) = true;
end
if all(silenceMask)
    silenceMask(:) = false;
end
end
