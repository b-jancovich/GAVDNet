function silenceMask = detectSilentRegions(audioIn, fs, minSilenceDuration)
% DETECTSILENTREGIONS Detect silent or near-silent regions in an audio signal
%
% silenceMask = detectSilentRegions(audioIn, fs, minSilenceDuration)
%
% Inputs:
%   audioIn          - Audio signal vector (mono)
%   fs               - Sample rate (Hz)
%   minSilenceDuration - Minimum duration of silence to report (seconds)
%
% Output:
%   silenceMask      - Logical vector same size as audioIn, with true
%                      indicating silent samples and false for non-silent
%
% This function uses adaptive RMS-based detection with statistical thresholding
% to robustly identify silent regions across signals with varying amplitude ranges.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Input validation
if ~isvector(audioIn)
    error('audioIn must be a vector');
end

% Ensure column vector
audioIn = audioIn(:);

% GPU array data is extremely slow... Come back to CPU for this funciton
if isgpuarray(audioIn)
    audioIn = gather(audioIn);
end

% Initialize silence mask
silenceMask = false(size(audioIn));

% Parameters for RMS calculation
windowDuration = 0.02; % 20 ms window (typical for speech/audio analysis)
windowSamples = round(windowDuration * fs);
hopSamples = round(windowSamples / 2); % 50% overlap

% Calculate RMS values using sliding window
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

% Remove any zero RMS values before calculating statistics
nonZeroRMS = rmsValues(rmsValues > 0);

if isempty(nonZeroRMS)
    % All zeros - entire signal is silent
    silenceMask(:) = true;
    return;
end

% Calculate adaptive threshold using percentile-based approach
% Use log domain for better handling of wide dynamic ranges
logRMS = log10(nonZeroRMS + eps); % Add eps to avoid log(0)

% Calculate statistics in log domain
medianLogRMS = median(logRMS);
madLogRMS = median(abs(logRMS - medianLogRMS)); % Median absolute deviation

% Set threshold at median minus k*MAD (in log domain)
% This adapts to the signal's dynamic range
k = 3; % Threshold factor (adjustable for sensitivity)
thresholdLog = medianLogRMS - k * madLogRMS;
threshold = 10^thresholdLog;

% Apply minimum threshold to avoid noise floor issues
noiseFloor = max(nonZeroRMS) * 1e-6;
threshold = max(threshold, noiseFloor);

% Find regions below threshold
belowThreshold = rmsValues < threshold;

% Convert RMS indices back to sample indices and mark silent regions
silentRegions = [];
inSilence = false;
startSilence = 0;

for i = 1:length(belowThreshold)
    if belowThreshold(i) && ~inSilence
        % Start of silent region
        inSilence = true;
        startSilence = (i-1) * hopSamples + 1;
    elseif ~belowThreshold(i) && inSilence
        % End of silent region
        inSilence = false;
        endSilence = (i-1) * hopSamples + windowSamples - 1;
        endSilence = min(endSilence, length(audioIn));

        % Check duration
        duration = (endSilence - startSilence + 1) / fs;
        if duration >= minSilenceDuration
            silentRegions = [silentRegions, [startSilence; endSilence]];
        end
    end
end

% Handle case where silence extends to end of signal
if inSilence
    endSilence = length(audioIn);
    duration = (endSilence - startSilence + 1) / fs;
    if duration >= minSilenceDuration
        silentRegions = [silentRegions, [startSilence; endSilence]];
    end
end

% Merge adjacent silent regions
if size(silentRegions, 2) > 1
    merged = [];
    currentStart = silentRegions(1, 1);
    currentEnd = silentRegions(2, 1);

    for i = 2:size(silentRegions, 2)
        % Check if regions are adjacent (within one window)
        if silentRegions(1, i) - currentEnd <= windowSamples
            % Extend current region
            currentEnd = silentRegions(2, i);
        else
            % Save current region and start new one
            merged = [merged, [currentStart; currentEnd]];
            currentStart = silentRegions(1, i);
            currentEnd = silentRegions(2, i);
        end
    end

    % Add last region
    merged = [merged, [currentStart; currentEnd]];
    silentRegions = merged;
end

% Set mask to true for all detected silent regions
for i = 1:size(silentRegions, 2)
    startIdx = silentRegions(1, i);
    endIdx = silentRegions(2, i);
    silenceMask(startIdx:endIdx) = true;
end

% Guard against marking entire file as silent
if all(silenceMask)
    silenceMask(:) = false;
end
end