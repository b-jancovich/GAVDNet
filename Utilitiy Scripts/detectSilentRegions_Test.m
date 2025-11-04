clear
close all
clc

% Detect silence parameters
minSilenceDuration = 1;

% Test params
min_max_silence_durations = [0.1, 600]; % seconds
numTests = 10;
scalingAbsMax = [1e-5, 1];

% Read in the audio
[audioIn, fs] = audioread("D:\GAVDNet\Chagos_DGS\Test Data\H08S1_071102-200000_TypicalLotsOfCalls.wav");

% Convert silence durations from seconds to samples
min_max_silence_samples = round(min_max_silence_durations * fs);

% List of signal scales to test
scalingAbsMaxList = linspace(scalingAbsMax(1), scalingAbsMax(2), numTests);

% Run tests
for i = 1:numTests
    % Scale the entire signal
    audioTest = scaleSignal(audioIn, scalingAbsMaxList(i));
    
    % Set scaling factor for silent regions relative to the signal amplitude
    % Must be aggressive enough to trigger the adaptive thresholding algorithm
    % The function uses median - 3*MAD threshold, so regions must be genuinely quiet
    scalingFactor = 0.00001 + rand() * 0.0001; % Random between 0.001% and 0.01% of original amplitude
    
    % Track ground truth silence regions for validation
    groundTruthRegions = [];
    
    % The first test should make the start of the signal silent or near silent
    if i == 1
        % Set silence indices
        silenceStartIdx = 1;
        silenceEndIdx = randi(min_max_silence_samples(2));
        % Make the signal at those indices silent or near silent
        audioTest(silenceStartIdx:silenceEndIdx) = audioTest(silenceStartIdx:silenceEndIdx) .* scalingFactor;
        groundTruthRegions = [groundTruthRegions; silenceStartIdx, silenceEndIdx];
        
    % The second test should make the end of the signal silent or near silent
    elseif i == 2
        % Set silence indices
        silenceStartIdx = length(audioTest) - randi(min_max_silence_samples(2)) + 1;
        silenceEndIdx = length(audioTest);
        % Make the signal at those indices silent or near silent
        audioTest(silenceStartIdx:silenceEndIdx) = audioTest(silenceStartIdx:silenceEndIdx) .* scalingFactor;
        groundTruthRegions = [groundTruthRegions; silenceStartIdx, silenceEndIdx];
        
    % Third test: Create both long and short silence regions to test duration filtering
    elseif i == 3
        % Create a short region (should be ignored)
        shortDuration = round(minSilenceDuration * fs * 0.5); % Half the minimum duration
        shortStart = randi(round(length(audioTest) * 0.3)); % Start in first third
        shortEnd = shortStart + shortDuration - 1;
        audioTest(shortStart:shortEnd) = audioTest(shortStart:shortEnd) .* scalingFactor;
        groundTruthRegions = [groundTruthRegions; shortStart, shortEnd];
        
        % Create a long region (should be detected) - ensure it fits
        longDuration = round(minSilenceDuration * fs * 2); % Twice the minimum duration
        longStart = round(length(audioTest) * 0.6); % Start in second third
        longEnd = min(longStart + longDuration - 1, length(audioTest));
        audioTest(longStart:longEnd) = audioTest(longStart:longEnd) .* scalingFactor;
        groundTruthRegions = [groundTruthRegions; longStart, longEnd];
        
    % The remaining tests should make random bits of the signal silent or near silent
    else
        % Select a random part of the signal to make silent or near silent
        maxStartIdx = length(audioTest) - min_max_silence_samples(2);
        silenceStartIdx = randi(maxStartIdx);
        silenceDuration = randi([min_max_silence_samples(1), min_max_silence_samples(2)]);
        silenceEndIdx = min(silenceStartIdx + silenceDuration - 1, length(audioTest));
        % Make the signal at those indices silent or near silent
        audioTest(silenceStartIdx:silenceEndIdx) = audioTest(silenceStartIdx:silenceEndIdx) .* scalingFactor;
        groundTruthRegions = [groundTruthRegions; silenceStartIdx, silenceEndIdx];
    end
    
    % Detect the silence
    silenceMask = detectSilentRegions(audioTest, fs, minSilenceDuration);
    
    % Validate results
    validateSilenceDetection(groundTruthRegions, silenceMask, fs, minSilenceDuration, i);
    
    % Plot results
    duration = length(audioTest) / fs;
    dt = 1/fs;
    t = 0:dt:duration-dt;
    figure;
    yyaxis left
    plot(t, audioTest);
    yyaxis right
    plot(t, silenceMask, 'r--');
    title(sprintf('Test %d: Signal with detected silence', i));
    xlabel('Time (s)');
    ylabel('Amplitude');
    legend('Audio', 'Detected Silence', 'Location', 'best');
    
end

%% Helper functions
function out = scaleSignal(in, scaleAbsMax)
% SCALESIGNAL Scale signal to specified maximum absolute value
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Find maximum absolute value to preserve dynamic range
max_abs_value = max(abs(in(:)));

% Calculate scaling factor
if max_abs_value > 0
    scale_factor = scaleAbsMax / max_abs_value;
    out = in * scale_factor;
else
    % Handle edge case of all-zero signal
    out = in;
end
end

function validateSilenceDetection(groundTruthRegions, silenceMask, fs, minSilenceDuration, testNum)
% VALIDATESILENCEDETECTION Compare expected vs detected silence regions
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Convert silence mask to regions
detectedRegions = maskToRegions(silenceMask);

% Check each ground truth region
for i = 1:size(groundTruthRegions, 1)
    gtStart = groundTruthRegions(i, 1);
    gtEnd = groundTruthRegions(i, 2);
    gtDuration = (gtEnd - gtStart + 1) / fs;
    
    if gtDuration >= minSilenceDuration
        % This region should be detected
        found = false;
        for j = 1:size(detectedRegions, 1)
            detStart = detectedRegions(j, 1);
            detEnd = detectedRegions(j, 2);
            
            % Check if boundaries match within 0.2s tolerance
            tol = fs*0.2;
            if abs(detStart - gtStart) <= tol && abs(detEnd - gtEnd) <= tol
                found = true;
                fprintf('Test %d: ✓ Silence region [%d-%d] correctly detected as [%d-%d]\n', ...
                    testNum, gtStart, gtEnd, detStart, detEnd);
                break;
            end
        end
        
        if ~found
            fprintf('Test %d: ✗ Expected silence region [%d-%d] (%.3fs) not detected\n', ...
                testNum, gtStart, gtEnd, gtDuration);
        end
    else
        % This region should NOT be detected (too short)
        found = false;
        for j = 1:size(detectedRegions, 1)
            detStart = detectedRegions(j, 1);
            detEnd = detectedRegions(j, 2);
            
            % Check if there's overlap with this short region
            if detStart <= gtEnd && detEnd >= gtStart
                found = true;
                break;
            end
        end
        
        if found
            fprintf('Test %d: ✗ Short silence region [%d-%d] (%.3fs) incorrectly detected\n', ...
                testNum, gtStart, gtEnd, gtDuration);
        else
            fprintf('Test %d: ✓ Short silence region [%d-%d] (%.3fs) correctly ignored\n', ...
                testNum, gtStart, gtEnd, gtDuration);
        end
    end
end

% Report total detected regions
fprintf('Test %d: %d regions detected, %d ground truth regions\n\n', ...
    testNum, size(detectedRegions, 1), size(groundTruthRegions, 1));
end

function regions = maskToRegions(mask)
% MASKTOREGIONS Convert logical mask to start/end indices of true regions
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

regions = [];
if isempty(mask) || ~any(mask)
    return;
end

% Find transitions
maskDiff = diff([false; mask(:); false]);
startIndices = find(maskDiff == 1);
endIndices = find(maskDiff == -1) - 1;

% Combine into regions matrix
regions = [startIndices, endIndices];
end