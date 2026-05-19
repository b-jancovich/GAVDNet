function [signalPresenceMask, confidenceScores] = detectChorus(...
   signal, bandOfInterest, middleBandProportion, extraEdgeBandwidth, minChorusDuration, ...
   lowRatioTolerance, ratioThresh, fs)
%
% Function that detects chorusing events by analyzing the spectral energy
% distribution. Specifically looks for high energy concentration in the 
% primary signal band (central "middleBandProportion" of band of interest) 
% compared to the upper and lower edge bands (remaining proportion/2 each).
%
%   Inputs:
%       signal                  - Audio signal [Nx1 real]
%       bandOfInterest          - Frequency band [fMin fMax] (Hz)
%       middleBandProportion    - Middle band width as proportion of 
%                                   bandOfInterest [0:1]
%       minChorusDuration       - Minimum duration of power ratio over 
%                                   threshold to qualify as chorus (s)
%       lowRatioTolerance       - duration of time that power ratio can 
%                                   drop below threshold, while not 
%                                   disqualifying an otherwise valid 
%                                   detection (s)
%       ratioThresh             - Power ratio threshold (dB)
%       fs                      - Sampling frequency (Hz)
%
%   Outputs:
%       signalPresenceMask - Binary detection mask [Nx1 logical]
%       confidenceScores   - Detection confidence scores [Nx1 real]
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%% Input Validation

debug = false;

%% Spectral Analysis Parameters
windowLen = round(fs * 4);          % 4s analysis window
overlap = round(windowLen * 0.5);  % 50% overlap
nfft = 2^nextpow2(windowLen);

% Define frequency bands
totalBand = bandOfInterest(2) - bandOfInterest(1);
edgeBandProportion = (1 - middleBandProportion) / 2;

lowerBandStart = bandOfInterest(1) - extraEdgeBandwidth;
lowerBandEnd = bandOfInterest(1) + (edgeBandProportion * totalBand);
middleBandStart = lowerBandEnd;
middleBandEnd = bandOfInterest(2) - (edgeBandProportion * totalBand);
upperBandStart = middleBandEnd;
upperBandEnd = bandOfInterest(2) + extraEdgeBandwidth;

% Set minimum chorus duration (in Frames)
minConsecutiveFrames = ceil(minChorusDuration * fs / (windowLen - overlap));
nToleranceFrames = ceil(lowRatioTolerance * fs / (windowLen - overlap));

%% Compute Spectral Power

% Compute spectrogram
[S, f, ~] = spectrogram(signal, windowLen, overlap, nfft, fs, 'yaxis');

% Extract Magnitude
S = abs(S);

% % Identify frequency bins for each band
lowerBand = (f >= lowerBandStart) & (f < lowerBandEnd);
middleBand = (f >= middleBandStart) & (f < middleBandEnd);
upperBand = (f >= upperBandStart) & (f <= upperBandEnd);

% Mean energy per band
lowerMagnitude = mean(S(lowerBand, :), 1);
middleMagnitude = mean(S(middleBand, :), 1);
upperMagnitude = mean(S(upperBand, :), 1);

%% Compute Power Ratio

% Convert from Mag to power units
powerLower = lowerMagnitude .^2;
powerMiddle = middleMagnitude .^2;
powerUpper = upperMagnitude .^2;

% Compute mean edge band power
powerEdges = (powerLower + powerUpper)/2;

% Calculate power ratio and convert to dB
energyRatio = powerMiddle ./ powerEdges;
energyRatiodB = 10*log10(energyRatio);

%% Detect Sustained Chorus


% Identify frames exceeding threshold
frameExceedsThresh = energyRatiodB > ratioThresh;

% Merge short gaps (i.e. segments) between true frames if the gap length is
% less than or equal to nToleranceFrames.
mergedMask = frameExceedsThresh;
i = 1;
while i <= length(mergedMask)
    if ~mergedMask(i)
        gapStart = i;
        while i <= length(mergedMask) && ~mergedMask(i)
            i = i + 1;
        end
        gapEnd = i - 1;
        gapLength = gapEnd - gapStart + 1;
        % Only fill the gap if it is bounded by true values on both sides and is within tolerance.
        if gapStart > 1 && i <= length(mergedMask) && gapLength <= nToleranceFrames
            mergedMask(gapStart:gapEnd) = true;
        end
    else
        i = i + 1;
    end
end

% Find sustained periods above threshold
% [runs, lengths] = detectRuns(frameExceedsThresh);
[runs, lengths] = detectRuns(mergedMask);  % Use merged mask instead
validChorus = false(size(frameExceedsThresh));

for i = 1:length(runs)
   if runs(i) && lengths(i) >= minConsecutiveFrames
       startIdx = sum(lengths(1:i-1)) + 1;
       endIdx = startIdx + lengths(i) - 1;
       validChorus(startIdx:endIdx) = true;
   end
end

%% Generate Output Masks

signalPresenceMask = false(size(signal));
confidenceScores = zeros(size(signal));

for i = 1:length(validChorus)
   if validChorus(i)
       startSample = (i-1)*(windowLen-overlap) + 1;
       endSample = min(length(signal), startSample + windowLen - 1);
       signalPresenceMask(startSample:endSample) = true;
       
       % Normalize using sigmoid centered on threshold
       normalizedConfidence = 1 / (1 + exp(-(energyRatiodB(i) - ratioThresh)));
       confidenceScores(startSample:endSample) = normalizedConfidence;
   end
end

% signalPresenceMask = false(size(signal));
% confidenceScores = zeros(size(signal));
% 
% % Map frame detections to samples
% for i = 1:length(validChorus)
%    if validChorus(i)
%        startSample = (i-1)*(windowLen-overlap) + 1;
%        endSample = min(length(signal), startSample + windowLen - 1);
%        signalPresenceMask(startSample:endSample) = true;
%        confidenceScores(startSample:endSample) = energyRatiodB(i);
%    end
% end

%% Debug Visualization
if debug == true
   tEnergy = (0:length(lowerMagnitude)-1) * (length(signal)/fs/length(lowerMagnitude));
   
   figure('Name', sprintf('Chorus Detection Debug'), ...
       'Position', [100 100 1200 600]);
   
   % Plot band energies
   subplot(2,1,1)
   plot(tEnergy, mag2db(lowerMagnitude), 'b-', 'DisplayName', 'Lower Band', ...
       'LineWidth', 0.5);
   hold on
   plot(tEnergy, mag2db(middleMagnitude), 'g-', 'DisplayName', 'Middle Band', ...
       'LineWidth', 0.5);
  plot(tEnergy, mag2db(upperMagnitude), 'r-', 'DisplayName', 'Upper Band', ...
   'LineWidth', 0.5);

   xlabel('Time (s)')
   ylabel('Power (dB)')
   title('Edge Band Energy Levels')
   legend('Location', 'eastoutside')
   grid on
   hold off
   
   % Plot power ratio
   subplot(2,1,2)
   plot(tEnergy, energyRatiodB, 'k-', 'DisplayName', 'Power Ratio', ...
       'LineWidth', 1);
   hold on
   yline(ratioThresh, 'r--', 'DisplayName', 'Threshold');
   
   xlabel('Time (s)')
   ylabel('Power Ratio (dB)')
   title('Detection Metric')
   legend('Location', 'eastoutside')
   grid on
   hold off

   linkaxes(findobj(gcf, 'Type', 'axes'), 'x');
end
end

%% Helper Functions

function [runs, lengths] = detectRuns(x)
% Identifies consecutive runs of true values in logical array
% Returns:
%   runs - logical array indicating if each run contains true values
%   lengths - length of each run
    
% Find edges
edges = diff([false; x(:); false]);
    
% Find start and end indices
startIdx = find(edges == 1);
endIdx = find(edges == -1) - 1;
    
% Calculate run properties
runs = x(startIdx);  % Value at start of each run
lengths = endIdx - startIdx + 1;  % Length of each run
end