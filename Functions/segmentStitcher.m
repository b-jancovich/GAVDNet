function [probabilities, varargout] = segmentStitcher(probs, splitIndices, hopLen, varargin)
% SEGMENTSTITCHER Stitches together probability vectors from overlapping audio segments
%
% probabilities = segmentStitcher(probs, splitIndices, hopLen)
% [probabilities, features] = segmentStitcher(probs, splitIndices, hopLen, featuresSegments)
%
% INPUTS:
%   probs            - Cell array containing probability vectors for each segment
%   splitIndices     - Nx2 matrix of [startIdx, endIdx] for each segment in audio samples
%   hopLen           - Hop length in samples used for spectrogram computation
%   featuresSegments - (optional) Cell array containing features matrices for each segment
%
% OUTPUTS:
%   probabilities - 1xT vector of stitched probabilities where T is the total
%                  number of time bins for the unsegmented audio
%   features      - (optional) 40xT matrix of stitched features when featuresSegments provided
%
% This function handles overlapping segments by averaging probability values
% and features in overlapping regions.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Convert audio sample indices to spectrogram time bin indices
timeIndices = round(splitIndices / hopLen) + 1;

% Calculate total length of final probability vector
totalTimeBins = timeIndices(end, 2);

% Initialize output arrays
probabilities = zeros(1, totalTimeBins);
overlapCounts = zeros(1, totalTimeBins);

% Stitch segments together
for i = 1:length(probs)
    startBin = timeIndices(i, 1);
    segmentLength = length(probs{i});
    endBin = startBin + segmentLength - 1;
    
    % Ensure we don't exceed the total length
    endBin = min(endBin, totalTimeBins);
    actualLength = endBin - startBin + 1;
    
    % Add this segment's probabilities to the output
    probabilities(startBin:endBin) = probabilities(startBin:endBin) + probs{i}(1:actualLength);
    overlapCounts(startBin:endBin) = overlapCounts(startBin:endBin) + 1;
end

% Average overlapping regions
probabilities = probabilities ./ overlapCounts;

% Process features if provided
if nargin > 3 && ~isempty(varargin{1})
    featuresSegments = varargin{1};
    
    % Get number of frequency bins from first segment
    numFreqBins = size(featuresSegments{1}, 1);
    
    % Initialize output arrays for features
    features = zeros(numFreqBins, totalTimeBins);
    featureOverlapCounts = zeros(numFreqBins, totalTimeBins);
    
    % Stitch feature segments together
    for i = 1:length(featuresSegments)
        startBin = timeIndices(i, 1);
        segmentLength = size(featuresSegments{i}, 2);
        endBin = startBin + segmentLength - 1;
        
        % Ensure we don't exceed the total length
        endBin = min(endBin, totalTimeBins);
        actualLength = endBin - startBin + 1;
        
        % Add this segment's features to the output
        features(:, startBin:endBin) = features(:, startBin:endBin) + featuresSegments{i}(:, 1:actualLength);
        featureOverlapCounts(:, startBin:endBin) = featureOverlapCounts(:, startBin:endBin) + 1;
    end
    
    % Average overlapping regions
    features = features ./ featureOverlapCounts;
    varargout{1} = features;
end

end

% function probabilities = segmentStitcher(probs, splitIndices, hopLen)
% % SEGMENTSTITCHER Stitches together probability vectors from overlapping 
% % audio segments of nonuniform size. TO be used in conjunction with
% % eventSplitter().
% %
% % probabilities = segmentStitcher(probs, splitIndices, hopLen)
% %
% % INPUTS:
% %   probs        - Cell array containing probability vectors for each segment
% %   splitIndices - Nx2 matrix of [startIdx, endIdx] for each segment in audio samples
% %   hopLen       - Hop length in samples used for spectrogram computation
% %
% % OUTPUTS:
% %   probabilities - 1xT vector of stitched probabilities where T is the total
% %                  number of time bins for the unsegmented audio
% %
% % This function handles overlapping segments by averaging probability values
% % in overlapping regions.
% %
% % Ben Jancovich, 2025
% % Centre for Marine Science and Innovation
% % School of Biological, Earth and Environmental Sciences
% % University of New South Wales, Sydney, Australia
% %
% 
% % Convert audio sample indices to spectrogram time bin indices
% timeIndices = round(splitIndices / hopLen) + 1;
% 
% % Calculate total length of final probability vector
% totalTimeBins = timeIndices(end, 2);
% 
% % Initialize output arrays
% probabilities = zeros(1, totalTimeBins);
% overlapCounts = zeros(1, totalTimeBins);
% 
% % Stitch segments together
% for i = 1:length(probs)
%     startBin = timeIndices(i, 1);
%     segmentLength = length(probs{i});
%     endBin = startBin + segmentLength - 1;
% 
%     % Ensure we don't exceed the total length
%     endBin = min(endBin, totalTimeBins);
%     actualLength = endBin - startBin + 1;
% 
%     % Add this segment's probabilities to the output
%     probabilities(startBin:endBin) = probabilities(startBin:endBin) + probs{i}(1:actualLength);
%     overlapCounts(startBin:endBin) = overlapCounts(startBin:endBin) + 1;
% end
% 
% % Average overlapping regions
% probabilities = probabilities ./ overlapCounts;
% 
% end