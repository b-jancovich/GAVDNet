function [probabilities, varargout] = segmentStitcher(probs, splitIndices, preprocParams, fileFs, varargin)
% SEGMENTSTITCHER Stitches together probability vectors from overlapping audio segments
%
% probabilities = segmentStitcher(probs, splitIndices, preprocParams, fileFs)
% [probabilities, features] = segmentStitcher(probs, splitIndices, preprocParams, fileFs, featuresSegments)
%
% INPUTS:
%   probs            - Cell array containing probability vectors for each segment
%   splitIndices     - Nx2 matrix of [startIdx, endIdx] for each segment in audio samples
%   preprocParams    - Preprocessing parameters structure containing windowLen, hopLen, fsTarget
%   fileFs           - Original file sample rate (Hz)
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

% Unpack preprocessing parameters
windowLen = preprocParams.windowLen;
hopLen = preprocParams.hopLen;
targetFs = preprocParams.fsTarget;

% Calculate total time bins for original unsegmented audio
padLen = ceil(windowLen/2);
originalAudioLength = max(splitIndices(:,2));

% Convert audio length to target sample rate domain for spectrogram calculation
% The splitIndices are in original file sample rate, but spectrograms are computed at targetFs
if fileFs ~= targetFs
    resampledAudioLength = ceil(originalAudioLength * targetFs / fileFs);
else
    resampledAudioLength = originalAudioLength;
end

totalTimeBins = ceil((resampledAudioLength + 2*padLen - windowLen) / hopLen) + 1;

% Convert segment indices to time bin indices
timeIndices = floor((splitIndices - 1) / hopLen) + 1;

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

% Average overlapping regions - only divide where counts > 0 to avoid NaN
nonZeroIdx = overlapCounts > 0;
probabilities(nonZeroIdx) = probabilities(nonZeroIdx) ./ overlapCounts(nonZeroIdx);

% Process features if provided
if nargin > 4 && ~isempty(varargin{1})
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
    
    % Average overlapping regions - only divide where counts > 0 to avoid NaN
    nonZeroIdx = featureOverlapCounts > 0;
    features(nonZeroIdx) = features(nonZeroIdx) ./ featureOverlapCounts(nonZeroIdx);
    varargout{1} = features;
end

end

% This version has issues when the audio at inference is resampled to
% fsTarget.
% function [probabilities, varargout] = segmentStitcher(probs, splitIndices, windowLen, hopLen, varargin)
% % SEGMENTSTITCHER Stitches together probability vectors from overlapping audio segments
% %
% % probabilities = segmentStitcher(probs, splitIndices, hopLen)
% % [probabilities, features] = segmentStitcher(probs, splitIndices, hopLen, featuresSegments)
% %
% % INPUTS:
% %   probs            - Cell array containing probability vectors for each segment
% %   splitIndices     - Nx2 matrix of [startIdx, endIdx] for each segment in audio samples
% %   hopLen           - Hop length in samples used for spectrogram computation
% %   featuresSegments - (optional) Cell array containing features matrices for each segment
% %
% % OUTPUTS:
% %   probabilities - 1xT vector of stitched probabilities where T is the total
% %                  number of time bins for the unsegmented audio
% %   features      - (optional) 40xT matrix of stitched features when featuresSegments provided
% %
% % This function handles overlapping segments by averaging probability values
% % and features in overlapping regions.
% %
% % Ben Jancovich, 2025
% % Centre for Marine Science and Innovation
% % School of Biological, Earth and Environmental Sciences
% % University of New South Wales, Sydney, Australia
% %
% 
% % Calculate total time bins for original unsegmented audio
% padLen = ceil(windowLen/2);
% originalAudioLength = max(splitIndices(:,2));
% totalTimeBins = ceil((originalAudioLength + 2*padLen - windowLen) / hopLen) + 1;
% 
% % Convert segment indices to time bin indices
% timeIndices = floor((splitIndices - 1) / hopLen) + 1;
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
% % Average overlapping regions - only divide where counts > 0 to avoid NaN
% nonZeroIdx = overlapCounts > 0;
% probabilities(nonZeroIdx) = probabilities(nonZeroIdx) ./ overlapCounts(nonZeroIdx);
% 
% % Process features if provided
% if nargin > 3 && ~isempty(varargin{1})
%     featuresSegments = varargin{1};
% 
%     % Get number of frequency bins from first segment
%     numFreqBins = size(featuresSegments{1}, 1);
% 
%     % Initialize output arrays for features
%     features = zeros(numFreqBins, totalTimeBins);
%     featureOverlapCounts = zeros(numFreqBins, totalTimeBins);
% 
%     % Stitch feature segments together
%     for i = 1:length(featuresSegments)
%         startBin = timeIndices(i, 1);
%         segmentLength = size(featuresSegments{i}, 2);
%         endBin = startBin + segmentLength - 1;
% 
%         % Ensure we don't exceed the total length
%         endBin = min(endBin, totalTimeBins);
%         actualLength = endBin - startBin + 1;
% 
%         % Add this segment's features to the output
%         features(:, startBin:endBin) = features(:, startBin:endBin) + featuresSegments{i}(:, 1:actualLength);
%         featureOverlapCounts(:, startBin:endBin) = featureOverlapCounts(:, startBin:endBin) + 1;
%     end
% 
%     % Average overlapping regions - only divide where counts > 0 to avoid NaN
%     nonZeroIdx = featureOverlapCounts > 0;
%     features(nonZeroIdx) = features(nonZeroIdx) ./ featureOverlapCounts(nonZeroIdx);
%     varargout{1} = features;
% end
% 
% end
% 
% % function [probabilities, varargout] = segmentStitcher(probs, splitIndices, windowLen, hopLen, varargin)
% % % SEGMENTSTITCHER Stitches together probability vectors from overlapping audio segments
% % %
% % % probabilities = segmentStitcher(probs, splitIndices, hopLen)
% % % [probabilities, features] = segmentStitcher(probs, splitIndices, hopLen, featuresSegments)
% % %
% % % INPUTS:
% % %   probs            - Cell array containing probability vectors for each segment
% % %   splitIndices     - Nx2 matrix of [startIdx, endIdx] for each segment in audio samples
% % %   hopLen           - Hop length in samples used for spectrogram computation
% % %   featuresSegments - (optional) Cell array containing features matrices for each segment
% % %
% % % OUTPUTS:
% % %   probabilities - 1xT vector of stitched probabilities where T is the total
% % %                  number of time bins for the unsegmented audio
% % %   features      - (optional) 40xT matrix of stitched features when featuresSegments provided
% % %
% % % This function handles overlapping segments by averaging probability values
% % % and features in overlapping regions.
% % %
% % % Ben Jancovich, 2025
% % % Centre for Marine Science and Innovation
% % % School of Biological, Earth and Environmental Sciences
% % % University of New South Wales, Sydney, Australia
% % %
% % 
% % % Calculate total time bins for original unsegmented audio
% % padLen = ceil(windowLen/2);
% % originalAudioLength = max(splitIndices(:,2));
% % totalTimeBins = ceil((originalAudioLength + 2*padLen - windowLen) / hopLen) + 1;
% % 
% % % Convert segment indices to time bin indices
% % timeIndices = floor((splitIndices - 1) / hopLen) + 1;
% % 
% % % Initialize output arrays
% % probabilities = zeros(1, totalTimeBins);
% % overlapCounts = zeros(1, totalTimeBins);
% % 
% % % Stitch segments together
% % for i = 1:length(probs)
% %     startBin = timeIndices(i, 1);
% %     segmentLength = length(probs{i});
% %     endBin = startBin + segmentLength - 1;
% % 
% %     % Ensure we don't exceed the total length
% %     endBin = min(endBin, totalTimeBins);
% %     actualLength = endBin - startBin + 1;
% % 
% %     % Add this segment's probabilities to the output
% %     probabilities(startBin:endBin) = probabilities(startBin:endBin) + probs{i}(1:actualLength);
% %     overlapCounts(startBin:endBin) = overlapCounts(startBin:endBin) + 1;
% % end
% % 
% % % Average overlapping regions
% % probabilities = probabilities ./ overlapCounts;
% % 
% % % Process features if provided
% % if nargin > 3 && ~isempty(varargin{1})
% %     featuresSegments = varargin{1};
% % 
% %     % Get number of frequency bins from first segment
% %     numFreqBins = size(featuresSegments{1}, 1);
% % 
% %     % Initialize output arrays for features
% %     features = zeros(numFreqBins, totalTimeBins);
% %     featureOverlapCounts = zeros(numFreqBins, totalTimeBins);
% % 
% %     % Stitch feature segments together
% %     for i = 1:length(featuresSegments)
% %         startBin = timeIndices(i, 1);
% %         segmentLength = size(featuresSegments{i}, 2);
% %         endBin = startBin + segmentLength - 1;
% % 
% %         % Ensure we don't exceed the total length
% %         endBin = min(endBin, totalTimeBins);
% %         actualLength = endBin - startBin + 1;
% % 
% %         % Add this segment's features to the output
% %         features(:, startBin:endBin) = features(:, startBin:endBin) + featuresSegments{i}(:, 1:actualLength);
% %         featureOverlapCounts(:, startBin:endBin) = featureOverlapCounts(:, startBin:endBin) + 1;
% %     end
% % 
% %     % Average overlapping regions
% %     features = features ./ featureOverlapCounts;
% %     varargout{1} = features;
% % end
% % 
% % end
