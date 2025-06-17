function maskFrames = maskBuffer(mask, featureVectorsPerSequence, overlapPercent)
% Breaks masks into overlapping frames. When the mask cannot be divided
% into an integer number of frames, leftover elements are kept as
% as the final frame. Only for use with RNN style networks that can handle
% inputs with variable sequence length.
%
% Inputs:
%   mask - vector indicating signal presence (1) or absence (0) for every
%   time bin of the 'features' spectrograms.
%   featureVectorsPerSequence - Total number of time bins in the spectrograms
%   overlapPercent - percent of frame overlap (0-100)
%   
% Outputs:
%   maskFrames = cell array containing frames. 
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
arguments
    mask {validateattributes(mask,{'single','double'},{'nonempty','vector','real','finite'},'maskBuffer','mask')}
    featureVectorsPerSequence {validateattributes(featureVectorsPerSequence,{'single','double'},{'nonempty','scalar','real','finite','positive','integer'},'maskBuffer','featureVectorsPerSequence')}
    overlapPercent {validateattributes(overlapPercent,{'single','double'},{'nonempty','scalar','real','finite','>=',0,'<',100},'maskBuffer','overlapPercent')}
end

featureVectorOverlap = round(featureVectorsPerSequence * (overlapPercent/100));
hopLength = featureVectorsPerSequence - featureVectorOverlap;
N = floor((size(mask,2) - featureVectorsPerSequence)/hopLength) + 1;

% Check for leftover feature vectors after the last complete frame
lastCompleteFrameEnd = 1 + (N-1)*hopLength + featureVectorsPerSequence - 1;
hasLeftover = lastCompleteFrameEnd < size(mask,2);

% Allocate cell array for frames (including potential leftover frame)
if hasLeftover
    maskFrames = cell(N+1,1);
else
    maskFrames = cell(N,1);
end

% Process complete frames
idx = 1;
for i = 1:N
    % Extract frame
    maskFrames{i} = mask(:,idx:idx + featureVectorsPerSequence - 1);
    idx = idx + hopLength;
end

% Process leftover frame if it exists
if hasLeftover
    leftoverStart = 1 + N*hopLength;
    maskFrames{N+1} = mask(:, leftoverStart:end);
end

end

