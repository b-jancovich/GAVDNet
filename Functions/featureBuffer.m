function featureFrames = featureBuffer(features, featureVectorsPerSequence, overlapPercent)
% Breaks spectrograms into overlapping frames. When features 
% cannot be divided into an integer number of frames, leftover feature 
% vectors are kept as as the final frame. Only for use with RNN style 
% networks that can handle inputs with variable sequence length.
%
% Inputs:
%   features - the 'features' spectrogram returned by the preprocessor
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

% Validate inputs
arguments
    features {validateattributes(features,{'single','double'},{'nonempty','2d','real','finite'},'featureBuffer','features')}
    featureVectorsPerSequence {validateattributes(featureVectorsPerSequence,{'single','double'},{'nonempty','scalar','real','finite','positive','integer'},'featureBuffer','featureVectorsPerSequence')}
    overlapPercent {validateattributes(overlapPercent,{'single','double'},{'nonempty','scalar','real','finite','>=',0,'<',100},'featureBuffer','overlapPercent')}
end

featureVectorOverlap = round(featureVectorsPerSequence * (overlapPercent/100));
hopLength = featureVectorsPerSequence - featureVectorOverlap;
N = floor((size(features,2) - featureVectorsPerSequence)/hopLength) + 1;

% Check for leftover feature vectors after the last complete frame
lastCompleteFrameEnd = 1 + (N-1)*hopLength + featureVectorsPerSequence - 1;
hasLeftover = lastCompleteFrameEnd < size(features,2);

% Allocate cell array for featureFrames (including potential leftover frame)
if hasLeftover
    featureFrames = cell(N+1,1);
else
    featureFrames = cell(N,1);
end

% Process complete frames
idx = 1;
for i = 1:N
    % Extract frame
    featureFrames{i} = features(:,idx:idx + featureVectorsPerSequence - 1);

    % standardize
    featureFrames{i} = standardizeMelSpect(featureFrames{i});

    idx = idx + hopLength;
end

% Process leftover frame if they exist
if hasLeftover
    leftoverStart = 1 + N*hopLength;
    featureFrames{N+1}  = features(:, leftoverStart:end);

    % standardize
    featureFrames{N+1} = standardizeMelSpect(featureFrames{N+1});
end
end

%% Helper functions
function y = standardizeMelSpect(x)
%standardizeMelSpect Standardize audio to zero mean and unity standard deviation

amean = mean(x,2);
astd = max(std(x,[],2),1e-10);
y = (x-amean)./astd;
end
