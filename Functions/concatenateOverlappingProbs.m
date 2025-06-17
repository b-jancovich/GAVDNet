function concatenatedProbs = concatenateOverlappingProbs(probs, numSpectrogramTimeBins, frameHopLength)
%CONCATENATEOVERLAPPINGPROBS Concatenate overlapping probability vectors
%   This function takes probability vectors from overlapping frames and
%   concatenates them while averaging overlapping regions.
%
%   Inputs:
%   probs - Cell array of probability vectors from each frame
%   numSpectrogramTimeBins - Number of time bins in original spectrogram before frame buffering
%   frameHopLength - Hop length between frames (number of time bins)
%
%   Outputs:
%   concatenatedProbs - Single probability vector for the entire spectrogram
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Initialize output arrays
concatenatedProbs = zeros(numSpectrogramTimeBins, 1);
counts = zeros(numSpectrogramTimeBins, 1);

% Process each frame
for i = 1:length(probs)
    % Calculate start index for this frame (same indexing as featureBuffer)
    startIdx = 1 + (i-1) * frameHopLength;
    
    % Calculate end index based on the actual length of this probability vector
    endIdx = startIdx + length(probs{i}) - 1;

    % Ensure we don't exceed the original spectrogram bounds
    if startIdx > numSpectrogramTimeBins
        break;
    end
    
    endIdx = min(endIdx, numSpectrogramTimeBins);
    validLength = endIdx - startIdx + 1;

    % Accumulate probabilities and counts
    concatenatedProbs(startIdx:endIdx) = concatenatedProbs(startIdx:endIdx) + (probs{i}(1:validLength))';
    counts(startIdx:endIdx) = counts(startIdx:endIdx) + 1;
end

% Average the probabilities where there were overlaps
nonZeroIdx = counts > 0;
concatenatedProbs(nonZeroIdx) = concatenatedProbs(nonZeroIdx) ./ counts(nonZeroIdx);
end