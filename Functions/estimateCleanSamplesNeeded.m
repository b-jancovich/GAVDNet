function numSamplesNeeded = estimateCleanSamplesNeeded(numSequences, sequenceDuration, ICI, ICI_variation)
% CALCULATECLEANSAMPLESNEEDED Calculates the number of clean call samples needed
% to construct the specified number of synthetic sequences
%
% Inputs:
%   numSequences - number of sequences to generate
%   sequenceDuration - duration of each sequence in seconds
%   ICI - Inter-Call Interval in seconds (average gap between calls)
%   ICI_variation - variation in ICI (± seconds)
%
% Outputs:
%   numSamplesNeeded - estimated number of clean samples needed
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Calculate maximum calls per sequence for a single individual
maxCallsPerSequence = floor(sequenceDuration / (ICI + ICI_variation));

% Number of single-individual sequences (half of total)
numSingleIndividualSequences = ceil(numSequences / 2);

% Number of multi-individual sequences (half of total)
numMultiIndividualSequences = floor(numSequences / 2);

% For single individual sequences:
% - Average number of calls will be around half of maximum for partial bouts
% - Some will be full bouts, some will be partial
averageCallsPerSingleSequence = maxCallsPerSequence * 0.6; % Accounting for partial bouts

% For multi-individual sequences:
% - Average of 3.5 individuals per sequence (between 2-5)
% - Each individual has calls spanning about 60% of their segment
averageIndividualsPerMultiSequence = 3.5; 
averageCallsPerIndividual = maxCallsPerSequence * 0.6 / averageIndividualsPerMultiSequence;
averageCallsPerMultiSequence = averageCallsPerIndividual * averageIndividualsPerMultiSequence;

% Calculate total estimated calls needed
totalCallsNeeded = (numSingleIndividualSequences * averageCallsPerSingleSequence) + ...
                   (numMultiIndividualSequences * averageCallsPerMultiSequence);

% Add a 20% safety margin
numSamplesNeeded = ceil(totalCallsNeeded * 1.2);

% Round to the nearest hundred
numSamplesNeeded = 10^2 * ceil(numSamplesNeeded / 10^2);

end