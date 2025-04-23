function [XTrainAll, TTrainAll] = concatenateTrainData(XTrain, TTrain)
%CONCATENATETRAINDATA Concatenates spectrogram chunks and target vector chunks
%   This function takes cell arrays XTrain and TTrain, which contain
%   spectrogram chunks and target vector chunks respectively, and
%   concatenates them into single cell arrays XTrainAll and TTrainAll.
%
%   Inputs:
%       XTrain - 1 x N cell array, each cell contains M x 1 cell array of chunks
%       TTrain - 1 x N cell array, each cell contains M x 1 cell array of chunks
%
%   Outputs:
%       XTrainAll - (N*M) x 1 cell array of spectrogram chunks
%       TTrainAll - (N*M) x 1 cell array of target vector chunks
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Check that XTrain and TTrain have the same length
if length(XTrain) ~= length(TTrain)
    fprintf('Warning: XTrain and TTrain have different lengths: %d vs %d\n', ...
        length(XTrain), length(TTrain));
end

% Get number of spectrograms
N = min(length(XTrain), length(TTrain));

% Print sizes to debug
fprintf('Number of spectrograms: %d\n', N);

% Initialize empty cell arrays
XTrainAll = {};
TTrainAll = {};

% Current index for filling output cell arrays
currentIdx = 1;

% Loop through each spectrogram and its corresponding target vector
for i = 1:N
    % Check if the current cells exist
    if i > length(XTrain) || i > length(TTrain)
        fprintf('Skipping index %d as it exceeds either XTrain or TTrain length\n', i);
        continue;
    end
    
    % Check if the current cells are empty
    if isempty(XTrain{i})
        fprintf('XTrain{%d} is empty, skipping\n', i);
        continue;
    end
    
    if isempty(TTrain{i})
        fprintf('TTrain{%d} is empty, skipping\n', i);
        continue;
    end
    
    % Get number of chunks for this spectrogram
    MX = length(XTrain{i});
    MT = length(TTrain{i});
    
    % Check if lengths match
    if MX ~= MT
        fprintf('Warning: XTrain{%d} and TTrain{%d} have different lengths: %d vs %d\n', ...
            i, i, MX, MT);
    end
    
    % Use the minimum length to avoid index errors
    M = min(MX, MT);
    
    if M == 0
        fprintf('Both XTrain{%d} and TTrain{%d} have zero elements, skipping\n', i, i);
        continue;
    end
    
    % Loop through each chunk
    for j = 1:M
        try
            % Check if indices are valid
            if j > length(XTrain{i})
                fprintf('j=%d exceeds XTrain{%d} length of %d, skipping\n', ...
                    j, i, length(XTrain{i}));
                continue;
            end
            
            if j > length(TTrain{i})
                fprintf('j=%d exceeds TTrain{%d} length of %d, skipping\n', ...
                    j, i, length(TTrain{i}));
                continue;
            end
            
            % Add spectrogram chunk to XTrainAll
            XTrainAll{currentIdx, 1} = XTrain{i}{j};
            
            % Add target vector chunk to TTrainAll
            TTrainAll{currentIdx, 1} = TTrain{i}{j};
            
            % Increment index
            currentIdx = currentIdx + 1;
        catch ME
            fprintf('Error at i=%d, j=%d, currentIdx=%d: %s\n', ...
                i, j, currentIdx, ME.message);
            % Continue to the next iteration rather than stopping
        end
    end
end

% Trim any unused cells if we had skips
XTrainAll = XTrainAll(1:currentIdx-1);
TTrainAll = TTrainAll(1:currentIdx-1);

fprintf('Successfully created XTrainAll with %d elements and TTrainAll with %d elements\n', ...
    length(XTrainAll), length(TTrainAll));
end