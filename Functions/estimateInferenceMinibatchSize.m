function optimalMinibatchSize = estimateInferenceMinibatchSize(availableMemoryBytes, T)
% This function estimates the largest minibatch size for the VAD network
% given the temporal length T of the input spectrogram.
%
% The minibatch size represents the number of 40×4 temporal frames that are
% processed simultaneously through the network. A 40×T spectrogram creates
% T/4 frames total, which are processed in batches of 'minibatch size' frames.
%
% Inputs:
%   availableMemoryBytes - memory available (on either GPU or system,
%                           whichever is in use)   
%   T - integer, temporal dimension of input spectrogram (40×T)
%
% Outputs:
%   optimalMinibatchSize - integer, maximum number of 40×4 frames that can
%                         be processed simultaneously in each mini-batch
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

if ~isnumeric(T) || ~isscalar(T) || T < 1
    error('T must be a positive integer.');
end

% Calculate memory requirements per frame in a mini-batch
memoryPerFrameBytes = estimateMemoryPerFrame();

% Calculate theoretical maximum mini-batch size
theoreticalMaxBatch = floor(availableMemoryBytes / memoryPerFrameBytes);

% Apply safety margin (use 70% of available memory)
safetyFactor = 0.7;
safeMaxBatch = floor(theoreticalMaxBatch * safetyFactor);

% Set reasonable bounds for mini-batch size
minBatchSize = 1;
maxBatchSize = 2048;  % Reasonable upper limit for frame batches

optimalMinibatchSize = max(minBatchSize, min(safeMaxBatch, maxBatchSize));
end

function memoryPerFrameBytes = estimateMemoryPerFrame()
% Estimate memory requirements per 40×4 frame in a mini-batch
% This includes CNN activations, GRU states, and FC outputs for one frame

bytesPerElement = 4;  % single precision

% 1. CNN layer activations per frame
% Input: 40×4×1
inputMemory = 40 * 4 * 1 * bytesPerElement;

% CNN1 layers
cnn1Conv1 = 40 * 4 * 16 * bytesPerElement;   % 16 filters
cnn1Pool1 = 40 * 2 * 16 * bytesPerElement;   % 1×2 pooling

% CNN2 layers
cnn2Conv1 = 40 * 2 * 32 * bytesPerElement;   % 32 filters
cnn2Pool2 = 40 * 1 * 32 * bytesPerElement;   % Final pooling

% Flattened features: 32 per frame
flattenedMemory = 32 * bytesPerElement;

cnnMemory = inputMemory + cnn1Conv1 + cnn1Pool1 + cnn2Conv1 + cnn2Pool2 + flattenedMemory;

% 2. GRU layer memory per frame
% Each frame requires hidden states and outputs
% Bidirectional GRU: forward (32) + reverse (32) + output (64) per layer
gruLayer1Memory = (32 + 32 + 64) * bytesPerElement;
gruLayer2Memory = (32 + 32 + 64) * bytesPerElement;
gruMemory = gruLayer1Memory + gruLayer2Memory;

% 3. Fully connected layer outputs per frame
fc1Memory = 16 * bytesPerElement;
fc2Memory = 16 * bytesPerElement;
fc3Memory = 1 * bytesPerElement;
fcMemory = fc1Memory + fc2Memory + fc3Memory;

% 4. Intermediate activations and workspace
intermediateMemory = cnnMemory * 0.2;  % Estimate for temporary buffers

% Total activation memory per frame
activationMemory = cnnMemory + gruMemory + fcMemory + intermediateMemory;

% 5. Amortized parameter memory
parameterMemory = estimateParameterMemory();
parameterMemoryPerFrame = parameterMemory / 1024;  % Amortize over reasonable number

% Total memory per frame
memoryPerFrameBytes = activationMemory + parameterMemoryPerFrame;

% Add overhead factor for CUDA operations and memory management
overheadFactor = 1.8;
memoryPerFrameBytes = memoryPerFrameBytes * overheadFactor;
end

function paramMemoryBytes = estimateParameterMemory()
% Estimate total parameter memory for the VAD network

bytesPerElement = 4;  % single precision

% CNN parameters
cnn1Conv1Params = (3 * 3 * 1 * 16) + 16;    % 144 + 16 = 160
cnn1Conv2Params = (3 * 3 * 16 * 16) + 16;   % 2304 + 16 = 2320
cnn2Conv1Params = (3 * 3 * 16 * 32) + 32;   % 4608 + 32 = 4640
cnn2Conv2Params = (3 * 3 * 32 * 32) + 32;   % 9216 + 32 = 9248

% GRU parameters (each GRU has 3 gates: reset, update, new)
% Input size = 32, hidden size = 32
% Parameters per gate: input_weights (32×32) + hidden_weights (32×32) + bias (32)
% Total per GRU: 3 * (32*32 + 32*32 + 32) = 3 * (1024 + 1024 + 32) = 6240
% Bidirectional: 2 * 6240 = 12480 per layer
gruParams = 2 * 12480;  % 2 GRU layers = 24960

% Fully connected parameters
fc1Params = (32 * 16) + 16;  % 512 + 16 = 528
fc2Params = (16 * 16) + 16;  % 256 + 16 = 272
fc3Params = (16 * 1) + 1;    % 16 + 1 = 17

% Normalization parameters (scale + offset for each channel)
% Layer norm for input (1), CNN layers (16+16+32+32), FC layers (16+16)
normParams = (1 + 16 + 16 + 32 + 32 + 16 + 16) * 2;  % 258

totalParams = cnn1Conv1Params + cnn1Conv2Params + cnn2Conv1Params + ...
    cnn2Conv2Params + gruParams + fc1Params + fc2Params + ...
    fc3Params + normParams;

paramMemoryBytes = totalParams * bytesPerElement;
end