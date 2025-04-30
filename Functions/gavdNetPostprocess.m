function varargout = gavdNetPostprocess(audioIn, fileFs, probs, preprocParams, postprocParams, features)
% This function converts the vector of detection probabilities (per 
% spectrpgram frame) to region of interest (roi) boundaries in samples, and
% applies heuristics to filter out detections that are of unrealisticlly 
% short duration and merge detections that are likely not discrete. It also
% sets the upper and lower probability thresholds for detections, and
% features an optional energy-based detector to refine the detection
% predictions made by the neural network.
% 
% Inputs: 
%   audioIn - the audio the model is operating on
%   fileFs - original sample rate of audioIn, prior to preprocessing (Hz)
%   probs - the raw output from the model when calling predict() or
%           minibatchpredict() on the features extracted from audioIn.
%   preprocParams - a struct containing preprocessing parameters used at 
%                   training. embedded in the trained model's metadata.
%   postprocPrams - a struct containing post-processing parameters i the
%                   following fields:
%                       - 'AT' (Activation threshold): sets the threshold
%                       for starting a vocalization segment. Specify as a 
%                       scalar in the range [0,1]. 
%                       - 'DT' (Deactivation threshold): sets the threshold
%                       for ending a vocalization segment. Specify as a 
%                       scalar in the range [0,1]. 
%                       - 'AT' (ApplyEnergyAVD): specifies whether to apply 
%                       an energy-based vocalization activity detector to 
%                       refine the neural network's detections. 
%                       - 'MT' (Merge threshold): merges vocalization regions
%                       that are separated by MT seconds or less. Specify 
%                       as a nonnegative scalar. 
%                       - 'LT' (Length threshold): removes vocalization regions
%                       that have a duration of LT seconds or less. Specify 
%                       as a nonnegative scalar. It is recommended to use 
%                       the "minimum call duration" parameter used for 
%                       training data synthesis. This value is stored in 
%                       the trained model's metadata. You can try using 
%                       smaller values if parts of your target call are 
%                       frequently missing due to propagation effects or 
%                       call variations that are not strongly represented 
%                       in the synthetic training data.
%
% Outputs:
%   roi - Call regions, returned as an N-by-2 matrix of indices into the 
%           input signal, where N is the number of individual call regions 
%           detected. The first column contains the index of the start of 
%           a speech region, and the second column contains the index of 
%           the end of a region.
%   probs - Probability of speech per sample of the input audio signal, 
%           returned as a column vector with the same size as the input
%           signal. (optional)
%   confidence - Confidence scores for each detected region, returned as an
%           N-by-1 vector, where N is the number of individual call regions.
%           Each value represents the mean probability within the region. (optional)
%   fig - a figure showing the input audio waveform and probabilities 
%           vector in the top tile, and the spectrogram and event
%           boundaries in the bottom tile. (optional)
%
%   gavdNetPostprocess(...) with no output arguments displays a plot of the
%   detected vocalization regions in the input signal.
%
% References:
%   This function is based on the MATLAB function "vadnetPostprocess" [1, 2]
%   That function is a port of code from the open source toolkit 
%   "SpeechBrain" [3]. 
%
%   [1] The MathWorks Inc. (2022-2024). Audio Toolbox version: 24.2 (R2024b), 
%   Natick, Massachusetts: The MathWorks Inc. https://www.mathworks.com
%
%   [2] The MathWorks Inc. (2022-2024). Deep Learning Toolbox version: 24.2 (R2024b), 
%   Natick, Massachusetts: The MathWorks Inc. https://www.mathworks.com
%
%   [3] Ravanelli, Mirco, et al. SpeechBrain: A General-Purpose Speech Toolkit. 
%   arXiv, 8 June 2021. arXiv.org, http://arxiv.org/abs/2106.04624
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%% Input validation & unpacking

arguments
    audioIn (:,1) {validateattributes(audioIn,{'single','double'},{'nonempty','vector','real','finite'},'gavdNetPostprocess','audioIn')}
    fileFs (1,1) {validateattributes(fileFs,{'single','double'},{'positive','real','nonnan','finite'},'gavdNetPostprocess','fs')}
    probs (:,1) {validateattributes(probs,{'single','double'},{'nonempty','vector','real','nonnan','finite'},'gavdNetPostprocess','probs')}
    preprocParams (1,1) struct
    postprocParams (1,1) struct
    features = []
end

if any(~isfield(preprocParams, {'fsTarget', 'hopDur', 'hopLen', 'windowLen', 'bandwidth'}))
    error('Input argument "preprocParams" does not contain the correct fields.')
end
if any(~isfield(postprocParams, {'AT', 'DT', 'AEAVD', 'MT', 'LT'}))
    error('Input argument "preprocParams" does not contain the correct fields.')
end

% Parameters should not be on the GPU
if isempty(coder.target)
    postprocParams = structfun(@gather, postprocParams, UniformOutput=false);
    [fileFs, probs] = gather(fileFs, probs);
end

% Unpack post-processing params 
activationThreshold = postprocParams.AT ;
deactivationThreshold = postprocParams.DT;
AEAVD = postprocParams.AEAVD;
mergeThreshold = postprocParams.MT;
lengthThreshold = postprocParams.LT;

% Unpack preprocessor parameters (used in both training & inference)
targetFs = preprocParams.fsTarget; % The new rate for the audio, resampled before stft (Hz)
hopDur = preprocParams.hopDur; % Duration of the STFT window hop (seconds)
hopLen = preprocParams.hopLen; % Length of the STFT window hop (samples)
windowLen = preprocParams.windowLen; % Length of the window function used to compute the STFT (samples)
bandwidth = preprocParams.bandwidth; % Bandwidth of the mel spectrogram.

% Time resolution of the spectrogram
timeResolution = hopDur;

% Validate that the audio duration is consistent with the probability vector
% Account for padding (half window length at each end) in the preprocessing function
padLen = round(windowLen/2);
resampledAudioLength = ceil(numel(audioIn) * targetFs / fileFs);
paddedLength = resampledAudioLength + (2 * padLen);
expNumHops = floor((paddedLength - windowLen) / hopLen) + 1;
if expNumHops ~= numel(probs)
    warning('Length of "audioIn" is %g samples with sample rate = %g Hz.\n', numel(audioIn), fileFs)
    fprintf('Preprocessor is resampling to %g Hz.\n', targetFs)
    fprintf('Preprocessor STFT hop is %g audio samples. \n', hopLen)
    fprintf('Expecting "probs" to be %g frames long\n', expNumHops)
    fprintf('Length of "probs" is %g frames.\n', numel(probs))
    error('Mismatched audio and probs lengths');
end

% Error if the deactivation threshold is greater than the activation threshold.
assert(deactivationThreshold < activationThreshold, ...
    'Deactivation threshold must be < activationThreshold');

% Convert thresholds in seconds to samples.
mergeThreshold = isecond2sample(mergeThreshold, targetFs);
lengthThreshold = isecond2sample(lengthThreshold, targetFs);

% Create mask by applying activation and deactivation thresholds
avdmask = iapplyThreshold(probs(:), activationThreshold, deactivationThreshold);

% Convert frame-based mask to frame-based roi
b1 = binmask2sigroi(avdmask);

% Use corrected time-to-sample conversion that accounts for preprocessing padding
boundaries = iframe2sample_corrected(b1, fileFs, targetFs, numel(audioIn), windowLen, hopLen, padLen);

% Apply energy-based detection refinement if requested
if AEAVD
    boundaries = ienergyAVD(audioIn, fileFs, timeResolution, boundaries, numel(probs(:)), hopLen);
    boundaries = min(boundaries, numel(audioIn));
end

% Apply merge and length thresholds to sample-based boundaries
if ~isempty(boundaries)
    if mergeThreshold~=inf
        b2 = mergesigroi(boundaries, mergeThreshold);
    else
        b2 = boundaries;
    end
    if lengthThreshold~=0
        b3 = removesigroi(b2, lengthThreshold);
    else
        b3 = b2;
    end
else
    b3 = b1;
end
sampleroi = b3;

% Convert probabilities to sample domain for confidence calculation
sampleprob = iframeprob2sampleprob(probs, fileFs, targetFs, numel(audioIn), windowLen, hopLen, padLen);

% Calculate confidence for each region (mean probability within the region)
confidence = [];
if ~isempty(sampleroi)
    confidence = zeros(size(sampleroi, 1), 1);
    for i = 1:size(sampleroi, 1)
        regionStart = sampleroi(i, 1);
        regionEnd = min(sampleroi(i, 2), numel(sampleprob));
        confidence(i) = mean(sampleprob(regionStart:regionEnd));
    end
end

% Convenience plot if no output requested.
switch nargout
    case 0
        iconveniencePlot(audioIn, fileFs, targetFs, sampleroi, probs, windowLen, hopLen, bandwidth, features);
    case 1
        varargout{1} = sampleroi;
    case 2
        varargout{1} = sampleroi;
        varargout{2} = sampleprob;
    case 3
        varargout{1} = sampleroi;
        varargout{2} = sampleprob;
        varargout{3} = confidence;
    case 4
        varargout{1} = sampleroi;
        varargout{2} = sampleprob;
        varargout{3} = confidence;
        iconveniencePlot(audioIn, fileFs, targetFs, sampleroi, probs, windowLen, hopLen, bandwidth, features);
end
end

function out = ienergyAVD(audioIn, fileFs, timeResolution, boundaries, numHops, hopLen)
%ienergyAVD Apply energy-based detection to fine-tune boundaries

% Compute analysis chunk length
padLength = round(hopLen/2);

% To support code generation, preallocate the largest output possible. The
% energy detection analyzes in timeResolution chunks with timeResolution hops. 
% The hop size is the same as the hop size of the probabilities output from the model. 
% It is not possible for two consecutive active regions to have separate boundaries.
% The max number of boundaries possible is numHops/2, where numHops is the
% number of hops output from gavdNetPreprocess (and therefore the model).
newBoundaries = zeros(ceil(numHops/2), 2, like=boundaries);

% Convert thresholds in seconds to frames.
mergeThreshold = isecond2frame(0.1, fileFs, timeResolution);
lengthThreshold = isecond2frame(0.1, fileFs, timeResolution);

% Process detected vocalization segments
idx = 1;
for ii = 1:size(boundaries, 1)
    % Isolate vocalization segment
    x = audioIn(boundaries(ii,1):boundaries(ii,2));

    % Pad the segment the same as gavdNetPreprocess
    xp = [zeros(padLength, 1, like=x); x(:); zeros(padLength, 1, like=x)];

    % Buffer segment into analysis chunks
    xb = audio.internal.buffer(xp, hopLen, hopLen);

    % Compute energy per chunk
    xbe = log(sum(abs(xb), 1) + eps(underlyingType(xb)));

    % Normalize energy
    xbe = (xbe - mean(xbe))/ (2*std(xbe)) + 0.5;

    % Apply threshold
    avdmask = iapplyThreshold(xbe(:), 0.5, 0);

    % Convert mask to roi
    b1 = binmask2sigroi(avdmask);
    b2 = mergesigroi(b1, mergeThreshold);
    b3 = removesigroi(b2, lengthThreshold);
    frameroi = b3;

    % Convert frame-based roi to sample-based roi
    energyBoundaries = iframe2sample_corrected(frameroi, fileFs, fileFs, numel(x), hopLen, hopLen, padLength);

    % Get the final boundaries in the original signal
    offset = boundaries(ii,1);
    N = size(energyBoundaries, 1);
    newBoundaries(idx:idx+N-1,:) = offset + [energyBoundaries(:,1), energyBoundaries(:,2)] - 1;

    idx = idx+N;
end

% Remove the extra rows preallocated from the new boundaries.
out = newBoundaries(1:idx-1,:);
end

function out = isecond2frame(x, fileFs, timeResolution)
%isecond2frame Convert seconds to frames

% This utility assumes the seconds are evenly spaced (not extended at the
% boundaries).
out = round(x*fileFs*timeResolution);
end

function xseconds = iframe2second(frame, timeResolution)
%iframe2second Convert frames to seconds

% Because the initial signal is front-padded in preprocessing, the first
% frame is at time 0.
xseconds = (frame-1)*timeResolution;

% xseconds represents the center of the windows. To not drop beginning and
% end samples, extend front and back to the midpoints between windows.
xseconds(:,1) = max(xseconds(:,1) - timeResolution/2, 0);
xseconds(:,2) = xseconds(:,2) + timeResolution/2;
end

function samples = isecond2sample(xseconds, fileFs)
%isecond2sample Convert seconds to samples
samples = max(floor(xseconds*fileFs), 1);
end

function samples = iframe2sample_corrected(frame, fileFs, targetFs, audioLength, windowLen, hopLen, padLen)
%iframe2sample_corrected Convert frames to samples, accounting for preprocessing
%
% This corrected version properly accounts for:
% 1. The padding added during preprocessing
% 2. The difference between original and target sample rates
% 3. The window and hop length used in the STFT

if isempty(frame)
    samples = frame;
    return;
end

% Convert frame indices to time positions (seconds) in the padded, resampled signal
timeResolution = hopLen / targetFs;
frameTimesInPaddedSignal = iframe2second(frame, timeResolution);

% Adjust for padding (half window at beginning)
paddingTimeOffset = padLen / targetFs;
frameTimesInResampledSignal = frameTimesInPaddedSignal - paddingTimeOffset;

% Ensure no negative times after padding adjustment
frameTimesInResampledSignal(:,1) = max(frameTimesInResampledSignal(:,1), 0);
frameTimesInResampledSignal(:,2) = max(frameTimesInResampledSignal(:,2), 0);

% Convert times to sample indices in the original signal
samples = round(frameTimesInResampledSignal * fileFs) + 1;

% Ensure boundaries are within the original audio
samples(:,1) = max(1, min(samples(:,1), audioLength));
samples(:,2) = max(1, min(samples(:,2), audioLength));
end

function avdmask = iapplyThreshold(x, activationThreshold, deactivationThreshold)
%iapplyThreshold Apply activation and deactivation thresholds to create
%mask

activation = (x >= activationThreshold);
deactivation = (x >= deactivationThreshold);
avdmask = activation + deactivation;

% AVDmask==2, active region
% AVDmask==1, between active and deactive thresholds, hold region as
%             active until deactive threshold is crossed.
% AVDmask==0, non-active region

avdmaskIsOne = (avdmask==1);
for ii = 2:numel(x)
    if (avdmask(ii-1)==2) && avdmaskIsOne(ii)
        avdmask(ii) = 2;
    end
end
avdmask = max(avdmask - 1, 0);

end

function prob = iclampProb(prob)
% Because the model is a regression network, the output can be greater than 1
% or less than 0. This is an edge case. Center clip the output to the range
% [0,1] to interpret as probability.
prob = max(min(prob, 1), 0);
end

function iconveniencePlot(audioIn, fileFs, targetFs, boundaries, prob, windowLen, hopLen, bandwidth, features)
%conveniencePlot Convenience plot for gavdNetPostprocess

% Create time vector
t_waveform = (0:length(audioIn)-1)/fileFs;

% Create a tiled layout
fig = figure;
tl = tiledlayout(2, 1);

% Create first tile for waveform
ax1 = nexttile;

% Plot the audio input against time
plot(ax1, t_waveform, audioIn)
grid on
hold on
ylabel('Amplitude')
xlabel('Time (s)')

% Plot the probability vector against time
yyaxis right
padLen = round(windowLen/2);
sampleprob = iframeprob2sampleprob(prob, fileFs, targetFs, numel(audioIn), windowLen, hopLen, padLen);
plot(ax1, t_waveform, sampleprob, Color=[0.8500 0.3250 0.0980], LineStyle="--", LineWidth=1.2)
ylim([0,1])
ylabel('Probability of target sound detection')
title('Waveform & Probabilities')
yyaxis left
hold off

% Add the spectrogram to tile 2
ax2 = nexttile;

% If the spectrogram is empty, recalculate it using the preprocessor
if isempty(features)
    [features, ~] = gavdNetPreprocess(audioIn, fileFs, targetFs, bandwidth, windowLen, hopLen);
end

% Calculate the time and frequency vectors
t_spectrogram = linspace(t_waveform(1), t_waveform(end), size(features, 2));
lowMel = 1127.01048 * log(1 + bandwidth(1)/700);
highMel = 1127.01048 * log(1 + bandwidth(2)/700);
melPoints = linspace(lowMel, highMel, 40 + 2);
f = 700 * (exp(melPoints(2:end-1)/1127.01048) - 1);

% Manually plot the spectrogram with correct time units
imagesc(ax2, t_spectrogram, f, features)
axis tight

% Set the y-axis limits according to bandwidth
ylim(ax2, bandwidth)
ylabel('Frequency (Hz)')
xlabel('Time (s)')
c = colorbar('Location', 'eastoutside');
ylabel(c, 'Power (dB)')
set(gca, "YDir", "normal")
title('Spectrogram & Final Detection Boundaries')
grid on

% Add patches to the spectrogram, showing event boundaries
for idx = 1:size(boundaries, 1)
    % Calculate time values for boundaries
    t_start = t_waveform(boundaries(idx,1));
    t_end = t_waveform(boundaries(idx,2));

    % Draw rectangles with red dotted outline and no fill
    rectangle('Position', [t_start, bandwidth(1), t_end-t_start, bandwidth(2)-bandwidth(1)], ...
        'EdgeColor', 'r', 'LineStyle', ':', 'LineWidth', 1.5);
end

% Make sure both plots have the same x range
xlim(ax2, [min(t_waveform), max(t_waveform)])

% Link the time axes of both plots
linkaxes([ax1, ax2], 'x');

% Add plot title
if isempty(boundaries)
    sgtitle('No Calls Detected')
else
    sgtitle(sprintf('%g Detected Calls', size(boundaries, 1)))
end
end

function sampleprob = iframeprob2sampleprob(probs, fileFs, targetFs, N, windowLen, hopLen, padLen)
% Convert frame probabilities to sample-based probabilities
% This corrected version accounts for:
% 1. The padding in the preprocessor (padLen)
% 2. The window and hop length
% 3. The different sample rates

% Calculate key parameters
frameDuration = hopLen / targetFs;  % Duration of each frame in seconds

% Create array to hold per-sample probabilities at target sample rate
paddedLength = ceil(N * targetFs / fileFs) + (2 * padLen);
sampleprob_targetFs = zeros(paddedLength, 1);

% For each probability value, fill in the corresponding samples
for i = 1:length(probs)
    % Calculate the start and end sample indices for this frame
    startSample = padLen + 1 + (i-1) * hopLen;
    endSample = min(paddedLength, startSample + hopLen - 1);
    
    % Fill in the probability for these samples
    sampleprob_targetFs(startSample:endSample) = probs(i);
end

% Remove the padding
sampleprob_targetFs = sampleprob_targetFs(padLen+1:end-padLen);

% Resample to the original file sample rate
if fileFs ~= targetFs
    % Need a cleaner resampling method than nearest neighbor for accurate boundaries
    sampleprob_fileFs = resample(double(sampleprob_targetFs), fileFs, targetFs);
else
    sampleprob_fileFs = sampleprob_targetFs;
end

% Adjust length to match the original audio
if length(sampleprob_fileFs) > N
    sampleprob_fileFs = sampleprob_fileFs(1:N);
elseif length(sampleprob_fileFs) < N
    % Pad with the last probability if needed
    sampleprob_fileFs = [sampleprob_fileFs; repelem(sampleprob_fileFs(end), N-length(sampleprob_fileFs), 1)];
end

% Clamp probabilities to [0,1] range
sampleprob = iclampProb(sampleprob_fileFs);
end