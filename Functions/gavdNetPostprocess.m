function varargout = gavdNetPostprocess(audioIn, fileFs, probs, preprocParams, postprocParams)
% GAVDNETPOSTPROCESS Postprocess frame-based animal call detection probabilities

% This funciton converts call probability per frame to roi boundaries in 
% samples. 
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
%                       as a nonnegative scalar. 
% Outputs:
%   roi - Call regions, returned as an N-by-2 matrix of indices into the 
%           input signal, where N is the number of individual call regions 
%           detected. The first column contains the index of the start of 
%           a speech region, and the second column contains the index of 
%           the end of a region.
%   probs - Probability of speech per sample of the input audio signal, 
%           returned as a column vector with the same size as the input signal.
%
%   gavdNetPostprocess(...) with no output arguments displays a plot of the
%   detected vocalization regions in the input signal.
%
% References:
%   This function is a customised version of the MATLAB function
%   "vadnetPostprocess", which itself is a port of code from the open 
%   source code toolkit "SpeechBrain" [1]. 
%
%   [1] Ravanelli, Mirco, et al. SpeechBrain: A General-Purpose Speech Toolkit. 
%   arXiv, 8 June 2021. arXiv.org, http://arxiv.org/abs/2106.04624
% 
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

arguments
    audioIn (:,1) {validateattributes(audioIn,{'single','double'},{'nonempty','vector','real','finite'},'gavdNetPostprocess','audioIn')}
    fileFs (1,1) {validateattributes(fileFs,{'single','double'},{'positive','real','nonnan','finite'},'gavdNetPostprocess','fs')}
    probs (:,1) {validateattributes(probs,{'single','double'},{'nonempty','vector','real','nonnan','finite'},'gavdNetPostprocess','probs')}
    preprocParams (1,1) struct
    postprocParams (1,1) struct
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

% Time resolution of the spectrogram
timeResolution = hopDur;

% Validate that the audio duration is consistent with the probability vector
expNumHops = floor(ceil(numel(audioIn) * targetFs / fileFs) / hopLen) + 1;
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

% Convert frame-based roi to sample-based roi
boundaries = iframe2sample(b1, fileFs, timeResolution);

% Clip the final boundary to the number of original audio samples
boundaries = min(boundaries, numel(audioIn));

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

% Convenience plot if no output requested.
switch nargout
    case 0
        iconveniencePlot(audioIn, fileFs, targetFs, sampleroi, probs, hopLen);
    case 1
        varargout{1} = sampleroi;
    case 2
        varargout{1} = sampleroi;
        varargout{2} = iframeprob2sampleprob(probs, fileFs, targetFs, numel(audioIn), hopLen);
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
    energyBoundaries = iframe2sample(frameroi, fileFs, timeResolution);

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

function samples = iframe2sample(frame, fileFs, timeResolution)
%iframe2sample Convert frames to samples, extending the boundaries

if isempty(frame)
    samples = frame;
else
    xseconds = iframe2second(frame, timeResolution);
    samples = isecond2sample(xseconds, fileFs);
end
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

function iconveniencePlot(audioIn, fileFs, targetFs, boundaries, prob, hopLen)
%conveniencePlot Convenience plot for gavdNetPostprocess

% Create time vector
t = (0:length(audioIn)-1)/fileFs;

% Get axes handle and clear both left and right
axeshandle = newplot;
cla reset

% Plot the audio input against time
plot(axeshandle, t, audioIn)
grid on
hold on
ylabel(getString(message('audio:convenienceplots:Amplitude')))
xlabel(getString(message('audio:detectCalls:TimeAxis')))

% Add patches indicating the regions of detected vocalization
amax = max(audioIn);
amin = min(audioIn);
for idx = 1:size(boundaries, 1)
    xline(axeshandle, t(boundaries(idx,1)), Color=[0 0.4470 0.7410], LineWidth=1.2);
    patch(axeshandle, [t(boundaries(idx,1)), t(boundaries(idx,1)), t(boundaries(idx,2)), t(boundaries(idx,2))], ...
        [amin, amax, amax, amin], ...
        [0.3010 0.7450 0.9330], ...
        FaceAlpha=0.2, EdgeColor="none");
    xline(axeshandle, t(boundaries(idx,2)), Color=[0 0.4470 0.7410], LineWidth=1.2);
end
axis tight

% Plot the probability vector against time
yyaxis right
sampleprob = iframeprob2sampleprob(prob, fileFs, targetFs, numel(audioIn), hopLen);
plot(axeshandle, t, sampleprob, Color=[0.8500 0.3250 0.0980], LineStyle="--", LineWidth=1.2)
ylim([0,1])
ylabel(getString(message('audio:convenienceplots:Probability')))

% Add plot title
if isempty(boundaries)
    title(getString(message('audio:detectCalls:PlotTitleNoCalls')))
else
    title(getString(message('audio:detectCalls:PlotTitle')))
end
yyaxis left
hold off
end

function sampleprob = iframeprob2sampleprob(probs, fileFs, targetFs, N, hopLen)
% Hold the probability for the number of samples per hop
% Because the last frame is not overlapped in front, its decision
% gets held longer.

% Use the same relative proportions as the original gavdnetPostprocess
% In original VADnet: first=80 (0.5*hopLen), middle=160 (hopLen), last=280 (1.75*hopLen)
firstSamples = round(hopLen / 2);       % Half hop 
middleSamples = hopLen;                 % Full hop 
lastSamples = round(hopLen * 1.75);     % Hop + padding 

% Create sample-based probabilities
sampleprob_targetFs = cat(1, repelem(probs(1), firstSamples, 1), ...
                         repelem(probs(2:end-1), middleSamples, 1), ...
                         repelem(probs(end), lastSamples, 1));

% Resample the probability back to original sample rate
if fileFs ~= targetFs
    sampleprob_fileFs = iNearestNeighborResample(sampleprob_targetFs, fileFs, targetFs);
else
    sampleprob_fileFs = sampleprob_targetFs;
end

% Clip off any padding introduced
sampleprob_fileFs = sampleprob_fileFs(1:min(N, numel(sampleprob_fileFs)));

% If the vector is too short, pad with the last value
if numel(sampleprob_fileFs) < N
    sampleprob_fileFs = [sampleprob_fileFs; repelem(sampleprob_fileFs(end), N - numel(sampleprob_fileFs), 1)];
end

% Clamp the probability to range [0,1]
sampleprob = iclampProb(sampleprob_fileFs);
end

function y = iNearestNeighborResample(x, targetFs, originalFs)
% Get total number of samples after resampling
numOriginalSamples = numel(x);
numResampledSamples = round(numOriginalSamples*(targetFs/originalFs));

% Get resample indices
resampleIndices = round(linspace(1, numOriginalSamples, numResampledSamples));

% Resample
y = x(resampleIndices);
end