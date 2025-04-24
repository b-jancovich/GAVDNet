function varargout = gavdNetPostprocess(audioIn, audioFs, probs, hopDur, options)
%gavdNetPostprocess Postprocess frame-based animal vocalisation probabilities

%   roi = vadnetPostprocess(audioIn,fs,probs) converts voice activity
%   probability per frame to roi boundaries in samples. Specify audioIn and
%   fs as the same values input to vadnetPreprocess. Specify probs as the
%   value output from calling predict on vadnet.
%
%   roi = vadnetPostprocess(...,ActivationThreshold=AT) sets the threshold
%   for starting a target sound segment. Specify ActivationThreshold as a scalar
%   in the range [0,1]. If unspecified, ActivationThreshold defaults to 0.5.
%
%   roi = vadnetPostprocess(...,DeactivationThreshold=DT) sets the
%   threshold for ending a target sound segment. Specify DeactivationThreshold as
%   a scalar in the range [0,1]. If unspecified, DeactivationThreshold
%   defaults to 0.25.
%
%   roi = vadnetPostprocess(...,ApplyEnergyVAD=AEVAD) specifies whether to
%   apply an energy-based voice activity detector to the target sound regions
%   detected by the neural network. If unspecified, ApplyEnergyVAD defaults
%   to false.
%
%   roi = vadnetPostprocess(...,MergeThreshold=MT) merges target sound regions
%   that are separated by MT seconds or less. Specify MergeThreshold as a
%   nonnegative scalar. If unspecified, MergeThreshold defaults to 0.25
%   seconds.
%
%   roi = vadnetPostprocess(...,LengthThreshold=LT) removes target sound regions
%   that have a duration of LT seconds or less. Specify LengthThreshold as
%   a nonnegative scalar. If unspecified, LengthThreshold defaults to 0.25
%   seconds.
%
%   vadnetPostprocess(...) with no output arguments displays a plot of the
%   detected target sound regions in the input signal.
%
arguments
    audioIn (:,1) {validateattributes(audioIn,{'single','double'},{'nonempty','vector','real','finite'},'vadnetPostprocess','audioIn')}
    audioFs (1,1) {validateattributes(audioFs,{'single','double'},{'positive','real','nonnan','finite'},'vadnetPostprocess','fs')}
    probs (:,1) {validateattributes(probs,{'single','double'},{'nonempty','vector','real','nonnan','finite'},'vadnetPostprocess','probs')}
    hopDur
    options.ActivationThreshold (1,1) {validateattributes(options.ActivationThreshold,{'single','double'},{'scalar','<=',1,'>=',0},'vadnetPostprocess','ActivationThreshold')} = 0.5;
    options.DeactivationThreshold (1,1) {validateattributes(options.DeactivationThreshold,{'single','double'},{'scalar','<=',1,'>=',0},'vadnetPostprocess','DeactivationThreshold')} = 0.25;
    options.ApplyEnergyVAD (1,1) {mustBeNumericOrLogical} = false;
    options.MergeThreshold (1,1) {validateattributes(options.MergeThreshold,{'numeric'},{'scalar','>=',0},'vadnetPostprocess','MergeThreshold')} = 0.25;
    options.LengthThreshold (1,1) {validateattributes(options.LengthThreshold,{'numeric'},{'scalar','>=',0},'vadnetPostprocess','LengthThreshold')} = 0.25;
end

% Parameters should not be on the GPU
if isempty(coder.target)
    options = structfun(@gather,options,UniformOutput=false);
    [audioFs,probs] = gather(audioFs,probs);
end

% % Time resolution (hop length) of vadnetPreprocess
% hopDur = 0.01;

% % Sample rate of intermediate representation input to network.
% audioFs = 16e3;

% Validate that the audio duration is consistent with the probability
% vector
expNumHops = floor(ceil(numel(audioIn)*audioFs/audioFs)/160) + 1;
coder.internal.errorIf(expNumHops~=numel(probs), ...
    'audio:vadnet:InconsistentLengths', ...
    numel(audioIn),numel(probs),'vadnetPreprocess')

% Error if the deactivation threshold is greater than the activation
% threshold.
coder.internal.errorIf(options.DeactivationThreshold>=options.ActivationThreshold, ...
    'audio:vadnet:InvalidActivationPair')

% Convert thresholds in seconds to samples.
options.MergeThreshold = isecond2sample(options.MergeThreshold,audioFs);
options.LengthThreshold = isecond2sample(options.LengthThreshold,audioFs);

% Create mask by applying activation and deactivation thresholds
vadmask = iapplyThreshold(probs(:),options.ActivationThreshold,options.DeactivationThreshold);

% Convert frame-based mask to frame-based roi
b1 = binmask2sigroi(vadmask);

% Convert frame-based roi to sample-based roi
boundaries = iframe2sample(b1,audioFs,hopDur);

% Clip the final boundary to the number of original audio samples
boundaries = min(boundaries,numel(audioIn));

% Apply energy-based VAD if requested
if options.ApplyEnergyVAD
    boundaries = ienergyVAD(audioIn,audioFs,hopDur,boundaries,numel(probs(:)));
    boundaries = min(boundaries,numel(audioIn));
end

% Apply merge and length thresholds to sample-based boundaries
if ~isempty(boundaries)
    if options.MergeThreshold~=inf
        b2 = mergesigroi(boundaries,options.MergeThreshold);
    else
        b2 = boundaries;
    end
    if options.LengthThreshold~=0
        b3 = removesigroi(b2,options.LengthThreshold);
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
        iconveniencePlot(audioIn,audioFs,audioFs,sampleroi,probs)
    case 1
        varargout{1} = sampleroi;
    case 2
        varargout{1} = sampleroi;
        varargout{2} = iframeprob2sampleprob(probs,audioFs,audioFs,numel(audioIn));
end

end

function out = ienergyVAD(audioIn,fs,timeResolution,boundaries,numHops)
%ienergyVAD Apply energy VAD to fine-tune boundaries

% Compute analysis chunk length
chunkLength = round(timeResolution*fs);
padLength = round(chunkLength/2);

% To support code generation, preallocate the largest output possible. The
% energy VAD analyzes in 10 ms chunks with 10 ms hops. The hop size is the
% same as the hop size of the probabilities output from vadnet. It is not
% possible for two consecutive active regions to have separate boundaries.
% The max number of boundaries possible is numHops/2, where numHops is the
% number of hops output from vadnetPreprocess (and therefore vadnet).
newBoundaries = zeros(ceil(numHops/2),2,like=boundaries);

% Convert thresholds in seconds to frames.
mergeThreshold = isecond2frame(0.1,fs,timeResolution);
lengthThreshold = isecond2frame(0.1,fs,timeResolution);

% Process detected target sound segments
idx = 1;
for ii = 1:size(boundaries,1)

    % Isolate target sound segment
    x = audioIn(boundaries(ii,1):boundaries(ii,2));

    % Pad the segment the same as vadnetPreprocess
    xp = [zeros(padLength,1,like=x);x(:);zeros(padLength,1,like=x)];

    % Buffer target sound segment into analysis chunks
    xb = audio.internal.buffer(xp,chunkLength,chunkLength);

    % Compute energy per chunk
    xbe = log(sum(abs(xb),1) + eps(underlyingType(xb)));

    % Normalize energy
    xbe = (xbe - mean(xbe))/ (2*std(xbe)) + 0.5;

    % Apply threshold
    vadmask = iapplyThreshold(xbe(:),0.5,0);

    % Convert mask to roi
    b1 = binmask2sigroi(vadmask);
    b2 = mergesigroi(b1,mergeThreshold);
    b3 = removesigroi(b2,lengthThreshold);
    frameroi = b3;

    % Convert frame-based roi to sample-based roi
    energyBoundaries = iframe2sample(frameroi,fs,timeResolution);

    % Get the final boundaries in the original signal
    offset = boundaries(ii,1);
    N = size(energyBoundaries,1);
    newBoundaries(idx:idx+N-1,:) = offset + [energyBoundaries(:,1),energyBoundaries(:,2)] - 1;

    idx = idx+N;

end
% Remove the extra rows preallocated from the new boundaries.
out = newBoundaries(1:idx-1,:);
end

function out = isecond2frame(x,expfs,timeResolution)
%isecond2frame Convert seconds to frames

% This utility assumes the seconds are evenly spaced (not extended at the
% boundaries).

out = round(x*expfs*timeResolution);
end

function xseconds = iframe2second(frame,timeResolution)
%iframe2second Convert frames to seconds

% Because the initial signal is front-padded in preprocessing, the first
% frame is at time 0.
xseconds = (frame-1)*timeResolution;

% xseconds represents the center of the windows. To not drop beginning and
% end samples, extend front and back to the midpoints between windows.
xseconds(:,1) = max(xseconds(:,1) - timeResolution/2,0);
xseconds(:,2) = xseconds(:,2) + timeResolution/2;
end

function samples = isecond2sample(xseconds,fs)
%isecond2sample Convert seconds to samples
samples = max(floor(xseconds*fs),1);
end

function samples = iframe2sample(frame,fs,timeResolution)
%iframe2sample Convert frames to samples, extending the boundaries

if isempty(frame)
    samples = frame;
else
    xseconds = iframe2second(frame,timeResolution);
    samples = isecond2sample(xseconds,fs);
end
end

function vadmask = iapplyThreshold(x,activationThreshold,deactivationThreshold)
%iapplyThreshold Apply activation and deactivation thresholds to create
%mask

activation = (x >= activationThreshold);
deactivation = (x >= deactivationThreshold);
vadmask = activation + deactivation;

% vadmask==2, active target sound region
% vadmask==1, between active and deactive thresholds, hold target sound region as
%             active until deactive threshold is crossed.
% vadmask==0, non target sound region

vadmaskIsOne = (vadmask==1);
for ii = 2:numel(x)
    if (vadmask(ii-1)==2) && vadmaskIsOne(ii)
        vadmask(ii) = 2;
    end
end
vadmask = max(vadmask - 1,0);

end

function prob = iclampProb(prob)
% Because vadnet is a regression network, the output can be greater than 1
% or less than 0. This is an edge case. Center clip the output to the range
% [0,1] to interpret as probability.
prob = max(min(prob,1),0);
end

function iconveniencePlot(audioIn,fs,expfs,boundaries,prob)
%conveniencePlot Convenience plot for vadnetPostprocess

% Create time vector
t = (0:length(audioIn)-1)/fs;

% Get axes handle and clear both left and right
axeshandle = newplot;
cla reset

% Plot the audio input against time
plot(axeshandle,t,audioIn)
grid on
hold on
ylabel(getString(message('audio:convenienceplots:Amplitude')))
xlabel(getString(message('audio:detecttargetsound:TimeAxis')))

% Add patches indicating the regions of detected target sound
amax = max(audioIn);
amin = min(audioIn);
for idx = 1:size(boundaries,1)
    xline(axeshandle,t(boundaries(idx,1)),Color=[0 0.4470 0.7410],LineWidth=1.2);
    patch(axeshandle,[t(boundaries(idx,1)),t(boundaries(idx,1)),t(boundaries(idx,2)),t(boundaries(idx,2))], ...
        [amin,amax,amax,amin], ...
        [0.3010 0.7450 0.9330], ...
        FaceAlpha=0.2,EdgeColor="none");
    xline(axeshandle,t(boundaries(idx,2)),Color=[0 0.4470 0.7410],LineWidth=1.2);
end
axis tight

% Plot the probability vector against time
yyaxis right
sampleprob = iframeprob2sampleprob(prob,fs,expfs,numel(audioIn));
plot(axeshandle,t,sampleprob,Color=[0.8500 0.3250 0.0980],LineStyle="--",LineWidth=1.2)
ylim([0,1])
ylabel(getString(message('audio:convenienceplots:Probability')))

% Add plot title
if isempty(boundaries)
    title(getString(message('audio:detecttargetsound:PlotTitleNotargetsound')))
else
    title(getString(message('audio:detecttargetsound:PlotTitle')))
end
yyaxis left
hold off

end

function sampleprob = iframeprob2sampleprob(probs,fs,expfs,N)
% Hold the probability for the number of samples per hop
% Because the last frame is not overlapped in front, its decision
% gets held longer.
sampleprob_expfs = cat(1,repelem(probs(1),80,1),repelem(probs(2:end-1),160,1),repelem(probs(end),280,1));

% Resample the probility back to original sample rate
sampleprob_fs = iNearestNeighborResample(sampleprob_expfs,fs,expfs);

% Clip off any padding introduced
sampleprob_fs = sampleprob_fs(1:N);

% Clamp the probability to range [0,1]
sampleprob = iclampProb(sampleprob_fs);
end

function y = iNearestNeighborResample(x,targetFs,originalFs)
% Get total number of samples after resampling
numOriginalSamples = numel(x);
numResampledSamples = round(numOriginalSamples*(targetFs/originalFs));

% Get resample indices
resampleIndices = round(linspace(1,numOriginalSamples,numResampledSamples));

% Resample
y = x(resampleIndices);
end