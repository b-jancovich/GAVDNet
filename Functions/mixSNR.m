function [noisySignal,requestedNoise] = mixSNR(signal,noise,ratio)
% [noisySignal,requestedNoise] = mixSNR(signal,noise,ratio) returns a noisy
% version of the signal, noisySignal. The noisy signal has been mixed with
% noise at the specified ratio in dB.

numSamples = size(signal,1);

% Convert to mono
noise = mean(noise,2);
signal = mean(signal,2);

% Remove DC component
noise = noise - mean(noise);
signal = signal - mean(signal);

% Trim or expand noise to match signal size
if size(noise,1)>=numSamples
    % Choose a random starting index such that you still have numSamples
    % after indexing the noise.
    start = randi(size(noise,1) - numSamples + 1);
    noise = noise(start:start+numSamples-1);
else
    numReps = ceil(numSamples/size(noise,1));
    temp = repmat(noise,numReps,1);
    start = randi(size(temp,1) - numSamples - 1);
    noise = temp(start:start+numSamples-1);
end

% Normalize
signalNorm = norm(signal);
noiseNorm = norm(noise);

% Calculate scaling factor
goalNoiseNorm = signalNorm/(10^(ratio/20));
factor = goalNoiseNorm/noiseNorm;

% Scale noise
requestedNoise = noise.*factor;

% Sum with signal
noisySignal = signal + requestedNoise;

noisySignal = noisySignal./max(abs(noisySignal));
end