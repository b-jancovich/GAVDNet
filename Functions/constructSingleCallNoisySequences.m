function constructSingleCallNoisySequences(ads_cleanSignals, ads_noise, numSequences, snrRange, bandwidth, sequencesPath)
% CONSTRUCTSINGLECALLNOISYSEQUENCEs Creates sequences with single noisy animal calls
%
% This function constructs synthetic audio sequences by placing a single clean 
% call sample at a random position within background noise. The power of 
% the call and noise are measured in band-limited and time-windowed segments, 
% then the call is scaled to achieve a specific signal-to-noise ratio.
%
% Sequence duration is set as 3x the duration of the clean signal.
%
% Inputs:
%   ads_cleanSignals - audioDatastore containing clean call samples
%   ads_noise - audioDatastore containing noise samples
%   numSequences - number of sequences to generate
%   snrRange - two-element vector [min, max] specifying SNR range in dB
%   bandwidth - two-element vector [min, max] specifying frequency range in Hz
%   sequencesPath - path to save the generated sequences and masks
%
% Outputs:
%   None (saves sequences and masks to disk as .mat files)
%
% Notes:
%   - Sequences are saved with filename 'audiosequence_and_mask_N.mat'
%   - Each .mat file contains:
%     * audioSequence - the noisy audio sequence
%     * mask - binary mask indicating call presence
%     * sequenceSNRs - SNR value used for the call
%   - The sequence duration is calculated as: cleanSignalDuration * 2
%   - Call is placed at a random position within the sequence
%   - Both noise and call are filtered within the specified bandwidth
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%% Begin

% Get sample rate from the clean signals datastore
reset(ads_cleanSignals);
[~, info] = read(ads_cleanSignals);
fs = info.SampleRate;
reset(ads_cleanSignals);

% Check if both datastores have the same sample rate
reset(ads_noise);
[~, noiseInfo] = read(ads_noise);
noiseFs = noiseInfo.SampleRate;
reset(ads_noise);
if fs ~= noiseFs
    error('Clean signals and noise must have the same sample rate');
end

% Init Filter Params
n = 8; % Order
Rp = 0.1; % Max passband ripple
Rs = 100; % Target stopband attenuation
nyq = fs / 2; % Nyquist Frequency
Wp = bandwidth / nyq; % Passband Edges (normalized freqs)

% Build bandpass filter
[b, a] = ellip(n, Rp, Rs, Wp, "bandpass", "ctf");

% Loop through each sequence
for seqIdx = 1:numSequences
    % User display updates
    if seqIdx == 1
        fprintf('Generating sequence %d of %d\n', seqIdx, numSequences);
    elseif mod(seqIdx, 1000) == 0
        fprintf('\n')
        fprintf('Generating sequence %d of %d\n', seqIdx, numSequences);
    elseif mod(seqIdx, 60) == 0
        fprintf('.')
    end
        
    % Read one clean signal
    if ~hasdata(ads_cleanSignals)
        reset(ads_cleanSignals);
    end
    [cleanSignal, ~] = read(ads_cleanSignals);
    
    % Ensure clean signal is a column vector
    cleanSignal = cleanSignal(:);
    cleanSignalDuration = length(cleanSignal) / fs;
    
    % Calculate sequence duration
    sequenceDuration = cleanSignalDuration * 3;

    % Read noise segment for the entire sequence
    noise = getNoiseEfficient(ads_noise, sequenceDuration, fs);
    
    % Ensure noise is a column vector
    noise = noise(:);

    % Generate random call position
    sequenceLengthSamples = length(noise);
    cleanSignalLengthSamples = length(cleanSignal);
    maxStartPos = sequenceLengthSamples - cleanSignalLengthSamples + 1;
    
    if maxStartPos < 1
        error('Clean signal is longer than sequence duration');
    end
    
    callStartIdx = randi([1, maxStartPos]);
    callEndIdx = callStartIdx + cleanSignalLengthSamples - 1;

    % Set SNR for this sequence
    sequenceSNR = snrRange(1) + (snrRange(2) - snrRange(1)) * rand(1, 1);

    % Init the sequence and the signal-presence mask
    cleanSequence = zeros(size(noise));
    mask = zeros(size(noise));

    % Get the noise that will be superimposed over this call
    noiseSegment = noise(callStartIdx:callEndIdx);

    % Filter the noise
    noiseSegment = ctffilt(b, a, noiseSegment);

    % Measure power of filtered noise segment
    noisePower = mean(noiseSegment.^2);

    % Filter the clean signal
    cleanSignalFiltered = ctffilt(b, a, cleanSignal);

    % Measure power of filtered clean signal
    cleanSigFilteredPower = mean(cleanSignalFiltered.^2);

    % Scale signal to achieve target SNR
    targetLinearSNR = 10^(sequenceSNR / 10);
    scalingFactor = sqrt(noisePower * targetLinearSNR / cleanSigFilteredPower);
    scaledCleanSignal = cleanSignal * scalingFactor;

    % Place the scaled call in the sequence
    cleanSequence(callStartIdx:callEndIdx) = scaledCleanSignal;
    
    % Mark the mask as true at the call indices
    mask(callStartIdx:callEndIdx) = 1;

    % Add the noise to the sequence:
    noisySequence = cleanSequence + noise;

    % Normalize to avoid clipping
    maxAmp = max(abs(noisySequence));
    noisySequence = noisySequence / maxAmp * 0.99; % Leave some headroom
    
    % Save the sequence and mask
    saveSequenceAndMask(noisySequence, mask, seqIdx, sequencesPath, sequenceSNR);
end
end

%% Helper Functions

function noise = getNoiseEfficient(ads_noise, duration, fs)
% Read noise from datastore with efficient handling of long noise files
% Uses persistent variables to store leftover noise between calls

    persistent remainingNoise;
    
    % Initialize persistent variable if empty
    if isempty(remainingNoise)
        remainingNoise = [];
    end
    
    noiseSamplesNeeded = round(duration * fs);
    
    % Check if we have enough remaining noise
    if length(remainingNoise) >= noiseSamplesNeeded
        % Use part of remaining noise
        noise = remainingNoise(1:noiseSamplesNeeded);
        remainingNoise = remainingNoise(noiseSamplesNeeded+1:end);
        return;
    end
    
    % Need to load more noise
    noise = remainingNoise; % Start with what we have
    failCount = 0;
    
    while length(noise) < noiseSamplesNeeded
        try
            % Read a chunk
            [chunk, ~] = read(ads_noise);

            % Normalize and DC Center the noise chunk
            chunk = (chunk - mean(chunk)) / max(abs(chunk - mean(chunk)));

            % Ensure chunk is valid
            if isValidAudio(chunk) == true
                % Ensure chunk is column vector
                chunk = chunk(:);
                
                % Check if this chunk is much longer than what we need
                samplesStillNeeded = noiseSamplesNeeded - length(noise);
                
                if length(chunk) > 2 * samplesStillNeeded
                    % Take what we need and save the rest
                    noise = [noise; chunk(1:samplesStillNeeded)];
                    remainingNoise = chunk(samplesStillNeeded+1:end);
                    break;
                else
                    % Use entire chunk
                    noise = [noise; chunk];
                end
            end
        catch
            failCount = failCount + 1;
            % if we have a file read error, skip this file.
            if failCount < 100
                continue
            else
                disp('Resetting ADS...')
                reset(ads_noise)
                failCount = 0;
            end
        end
    end

    % If we still don't have enough after several cycles, recycle what we have
    if length(noise) < noiseSamplesNeeded && ~isempty(noise)
        originalNoise = noise;
        numRepeats = ceil(noiseSamplesNeeded / length(noise));
        noiseExtended = [];

        for j = 1:numRepeats
            if mod(j, 2) == 1
                % Odd repetitions: keep original direction
                noiseExtended = [noiseExtended; originalNoise];
            else
                % Even repetitions: reverse direction
                noiseExtended = [noiseExtended; flipud(originalNoise)];
            end
        end

        noise = noiseExtended;
    end

    % Trim or pad noise to exact length
    if length(noise) >= noiseSamplesNeeded
        % If we have extra, save it for next time
        if length(noise) > noiseSamplesNeeded
            remainingNoise = noise(noiseSamplesNeeded+1:end);
        end
        noise = noise(1:noiseSamplesNeeded);
    else
        % Pad with zeros, ensuring column vector
        noise = [noise; zeros(noiseSamplesNeeded - length(noise), 1)];
    end
    
    % Final ensure column vector
    noise = noise(:);
end

function saveSequenceAndMask(audioSequence, mask, seqIdx, sequencesPath, sequenceSNRs)
    % Save the sequence and mask to a mat file
    filename = fullfile(sequencesPath, sprintf('audiosequence_and_mask_%d.mat', seqIdx));
    save(filename, 'audioSequence', 'mask', 'sequenceSNRs');
end