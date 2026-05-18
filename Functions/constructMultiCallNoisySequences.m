function constructMultiCallNoisySequences(ads_cleanSignals, ads_noise, ...
    numSequences, numCallsPerSequence, sequenceDuration, minCallSeparation,...
    snrRange, bandwidth, sequencesPath, chorusParams)
% constructMultiCallNoisySequences Creates sequences of noisy animal calls with controlled SNR
%
% This function constructs synthetic audio sequences by placing clean call samples
% at random non-overlapping positions within background noise. Both the calls and
% noise are bandpass filtered, then the calls are scaled based on filtered power
% measurements to achieve specific signal-to-noise ratios. Optionally, a
% synthetic chorus track (overlapping distant conspecific calls) is mixed
% into a configurable fraction of sequences. Chorus is always labelled
% negative because the per-sample mask is set only inside the discrete
% call placement loop; chorus is added to the noise track after that
% loop, never to the mask.
%
% Inputs:
%   ads_cleanSignals    - audioDatastore containing clean call samples
%   ads_noise           - audioDatastore containing noise samples
%   numSequences        - number of sequences to generate
%   numCallsPerSequence - number of calls to place in each sequence
%   sequenceDuration    - duration of each sequence in seconds
%   minCallSeparation   - minimum separation between calls in seconds
%   snrRange            - two-element vector [min, max] specifying SNR range in dB
%   bandwidth           - two-element vector [min, max] specifying frequency range in Hz
%   sequencesPath       - path to save the generated sequences and masks
%   chorusParams        - (optional) struct with chorus injection settings. Required fields:
%                           .enableChorus                       (logical)
%                           .chorus_probability                 (scalar, [0,1])
%                           .num_calls_in_chorus                (positive int)
%                           .chorus_calls_level_range           ([min, max] dB)
%                           .chorus_call_overlap_range          ([min, max] frac in (0,1))
%                           .chorus_sequence_level_range        (scalar dB, positive)
%                           .chorus_modulation_period_s         (scalar s)
%                           .chorus_to_calls_snr_offset_range   ([min, max] dB; negative offsets keep
%                                                                chorus below the loudest discrete call)
%                         If omitted, chorus injection is disabled.
%
% Outputs:
%   None (saves sequences and masks to disk as .mat files)
%
% Notes:
%   - Both datastores must have the same sample rate
%   - Sequences are saved with filename 'audiosequence_and_mask_N.mat'
%   - Each .mat file contains:
%     * audioSequence  - the noisy audio sequence (normalized to 0.99 max amplitude)
%     * mask           - binary mask indicating call presence (chorus does NOT set mask=true)
%     * sequenceSNRs   - array of intended per-call SNR values (dB). When chorus is
%                        present, the EFFECTIVE SNR experienced by the network is lower
%                        than these values; chorusSNR_used and chorusGain_used quantify
%                        the chorus contribution.
%     * chorusEnabled  - logical, whether chorus was injected for this sequence
%     * chorusSNR_used - chorus SNR vs filtered noise (dB), or NaN if chorus skipped
%     * chorusGain_used- linear gain applied to the unit-RMS chorus track, or NaN
%   - Calls are placed with random non-overlapping positions with minimum separation
%   - Both noise and calls are bandpass filtered using 8th-order elliptic filter
%   - SNR scaling is based on power measurements of filtered signals
%   - If insufficient noise is available, noise segments are recycled (forward/reverse)
%   - Call positions are sorted chronologically after random placement
%
% Filter specifications:
%   - 8th-order elliptic bandpass filter
%   - 0.1 dB maximum passband ripple
%   - 100 dB target stopband attenuation
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%% Begin

% Default: chorus injection disabled if chorusParams was not supplied.
if nargin < 10 || isempty(chorusParams)
    chorusParams = struct('enableChorus', false);
end

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
    if seqIdx == 1 || mod(seqIdx, 100) == 0
        fprintf('Generating sequence %d of %d\n', seqIdx, numSequences);
    end
    
    % Read noise segment for the entire sequence
    noise = getNoise(ads_noise, sequenceDuration, fs);
    
    % Ensure noise is a column vector
    noise = noise(:);

    % Pre-read all clean signals for this sequence to get their actual lengths
    cleanSignals = cell(numCallsPerSequence, 1);
    callLengths = zeros(numCallsPerSequence, 1);
    
    for i = 1:numCallsPerSequence
        if ~hasdata(ads_cleanSignals)
            reset(ads_cleanSignals);
        end
        [cleanSignal, ~] = read(ads_cleanSignals);
        % Ensure clean signal is a column vector
        cleanSignals{i} = cleanSignal(:);
        callLengths(i) = length(cleanSignals{i});
    end

    % Generate call positions based on actual call lengths
    [callStartIndices, callEndIndices, sortOrder] = generateCallPositions(callLengths, ...
        minCallSeparation, fs, sequenceDuration, seqIdx);

    % Reorder clean signals and call lengths to match sorted positions
    cleanSignals = cleanSignals(sortOrder);

    % Set SNRs for this sequence
    sequenceSNRs = snrRange(1) + (snrRange(2) - snrRange(1)) * rand(1, numCallsPerSequence);

    % Init the sequence and the signal-presence mask
    cleanSequence = zeros(size(noise));
    mask = zeros(size(noise));

    % Place calls in the sequence
    for i = 1:numCallsPerSequence
        % Retrieve pre-read clean signal (now in sorted order)
        cleanSignal = cleanSignals{i};
        
        % Use the pre-calculated indices
        callStartIdx = callStartIndices(i);
        callEndIdx = callEndIndices(i);

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
        targetLinearSNR = 10^(sequenceSNRs(i) / 10);
        scalingFactor = sqrt(noisePower * targetLinearSNR / cleanSigFilteredPower);
        scaledCleanSignal = cleanSignal * scalingFactor;
        
        % Ensure the scaled signal has the correct length and orientation
        if length(scaledCleanSignal) ~= (callEndIdx - callStartIdx + 1)
            error('Signal length mismatch: scaled signal has %d samples but trying to place in %d samples', ...
                length(scaledCleanSignal), callEndIdx - callStartIdx + 1);
        end

        % Place the scaled call in the sequence
        cleanSequence(callStartIdx:callEndIdx) = scaledCleanSignal;
        
        % Mark the mask as true at the call indices
        mask(callStartIdx:callEndIdx) = true;
    end

    % --- Optional chorus injection --------------------------------------
    % Chorus is mixed into the noise track AFTER the discrete call
    % placement loop. The mask is never touched here, so any chorus that
    % coincides with a discrete call inherits the call's mask=true (the
    % call's identity wins); chorus elsewhere stays mask=false (the
    % implicit-negative invariant the network needs). The whole noise
    % track (including chorus) then goes through the existing peak
    % normalisation below, so no further scaling is required.
    chorusEnabled    = false;
    chorusSNR_used   = NaN;
    chorusGain_used  = NaN;
    if chorusParams.enableChorus && rand() < chorusParams.chorus_probability
        chorus = generateChorus(ads_cleanSignals.Files, fs, sequenceDuration, chorusParams);

        % Scale chorus so its filtered RMS matches a target SNR vs the
        % filtered noise. The target SNR is tied to the loudest discrete
        % call's SNR via chorus_to_calls_snr_offset_range.
        nFilt = ctffilt(b, a, noise);
        cFilt = ctffilt(b, a, chorus);
        nPow  = mean(nFilt .^ 2);
        cPow  = mean(cFilt .^ 2);

        offRange  = chorusParams.chorus_to_calls_snr_offset_range;
        offset_dB = offRange(1) + (offRange(2) - offRange(1)) * rand;
        chorusSNR_used  = max(sequenceSNRs) + offset_dB;
        chorusGain_used = sqrt(nPow * 10 ^ (chorusSNR_used / 10) / max(eps, cPow));

        noise = noise + chorus * chorusGain_used;
        chorusEnabled = true;
    end
    % --------------------------------------------------------------------

    % Add the noise to the sequence:
    noisySequence = cleanSequence + noise;

    % Normalize to avoid clipping
    maxAmp = max(abs(noisySequence));
    noisySequence = noisySequence / maxAmp * 0.99; % Leave some headroom
    
    % % Debug figure
    % figure(1)
    % tiledlayout(4,1)
    % nexttile
    % plot(cleanSequence)
    % nexttile
    % plot(noisySequence)
    % nexttile
    % plot(mask)
    % nexttile
    % spectrogram(noisySequence, 250, 230, 2048, fs, 'yaxis')
    
    % Save the sequence and mask
    saveSequenceAndMask(noisySequence, mask, seqIdx, sequencesPath, ...
        sequenceSNRs, chorusEnabled, chorusSNR_used, chorusGain_used);
end
end

%% Helper Functions

function noise = getNoise(ads_noise, duration, fs)
% read noise from datastore and concatenate, loop, reverse-loop until we
% have noise of sufficient duration.

    % Read noise chunk
    noiseSamplesNeeded = round(duration * fs);
    noise = [];
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
                noise = [noise; chunk(:)]; 
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
        noise = noise(1:noiseSamplesNeeded);
    else
        % Pad with zeros, ensuring column vector
        noise = [noise; zeros(noiseSamplesNeeded - length(noise), 1)];
    end
    
    % Final ensure column vector
    noise = noise(:);
end

function [callStartIndices, callEndIndices, sortOrder] = generateCallPositions(callLengthsSamples, ...
    minSeparation, fs, sequenceDuration, seqIdx)
% GENERATECALLPOSITIONS Generate random non-overlapping positions for calls
% Inputs:
%   callLengthsSamples - Array of actual call lengths in samples
%   minSeparation      - Minimum separation between calls (seconds)
%   fs                 - Sample rate (Hz)
%   sequenceDuration   - Total duration of sequence (seconds)
%   seqIdx             - Sequence index (for error reporting)
%
% Outputs:
%   callStartIndices - Start sample indices for each call (sorted by start time)
%   callEndIndices   - End sample indices for each call (sorted by start time)
%   sortOrder        - Permutation vector showing original call order

% Number of calls
nCallsPerSequence = length(callLengthsSamples);

% Initialize output arrays
callStartIndices = zeros(nCallsPerSequence, 1);
callEndIndices = zeros(nCallsPerSequence, 1);

% Convert time-based parameters to samples
minSepSamples = round(minSeparation * fs);
sequenceLengthSamples = round(sequenceDuration * fs);

% Calculate minimum space required
totalCallSamples = sum(callLengthsSamples);
totalMinSpace = totalCallSamples + (nCallsPerSequence - 1) * minSepSamples;

if totalMinSpace > sequenceLengthSamples
    error('Sequence %d: Cannot fit %d calls with total duration %.2fs and separation %.2fs in %.2fs sequence', ...
        seqIdx, nCallsPerSequence, totalCallSamples/fs, minSeparation, sequenceDuration);
end

% Generate non-overlapping random positions
placedCalls = struct('start', {}, 'end', {}, 'callIdx', {});

for callIdx = 1:nCallsPerSequence
    validPosition = false;
    attempts = 0;
    maxAttempts = 10000;
    
    currentCallLength = callLengthsSamples(callIdx);
    
    while ~validPosition && attempts < maxAttempts
        attempts = attempts + 1;
        
        % Generate random start position
        maxStartPos = sequenceLengthSamples - currentCallLength + 1;
        if maxStartPos < 1
            error('Call %d (length %d samples) cannot fit in sequence', callIdx, currentCallLength);
        end
        
        candidateStart = randi([1, maxStartPos]);
        candidateEnd = candidateStart + currentCallLength - 1;
        
        % Check for conflicts with previously placed calls
        validPosition = true;
        for prevIdx = 1:length(placedCalls)
            % Check if candidate overlaps with existing call considering minimum separation
            if ~((candidateEnd + minSepSamples < placedCalls(prevIdx).start) || ...
                 (candidateStart > placedCalls(prevIdx).end + minSepSamples))
                validPosition = false;
                break;
            end
        end
        
        if validPosition
            % Add to placed calls
            placedCalls(end+1).start = candidateStart;
            placedCalls(end).end = candidateEnd;
            placedCalls(end).callIdx = callIdx;
            
            callStartIndices(callIdx) = candidateStart;
            callEndIndices(callIdx) = candidateEnd;
        end
    end
    
    if ~validPosition
        error('Failed to place call %d (length %d samples) in sequence %d after %d attempts', ...
            callIdx, currentCallLength, seqIdx, maxAttempts);
    end
end

% Sort calls by start time for sequential processing
[callStartIndices, sortOrder] = sort(callStartIndices);
callEndIndices = callEndIndices(sortOrder);

% Verify minimum separation in sorted order (debugging check)
for i = 2:nCallsPerSequence
    separation = callStartIndices(i) - callEndIndices(i-1) - 1;
    if separation < minSepSamples
        warning('Sequence %d: Separation between calls %d and %d is %d samples (%.3fs), less than minimum %d samples (%.3fs)', ...
            seqIdx, i-1, i, separation, separation/fs, minSepSamples, minSeparation);
    end
end

end

function saveSequenceAndMask(audioSequence, mask, seqIdx, sequencesPath, ...
        sequenceSNRs, chorusEnabled, chorusSNR_used, chorusGain_used)
    % Save the sequence, mask, intended per-call SNRs, and chorus metadata
    filename = fullfile(sequencesPath, sprintf('audiosequence_and_mask_%d.mat', seqIdx));
    save(filename, 'audioSequence', 'mask', 'sequenceSNRs', ...
        'chorusEnabled', 'chorusSNR_used', 'chorusGain_used');
end
