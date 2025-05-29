function constructNoisySequences(ads_clean_signals, ads_noise, numSequences,...
    sequenceDuration, snrRange, ICI, ICI_variation, sequencesPath)
% CONSTRUCTNOISYSEQUENCES Creates sequences of noisy animal calls with varying patterns
%
% Inputs:
%   ads_clean_signals - audioDatastore containing clean call samples
%   ads_noise - audioDatastore containing noise samples
%   numSequences - number of sequences to generate
%   sequenceDuration - duration of each sequence in seconds
%   snrRange - two-element vector [min, max] specifying SNR range in dB
%   ICI - Inter-Call Interval in seconds (average gap between calls)
%   ICI_variation - variation in ICI (± seconds)
%   minSilenceSegment - minimum silence between different individuals' calls
%   sequencesPath - path to save the generated sequences and masks
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Get sample rate from the clean signals datastore
reset(ads_clean_signals);
[~, info] = read(ads_clean_signals);
fs = info.SampleRate;
reset(ads_clean_signals);

% Check if both datastores have the same sample rate
reset(ads_noise);
[~, noiseInfo] = read(ads_noise);
noiseFs = noiseInfo.SampleRate;
reset(ads_noise);

if fs ~= noiseFs
    error('Clean signals and noise must have the same sample rate');
end

% Estimate average call duration for noise power measurement
avgCallDuration = estimateAverageCallDuration(ads_clean_signals, fs);
windowSize = round(avgCallDuration * fs);

% Loop through each sequence
for seqIdx = 1:numSequences
    fprintf('Generating sequence %d of %d\n', seqIdx, numSequences);
    
    % For every even numbered sequence, model a single individual, and for 
    % odd numbered sequences, multiple individuals.
    isSingleIndividual = mod(seqIdx, 2) == 0;
    
    % Read noise segment for the entire sequence
    noise = readNoiseSegment(ads_noise, sequenceDuration, fs);
    
    % Measure noise power in windows
    noisePowers = measureNoiseInWindows(noise, windowSize);
    
    if isSingleIndividual
        % Single individual sequence
        sequenceSNR = snrRange(1) + (snrRange(2) - snrRange(1)) * rand();
        [audioSequenceClean, mask] = createSingleIndividualSequence(...
            ads_clean_signals, noise, noisePowers, fs, sequenceSNR, ...
            ICI, ICI_variation, windowSize);
    else
        % Multiple individuals sequence
        [audioSequenceClean, mask] = createMultipleIndividualsSequence(...
            ads_clean_signals, noise, noisePowers, fs, sequenceDuration, ...
            snrRange, ICI, ICI_variation, windowSize);
    end
   
    % Add noise to the sequence
    audioSequence = audioSequenceClean + noise;

    % Normalize to avoid clipping
    maxAmp = max(abs(audioSequence));
    if maxAmp > 0
        audioSequence = audioSequence / maxAmp * 0.9; % Leaving some headroom
    end
    
    % Save the sequence and mask
    saveSequenceAndMask(audioSequence, mask, seqIdx, sequencesPath);
end
end

function avgDuration = estimateAverageCallDuration(ads_clean_signals, fs)
    % Estimate average call duration from a sample of clean calls
    numSamplesToRead = min(10, length(ads_clean_signals.Files));
    callDurations = zeros(numSamplesToRead, 1);
    
    reset(ads_clean_signals);
    for i = 1:numSamplesToRead
        if ~hasdata(ads_clean_signals)
            reset(ads_clean_signals);
        end
        try
            [callAudio, ~] = read(ads_clean_signals);
            if isValidAudio(callAudio)
                callDurations(i) = length(callAudio) / fs;
            else
                callDurations(i) = 0.1; % Default if invalid
            end
        catch
            callDurations(i) = 0.1; % Default if error
        end
    end
    reset(ads_clean_signals);
    
    % Calculate average, removing zeros
    validDurations = callDurations(callDurations > 0);
    if isempty(validDurations)
        avgDuration = 0.1; % Default if all samples were invalid
    else
        avgDuration = mean(validDurations);
    end
end

function noise = readNoiseSegment(ads_noise, duration, fs)
   
    % Read noise segment
    noiseSamplesNeeded = round(duration * fs);
    noise = [];
    failCount = 0;
    while length(noise) < noiseSamplesNeeded
        try
            [chunk, ~] = read(ads_noise);
            
            % Ensure chunk is valid
            if isValidAudio(chunk) == true
                noise = [noise; chunk];
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
    if length(noise) < noiseSamplesNeeded && length(noise) > 0
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
        noise = [noise; zeros(noiseSamplesNeeded - length(noise), 1)];
    end
end

function noisePowers = measureNoiseInWindows(noise, windowSize)
    % Measure average noise power in overlapping windows
    
    % Use 50% overlap for windows
    hopSize = round(windowSize/2);
    numWindows = floor((length(noise) - windowSize) / hopSize) + 1;
    
    noisePowers = zeros(numWindows, 1);
    
    for i = 1:numWindows
        windowStart = (i-1)*hopSize + 1;
        windowEnd = min(length(noise), windowStart + windowSize - 1);
        
        % Extract window
        window = noise(windowStart:windowEnd);
        
        % Calculate power (mean squared amplitude)
        noisePowers(i) = mean(window.^2);
    end
end

function [audioSequence, mask] = createSingleIndividualSequence(...
    ads_clean_signals, noise, noisePowers, fs, sequenceSNR, ...
    ICI, ICI_variation, windowSize)

    % Initialize sequence and mask
    seqLengthSamples = length(noise);
    
    % Determine call density pattern (start and end of calling bout)
    callPatternType = randi(5); % 5 different patterns
    
    switch callPatternType
        case 1 % Calls throughout the sequence
            boutStart = 1;
            boutEnd = seqLengthSamples;
        case 2 % Calls at the beginning
            boutStart = 1;
            boutEnd = round(seqLengthSamples * (0.25 + 0.35 * rand())); % First 25-60% of sequence
        case 3 % Calls at the end
            boutStart = round(seqLengthSamples * (0.65 - 0.35 * rand())); % Last 35-70% of sequence
            boutEnd = seqLengthSamples;
        case 4 % Calls in the middle
            midpoint = seqLengthSamples / 2;
            halfWidth = seqLengthSamples * (0.2 + 0.3 * rand()); % 20-50% of sequence centered in middle
            boutStart = max(1, round(midpoint - halfWidth/2));
            boutEnd = min(seqLengthSamples, round(midpoint + halfWidth/2));
        case 5 % Two separate bouts
            firstBoutEnd = round(seqLengthSamples * (0.3 + 0.2 * rand())); % First 30-50% is first bout
            secondBoutStart = round(seqLengthSamples * (0.6 + 0.2 * rand())); % Second bout starts at 60-80%
            
            % Place calls in both bouts
            [audioSequence1, mask1] = placeCallsInBout(ads_clean_signals, ...
                noise, noisePowers, fs, 1, firstBoutEnd, ICI, ...
                ICI_variation, sequenceSNR, windowSize);
            [audioSequence2, mask2] = placeCallsInBout(ads_clean_signals, ...
                noise, noisePowers, fs, secondBoutStart, seqLengthSamples, ...
                ICI, ICI_variation, sequenceSNR, windowSize);
            
            % Combine the results
            audioSequence = audioSequence1 + audioSequence2;
            mask = max(mask1, mask2);
            return;
    end
    
    % Place calls in the determined bout
    [audioSequence, mask] = placeCallsInBout(ads_clean_signals, noise, ...
        noisePowers, fs, boutStart, boutEnd, ICI, ICI_variation, ...
        sequenceSNR, windowSize);
end

function [audioSequence, mask] = placeCallsInBout(ads_clean_signals, noise, ...
    noisePowers, fs, boutStart, boutEnd, ICI, ICI_variation, ...
    sequenceSNR, windowSize)

    % Initialize sequence and mask
    seqLengthSamples = length(noise);
    audioSequence = zeros(seqLengthSamples, 1);
    mask = zeros(seqLengthSamples, 1);
    
    % Calculate available duration for the bout
    boutDuration = (boutEnd - boutStart + 1) / fs;
    
    % Calculate maximum possible number of calls for this bout
    avgGap = ICI + ICI_variation/2; % Average gap between calls
    maxCalls = floor(boutDuration / avgGap);
    
    % Determine actual number of calls (at least 1, up to maxCalls)
    numCalls = max(1, min(maxCalls, randi(maxCalls)));
    
    % Calculate call positions
    callPositions = distributeCallsInBout(boutStart, boutEnd, numCalls,...
        windowSize/fs, ICI, ICI_variation, fs);
    
    % Place calls at determined positions
    for i = 1:length(callPositions)
        position = callPositions(i);
        
        % Skip if position is invalid
        if position <= 0 || position + windowSize > seqLengthSamples
            continue;
        end
        
        % Read a clean call
        if ~hasdata(ads_clean_signals)
            reset(ads_clean_signals);
        end
        
        try
            [cleanCall, ~] = read(ads_clean_signals);
            
            % Skip if invalid audio
            if ~isValidAudio(cleanCall)
                continue;
            end
            
            % Random SNR variation around the sequence SNR (±1.5 dB)
            callSNR = sequenceSNR + (rand() * 3 - 1.5);
            
            % Find corresponding noise window for this position
            windowIdx = min(length(noisePowers), max(1, floor(position / (windowSize/2))));
            thisNoisePower = noisePowers(windowIdx);
            
            % Scale clean call to achieve desired SNR
            cleanCallPower = mean(cleanCall.^2);
            targetPower = thisNoisePower * 10^(callSNR/10);
            scaleFactor = sqrt(targetPower / cleanCallPower);
            scaledCall = cleanCall * scaleFactor;
            
            % Place call in sequence
            callEnd = min(seqLengthSamples, position + length(scaledCall) - 1);
            validLength = callEnd - position + 1;
            
            if validLength < length(scaledCall)
                scaledCall = scaledCall(1:validLength);
            end
            
            audioSequence(position:callEnd) = scaledCall;
            mask(position:callEnd) = 1;
        catch
            % Skip this call if there was an error
            continue;
        end
    end
end

function callPositions = distributeCallsInBout(boutStart, boutEnd, ...
    numCalls, avgCallDuration, ICI, ICI_variation, fs)
    % Calculate the time positions for calls within a bout
    
    % Initialize positions array
    callPositions = zeros(numCalls, 1);
    
    if numCalls <= 0
        return;
    end
    
    % Calculate mean gap between calls in samples
    meanGap = round(ICI * fs);
    variation = round(ICI_variation * fs);
    
    % Calculate average call duration in samples
    callDurSamples = round(avgCallDuration * fs);
    
    % Distribute calls
    if numCalls == 1
        % If only one call, place it randomly within the bout
        callPositions(1) = boutStart + randi(max(1, boutEnd - boutStart - callDurSamples));
    else
        % For multiple calls, calculate positions with ICI variation
        currentPos = boutStart;
        
        for i = 1:numCalls
            % Place the first call at start of bout
            if i == 1
                callPositions(i) = currentPos;
            else
                % Add random variation to gap for subsequent calls
                gap = meanGap + randi([-variation, variation]);
                currentPos = currentPos + gap;
                callPositions(i) = currentPos;
            end
            
            % Update position for next call
            currentPos = currentPos + callDurSamples;
            
            % Ensure we don't exceed the bout end
            if currentPos >= boutEnd
                % Truncate to actual placed calls
                callPositions = callPositions(1:i);
                break;
            end
        end
    end
end

function [audioSequence, mask] = createMultipleIndividualsSequence(...
    ads_clean_signals, noise, noisePowers, fs, sequenceDuration, snrRange,...
    ICI, ICI_variation, windowSize)

    % Initialize sequence and mask
    seqLengthSamples = length(noise);
    audioSequence = zeros(seqLengthSamples, 1);
    mask = zeros(seqLengthSamples, 1);
    
    % Determine number of individuals (2-5)
    numIndividuals = randi([2, 5]);
    
    % Create calls for each individual
    for indiv = 1:numIndividuals
        % Determine individual SNR for this caller
        individualSNR = snrRange(1) + (snrRange(2) - snrRange(1)) * rand();
        
        % Calculate bout start and end for this individual
        % Distribute individuals throughout the sequence with some overlap
        segmentSize = sequenceDuration / numIndividuals * 1.5; % Allow some overlap between individuals
        segmentStart = max(0, (indiv-1) * sequenceDuration/numIndividuals - segmentSize * 0.25);
        segmentEnd = min(sequenceDuration, segmentStart + segmentSize);
        
        boutStart = round(segmentStart * fs) + 1;
        boutEnd = round(segmentEnd * fs);
        
        % Place calls for this individual
        [individualAudio, individualMask] = placeCallsInBout(ads_clean_signals, ...
            noise, noisePowers, fs, boutStart, boutEnd, ICI, ICI_variation, ...
            individualSNR, windowSize);
        
        % Add to the sequence and update mask
        audioSequence = audioSequence + individualAudio;
        mask = max(mask, individualMask); % Combine masks
    end
end

function saveSequenceAndMask(audioSequence, mask, seqIdx, sequencesPath)
    % Save the sequence and mask to a mat file
    filename = fullfile(sequencesPath, sprintf('audiosequence_and_mask_%d.mat', seqIdx));
    save(filename, 'audioSequence', 'mask');
end