function [audioSegments, splitIndices, changePtsIndices] = eventSplitter(audio, fs, smoothWindowDuration, overlapDuration)

% Remove DC Offset
audioMeasure = audio - mean(audio);

% Design high pass filter @ 5Hz
n = 8; % Order
Rp = 0.1; % Passband Ripple
Rs = 90; % Stopband Ripple
nyq = fs/2; % Nyquist Freq
Wp = 5 / nyq; % Normalized cutoff frequency
[b,a] = ellip(n, Rp, Rs, Wp, "high", "ctf");

% Filter audio
audioMeasure = ctffilt(b, a, audioMeasure);

% Normalize to max=1
audioMeasure = audioMeasure ./ max(abs(audioMeasure));

if isgpuarray(audioMeasure)
    audioMeasure = gather(audioMeasure);
end

% Compute RMS envelope of audio with window sizes of 1s and 60s
[audioEnvShort, ~] = envelope(audioMeasure, 1*fs, 'rms');
[audioEnvLong, ~] = envelope(audioMeasure, 60*fs, 'rms');

% Run detection on long and short-time envelopes
[isExtremeShort, ~, ~] = detectExtremeEvents(audioEnvShort);
[isExtremeLong, ~, ~] = detectExtremeEvents(audioEnvLong);

% If extreme events look likely, proceed with splitting.
if isExtremeShort || isExtremeLong

    % Choose envelope based on detection results (keep your existing logic)
    if isExtremeLong && isExtremeShort
        masterEnv = audioEnvShort;
    elseif isExtremeLong && ~isExtremeShort
        masterEnv = audioEnvLong;
    elseif ~isExtremeLong && isExtremeShort
        masterEnv = audioEnvShort;
    end

    % smooth the envelope 
    smoothWindowSamps = fs*(smoothWindowDuration);
    masterEnv = smoothdata(masterEnv, "movmean", smoothWindowSamps);

    % Find change points
    % changePts = findchangepts(masterEnv);
    changePtsLogical = ischange(masterEnv, 'mean', Threshold=10);
    changePtsIndices = find(changePtsLogical);
    
    % Define audio segment boundaries including start and end points
    segmentBounds = [1; changePtsIndices(:); length(audio)];
    segmentBounds = unique(segmentBounds);

    % Initialize cell array for subsequences
    numSegments = length(segmentBounds) - 1;
    audioSegments = cell(numSegments, 1);

    % Extract each segment with overlap
    overlapLen = overlapDuration * fs;
    splitIndices = zeros(numSegments, 2);
    for i = 1:numSegments
        startIdx = segmentBounds(i);
        if i < numSegments  % Add overlap for all segments except the last
            endIdx = min(segmentBounds(i+1) - 1 + overlapLen, length(audio));
        else  % Last segment - no overlap
            endIdx = segmentBounds(i+1) - 1;
        end
        audioSegments{i} = audio(startIdx:endIdx);
        splitIndices(i, :) = [startIdx, endIdx];
    end

    fprintf('\tSplit audio into %d segments with %d samples overlap\n', numSegments, overlapLen);
else
    fprintf('\tNo extreme events detected. Splitting not required.\n');
    audioSegments = {audio};
    splitIndices = [1, length(audio)];
    changePtsIndices = [];
end
