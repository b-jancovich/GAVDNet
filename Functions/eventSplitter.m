function [audioSegments, splitIndices, changePtsIndices] = eventSplitter(audio, fs, smoothWindowDuration, overlapDuration)
%EVENTSPLITTER Automatically segments audio at detected extreme events
%
%   [audioSegments, splitIndices, changePtsIndices] = eventSplitter(audio, fs, smoothWindowDuration, overlapDuration)
%
%   Detects extreme events in audio signals by analyzing RMS envelopes at
%   multiple time scales, then segments the audio at detected change points
%   with configurable overlap between segments.
%
%   INPUTS:
%       audio                 - Input audio signal (vector)
%       fs                    - Sample rate in Hz
%       smoothWindowDuration  - Duration for envelope smoothing in seconds
%       overlapDuration       - Overlap duration between segments in seconds
%
%   OUTPUTS:
%       audioSegments         - Cell array containing segmented audio data
%       splitIndices          - N×2 matrix of [start, end] indices for each segment
%       changePtsIndices      - Indices where change points were detected
%
%   The function uses dual-scale RMS envelope analysis (1s and 60s windows)
%   to detect extreme events, then applies change point detection using
%   mean-based thresholding. Audio is segmented at change points with
%   configurable overlap between adjacent segments.
%
%   ALGORITHM:
%   1. Preprocesses audio (DC removal, 5Hz high-pass filtering, normalization)
%   2. Computes short-term (1s) and long-term (60s) RMS envelopes
%   3. Detects extreme events in both envelopes using detectExtremeEvents()
%   4. Selects appropriate envelope based on detection results
%   5. Applies smoothing and change point detection
%   6. Segments audio with specified overlap
%
%   NOTES:
%   - Uses 8th-order elliptic high-pass filter with 5Hz cutoff
%   - Change point detection uses ischange() with mean threshold of 10
%   - GPU arrays are automatically converted to CPU arrays for processing
%   - If no extreme events detected, returns original audio as single segment
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation  
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
persistent b a

% Remove DC Offset
audioMeasure = audio - mean(audio);

% Design high pass filter @ 5Hz
if isempty(b) || isempty(a)
    n = 8; % Order
    Rp = 0.1; % Passband Ripple
    Rs = 90; % Stopband Ripple
    nyq = fs/2; % Nyquist Freq
    Wp = 5 / nyq; % Normalized cutoff frequency
    [b,a] = ellip(n, Rp, Rs, Wp, "high", "ctf");
end

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
