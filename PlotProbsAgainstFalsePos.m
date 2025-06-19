clear
close all
clc

% Path to disagreements file
disagreementsPath = "D:\GAVDNet\Chagos_DGS\Test Results\detector_vs_GT_disagreements_17-Jun-2025_07-47-32.mat";

% Path to test audio
audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\2007subset_small";

windowDur = 1;
overlapPerc = 95;
dynamicRange = 40;
bandwidth = [10, 60];
callDuration = 40;

% Load disagreements
load(disagreementsPath)

% Get false positives
FP = disagreements.falsePositives;

% Get list of audio files
audioFileList = dir(fullfile(audioPath, '*.wav'));

for i = 1:length(FP)
    filename = FP(i).AudioFilename;
    startIdx = FP(i).DetectionStartSamp;
    probs = FP(i).DetProbsSampleDomain;
    
    % Get audo
    if i==1 
        % Always retrieve the first file 
        [audio, fs] = audioread(fullfile(audioPath, filename));
        % Index out the detection from the file
        endIdx = startIdx + (callDuration*fs);
        detAudio = audio(startIdx:endIdx);

    elseif i > 1 && strcmp(filename, FP(i-1).AudioFilename) == true
        % If the current detection is in the same file as the last one, 
        % don't re-read the file, just index out the detection
        endIdx = startIdx + (callDuration*fs);
        detAudio = audio(startIdx:endIdx);

    elseif i > 1 && strcmp(filename, FP(i-1).AudioFilename) == false
        % This detection is in a new file. Read it in
        [audio, fs] = audioread(fullfile(audioPath, filename));
        % Index out the detection from the file
        endIdx = startIdx + (callDuration*fs);
        detAudio = audio(startIdx:endIdx);
    end

    % Audio time vector
    duration = length(detAudio)/fs;
    dt = 1/fs;
    tAudio = 0:dt:duration-dt;
    endPadLen = abs(length(detAudio) - length(probs));
    probsPadded = [probs, NaN(1, endPadLen)];

    % Compute spectrogram in power dB
    winLen = round(windowDur * fs);
    ovlp = round(winLen * (overlapPerc/100));
    [s, f, t] = spectrogram(detAudio, winLen, ovlp, 2048, fs, 'yaxis');
    s = pow2db(abs(s).^2);

    % Index out the bandwidth of interest
    bandwidthIndices = dsearchn(f, bandwidth(:));
    f = f(bandwidthIndices(1):bandwidthIndices(2));
    s = s(bandwidthIndices(1):bandwidthIndices(2), :);
    
    % Set color limits
    cMax = max(s, [], 'all');
    cMin = cMax - dynamicRange;

    % Draw figure
    figure(1)
    tiledlayout(2,1)
    nexttile
    yyaxis left
    plot(tAudio, detAudio)
    ylabel('Amplitude')
    xlabel('Time (s)')
    yyaxis right
    plot(tAudio, probsPadded)
    ylim([0, 1])
    xlim([tAudio(1), tAudio(end)])
    ylabel('Probability Score')

    nexttile
    imagesc(t, f, s)
    set(gca, 'ydir', 'normal')
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    colorbar
    clim([cMin, cMax])

    sgtitle(sprintf('Detection audio spectrogram and Probabilities - Dynamic Range Setting: %ddB\n%s - Sample Index %d to %d', ...
        dynamicRange, filename, startIdx, endIdx), Interpreter="none")

    waitforbuttonpress
end
