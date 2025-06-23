clear
close all
clc

% Path to disagreements file
disagreementsPath = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\detector_vs_GT_disagreements_23-Jun-2025_08-31-56.mat";

% Path to test audio
audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\2007subset";

% Visualization parameters
windowDur = 0.85;
hopDur = 0.05;
dynamicRange = 40;
bandwidth = [10, 60];
callDuration = 40;  % Context duration around detection

% Load disagreements
load(disagreementsPath)

% Get false positives
FP = disagreements.falsePositives;

% Get list of audio files
audioFileList = dir(fullfile(audioPath, '*.wav'));

for i = 1:length(FP)
    filename = FP(i).AudioFilename;
    startIdx = FP(i).DetectionStartSamp;
    endIdx_detection = FP(i).DetectionEndSamp;
    probs = FP(i).probabilities;
    
    % Validate detection indices
    if isnan(startIdx) || isnan(endIdx_detection)
        warning('Invalid detection indices for FP #%d', i);
        continue;
    end
    
    % Get audio
    if i==1
        % Always retrieve the first file
        [audio, fs] = audioread(fullfile(audioPath, filename));
        
        % Check if file was read successfully
        if isempty(audio)
            warning('Could not read audio file: %s', filename);
            continue;
        end
        
    elseif i > 1 && strcmp(filename, FP(i-1).AudioFilename) == false
        % This detection is in a new file. Read it in
        [audio, fs] = audioread(fullfile(audioPath, filename));
        
        % Check if file was read successfully
        if isempty(audio)
            warning('Could not read audio file: %s', filename);
            continue;
        end
    end
    
    % Extract context audio with boundary checking
    contextStartIdx = max(1, startIdx - round(5*fs));  % 5s before detection
    contextEndIdx = min(length(audio), max(endIdx_detection + round(5*fs), startIdx + (callDuration*fs)));
    
    % Ensure we have valid indices
    if contextStartIdx >= contextEndIdx || contextStartIdx > length(audio)
        warning('Invalid context indices for FP #%d', i);
        continue;
    end
    
    detAudio = audio(contextStartIdx:contextEndIdx);
    
    % Skip if detection audio is too short
    if length(detAudio) < fs * 1  % At least 1 second
        warning('Detection audio too short (%d samples) for file %s', length(detAudio), filename);
        continue;
    end
    
    % Calculate detection boundaries relative to extracted audio segment
    detectionStartTime_rel = (startIdx - contextStartIdx) / fs;
    detectionEndTime_rel = (endIdx_detection - contextStartIdx) / fs;
    detectionDuration = detectionEndTime_rel - detectionStartTime_rel;
    
    % Ensure detection boundaries are within the extracted audio
    detectionStartTime_rel = max(0, detectionStartTime_rel);
    detectionEndTime_rel = min(length(detAudio)/fs, detectionEndTime_rel);
    
    % Audio time vector
    duration = length(detAudio)/fs;
    dt = 1/fs;
    tAudio = 0:dt:duration-dt;
    
    % Compute spectrogram in power dB
    winLen = round(windowDur * fs);
    hopLen = round(hopDur * fs);
    ovlp = winLen - hopLen;
    [s, f, t] = spectrogram(detAudio, winLen, ovlp, 2048, fs, 'yaxis');
    s = pow2db(abs(s).^2);
    
    % Index out the bandwidth of interest
    bandwidthIndices = dsearchn(f, bandwidth(:));
    f = f(bandwidthIndices(1):bandwidthIndices(2));
    s = s(bandwidthIndices(1):bandwidthIndices(2), :);
    
    % Create probability time vector aligned with detection boundaries
    if ~isempty(probs) && length(probs) > 1
        % Create time vector for probabilities spanning the detection period
        tProbs = linspace(detectionStartTime_rel, detectionEndTime_rel, length(probs));
        probsAligned = probs;
    else
        % Handle case with no or single probability value
        tProbs = [detectionStartTime_rel, detectionEndTime_rel];
        if isempty(probs)
            probsAligned = [0, 0];
        else
            probsAligned = [probs(1), probs(1)];
        end
    end
    
    % Set color limits
    cMax = max(s, [], 'all');
    cMin = cMax - dynamicRange;
    
    % Draw figure
    figure(1)
    clf  % Clear the figure for each detection
    tiledlayout(2,1)
    
    % Top subplot: Waveform and probabilities
    ax1 = nexttile;
    yyaxis left
    plot(tAudio, detAudio)
    ylabel('Amplitude')
    xlabel('Time (s)')
    yLimLeft = ylim;
    xlim([0, max(tAudio)])
    
    % Draw detection boundary box on waveform (left axis)
    hold on
    if detectionStartTime_rel >= 0 && detectionEndTime_rel <= max(tAudio)
        % Draw red dotted rectangle for detection boundaries
        rectangle('Position', [detectionStartTime_rel, yLimLeft(1), ...
                  detectionDuration, diff(yLimLeft)], ...
                  'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 2, 'FaceColor', 'none');
        
        % Add vertical lines at boundaries
        plot([detectionStartTime_rel, detectionStartTime_rel], yLimLeft, ...
             'r--', 'LineWidth', 2);
        plot([detectionEndTime_rel, detectionEndTime_rel], yLimLeft, ...
             'r--', 'LineWidth', 2);
    end
    
    yyaxis right
    plot(tProbs, probsAligned)
    ylim([0, 1])
    xlim([0, max(tAudio)])
    ylabel('Probability Score')
    
    hold off
    title(sprintf('FP #%d: Waveform and Neural Network Probabilities (Red box = Detection)', i))
    
    % Bottom subplot: Spectrogram
    ax2 = nexttile;
    imagesc(t, f, s)
    set(gca, 'ydir', 'normal')
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    c = colorbar;
    ylabel(c, 'Power (dB)')
    clim([cMin, cMax])
    
    % Draw detection boundary box on spectrogram
    hold on
    if detectionStartTime_rel >= 0 && detectionEndTime_rel <= max(t)
        % Draw red dotted rectangle for detection boundaries
        rectangle('Position', [detectionStartTime_rel, min(f), ...
                  detectionDuration, max(f) - min(f)], ...
                  'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 2, 'FaceColor', 'none');
        
        % Add vertical lines at boundaries  
        plot([detectionStartTime_rel, detectionStartTime_rel], [min(f), max(f)], ...
             'r--', 'LineWidth', 2);
        plot([detectionEndTime_rel, detectionEndTime_rel], [min(f), max(f)], ...
             'r--', 'LineWidth', 2);
    end
    hold off
    
    title('Spectrogram (Red box = Detection boundaries)')
    
    % Overall title with detection information
    sgtitle(sprintf('False Positive Detection #%d\nFile: %s | Detection Samples: %d-%d (%.2fs duration) | Confidence: %.3f\nContext Samples: %d-%d | Dynamic Range: %ddB', ...
        i, filename, startIdx, endIdx_detection, detectionDuration, FP(i).Confidence, ...
        contextStartIdx, contextEndIdx, dynamicRange), 'Interpreter', "none")
    
    % Print detection info to console
    fprintf('FP #%d: %s\n', i, filename);
    fprintf('  Detection samples: %d-%d (%.2fs duration)\n', startIdx, endIdx_detection, detectionDuration);
    fprintf('  Context samples: %d-%d\n', contextStartIdx, contextEndIdx);
    fprintf('  Detection time in plot: %.2f-%.2fs\n', detectionStartTime_rel, detectionEndTime_rel);
    fprintf('  Confidence: %.3f\n', FP(i).Confidence);
    fprintf('  Probability vector length: %d\n', length(probs));
    
    % Wait for user input before showing next detection
    fprintf('Showing False Positive %d of %d. Press any key to continue...\n', i, length(FP));
    waitforbuttonpress
end

fprintf('Finished displaying all %d false positive detections.\n', length(FP));

% clear
% close all
% clc
% 
% % Path to disagreements file
% disagreementsPath = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\detector_vs_GT_disagreements_23-Jun-2025_08-31-56.mat";
% 
% % Path to test audio
% audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\2007subset";
% 
% % Visualization parameters
% windowDur = 0.85;
% hopDur = 0.05;
% dynamicRange = 40;
% bandwidth = [10, 50];
% callDuration = 40;
% 
% % Load disagreements
% load(disagreementsPath)
% 
% % Get false positives
% FP = disagreements.falsePositives;
% 
% % Get list of audio files
% audioFileList = dir(fullfile(audioPath, '*.wav'));
% 
% for i = 1:length(FP)
%     filename = FP(i).AudioFilename;
%     startIdx = FP(i).DetectionStartSamp;
%     detEndIdx = FP(i).DetectionEndSamp;
%     probs = FP(i).probabilities;
% 
%     % Get audio
%     if i==1
%         % Always retrieve the first file
%         [audio, fs] = audioread(fullfile(audioPath, filename));
% 
%         % Check if file was read successfully
%         if isempty(audio)
%             warning('Could not read audio file: %s', filename);
%             continue;
%         end
% 
%         % Index out the detection from the file with boundary checking
%         endIdx = min(startIdx + (callDuration*fs), length(audio));
% 
%         % Ensure startIdx is valid
%         if startIdx < 1 || startIdx > length(audio)
%             warning('Invalid start index %d for file %s (length %d)', startIdx, filename, length(audio));
%             continue;
%         end
% 
%         detAudio = audio(startIdx:endIdx);
% 
%     elseif i > 1 && strcmp(filename, FP(i-1).AudioFilename) == true
%         % If the current detection is in the same file as the last one,
%         % don't re-read the file, just index out the detection
%         endIdx = min(startIdx + (callDuration*fs), length(audio));
% 
%         % Ensure startIdx is valid
%         if startIdx < 1 || startIdx > length(audio)
%             warning('Invalid start index %d for file %s (length %d)', startIdx, filename, length(audio));
%             continue;
%         end
% 
%         detAudio = audio(startIdx:endIdx);
% 
%     elseif i > 1 && strcmp(filename, FP(i-1).AudioFilename) == false
%         % This detection is in a new file. Read it in
%         [audio, fs] = audioread(fullfile(audioPath, filename));
% 
%         % Check if file was read successfully
%         if isempty(audio)
%             warning('Could not read audio file: %s', filename);
%             continue;
%         end
% 
%         % Index out the detection from the file with boundary checking
%         endIdx = min(startIdx + (callDuration*fs), length(audio));
% 
%         % Ensure startIdx is valid
%         if startIdx < 1 || startIdx > length(audio)
%             warning('Invalid start index %d for file %s (length %d)', startIdx, filename, length(audio));
%             continue;
%         end
% 
%         detAudio = audio(startIdx:endIdx);
%     end
% 
%     % Skip if detection audio is too short
%     if length(detAudio) < fs * 1  % At least 1 second
%         warning('Detection audio too short (%d samples) for file %s', length(detAudio), filename);
%         continue;
%     end
% 
%     % Audio time vector
%     duration = length(detAudio)/fs;
%     dt = 1/fs;
%     tAudio = 0:dt:duration-dt;
% 
%     % Compute spectrogram in power dB
%     winLen = round(windowDur * fs);
%     hopLen = round(hopDur * fs);
%     ovlp = winLen - hopLen;
%     [s, f, t] = spectrogram(detAudio, winLen, ovlp, 2048, fs, 'yaxis');
%     s = pow2db(abs(s).^2);
% 
%     % Index out the bandwidth of interest
%     bandwidthIndices = dsearchn(f, bandwidth(:));
%     f = f(bandwidthIndices(1):bandwidthIndices(2));
%     s = s(bandwidthIndices(1):bandwidthIndices(2), :);
% 
%     % Properly align probabilities with spectrogram time vector
%     if length(probs) == length(t)
%         % Perfect match - no adjustment needed
%         probsAligned = probs;
%         tProbs = t;
%     elseif length(probs) > length(t)
%         % Probabilities are longer - truncate to match spectrogram
%         probsAligned = probs(1:length(t));
%         tProbs = t;
%         warning('Probability vector longer than spectrogram - truncating');
%     else
%         % Probabilities are shorter - pad with NaN
%         probsAligned = [probs, NaN(1, length(t) - length(probs))];
%         tProbs = t;
%         warning('Probability vector shorter than spectrogram - padding with NaN');
%     end
% 
%     % Set color limits
%     cMax = max(s, [], 'all');
%     cMin = cMax - dynamicRange;
% 
%     % Draw figure
%     figure(1)
%     clf  % Clear the figure for each detection
%     tiledlayout(2,1)
% 
%     % Top subplot: Waveform and probabilities
%     nexttile
%     yyaxis left
%     plot(tAudio, detAudio)
%     ylabel('Amplitude')
%     xlabel('Time (s)')
% 
%     yyaxis right
%     plot(tProbs, probsAligned, 'LineWidth', 1.5)
%     hold on
%     xline(detEndIdx/fs, '--r')
%     hold off
%     ylim([0, 1])
%     xlim([tAudio(1), tAudio(end)])
%     ylabel('Probability Score')
%     title(sprintf('FP #%d: Waveform and Neural Network Probabilities', i))
% 
%     % Bottom subplot: Spectrogram
%     nexttile
%     imagesc(t, f, s)
%     set(gca, 'ydir', 'normal')
%     xlabel('Time (s)')
%     ylabel('Frequency (Hz)')
%     c = colorbar;
%     ylabel(c, 'Power (dB)')
%     clim([cMin, cMax])
%     title('Spectrogram')
% 
%     % Overall title with detection information
%     sgtitle(sprintf('False Positive Detection #%d\nFile: %s | Samples: %d-%d | Confidence: %.3f\nDynamic Range: %ddB', ...
%         i, filename, startIdx, endIdx, FP(i).Confidence, dynamicRange), 'Interpreter', "none")
% 
%     % Wait for user input before showing next detection
%     fprintf('Showing False Positive %d of %d. Press any key to continue...\n', i, length(FP));
%     waitforbuttonpress
% end
% 
% fprintf('Finished displaying all %d false positive detections.\n', length(FP));