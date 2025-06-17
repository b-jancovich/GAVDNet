clear all
clc

% configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_SORP_BmAntZ.m";
% sequence_Path = 'C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\BmAntZ_SORP\sequences';
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";
sequence_Path = "D:\GAVDNet\Chagos_DGS\Training & Models\sequences";
sequenceFs = 250 ; 

X_T_Path = "D:\GAVDNet\Chagos_DGS\Training & Models\trainXandT";
% X_T_Path = 'C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\BmAntZ_SORP\trainXandT';

% Path to some real recordings to compare with:spectPCENized
audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\H08S1_150605-120000_calls+extremelyHighPowerNoise.wav";
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-000000_EarthquakeDynamicRangeTest.wav"; 
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-160000_HighDynamicRangeCalls.wav";
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-200000_TypicalLotsOfCalls_TrimmedTo2.08.39.wav";
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\Chagos_whale_song_DGS_071102.wav";
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\BmAnt_ZCall_Casey_2014-03-30_04-00-00.wav";

% Clean signals path
cleanSigPath = "D:\GAVDNet\Chagos_DGS\Training & Models\clean_signals";

run(configPath) % Load config file
projectRoot = pwd;
[gitRoot, ~, ~] = fileparts(projectRoot);
addpath(fullfile(projectRoot, "Functions"))
addpath(fullfile(gitRoot, "Utilities"))

%% Load and analyse model

% % Handle multiple model files with a UI dialog:
% modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
% if isscalar(modelList)
%     load(fullfile(modelList.folder, modelList.name))
%     fprintf('Loading model: %s\n', modelList.name)
% else
%     [file, location] = uigetfile(gavdNetDataPath, 'Select a model to load:');
%     load(fullfile(location, file))
% end
% 
% % Get LT & LT scaler post proc parameters
% maxDetectionDuration = model.dataSynthesisParams.maxTargetCallDuration;
% postProcOptions.LT = model.dataSynthesisParams.minTargetCallDuration .* ...
%     postProcOptions.LT_scaler;

% % Draw model training history
% figure(1)
% plot(model.trainInfo.TrainingHistory.Iteration, model.trainInfo.TrainingHistory.Loss)
% yscale('log')
% hold on
% plot(model.trainInfo.ValidationHistory.Iteration, model.trainInfo.ValidationHistory.Loss)
% ylabel('Binary Cross Entropy')
% xlabel('Iteration')

%% Show a real recording for comparison
% 
% % Read audio
% [realAudio, realAudioFs] = audioread(audioPath);
% 
% % Build time vector for audio
% realAudioDuration = length(realAudio) / realAudioFs;
% dtRealAudioFs = 1/realAudioFs;
% realAudioTimeVec = 0:dtRealAudioFs:realAudioDuration-dtRealAudioFs;
% 
% % Set window & hop
% windowLen = round(windowDur * fsTarget);
% hopLen = round(hopDur * fsTarget);
% 
% % Preprocess data
% meanTargetCallDuration = 15;
% realAudioFeatures = gavdNetPreprocess(realAudio, realAudioFs, fsTarget, ...
%     bandwidth, windowLen, hopLen, saturationRange);
% 
% % Build time vector for features
% featuresTimeVec = linspace(0, realAudioDuration, size(realAudioFeatures, 2));
% fvec = linspace(bandwidth(1), bandwidth(2), 40);
% 
% % Draw plot
% figure(2)
% imagesc(featuresTimeVec, fvec, realAudioFeatures)
% xlabel('Time (seconds)')
% ylabel('Frequency (Hz)')
% set(gca, 'YDir', 'normal')
% colorbar
% title('Real recording spectrogram')

%% Draw some clean signals

% cleanSigList= dir(fullfile(cleanSigPath, '*.wav'));
% FFTLen = 4096;
% [filterBank, Fc, ~] = designAuditoryFilterBank(fsTarget, ...
%     FFTLength = FFTLen, ...
%     Normalization = "none", ...
%     OneSided = true, ...
%     FrequencyRange = bandwidth, ...
%     FilterBankDesignDomain = "warped", ...
%     FrequencyScale = "mel", ...
%     NumBands = 40);
% 
% % Apply filter bank
% figure(3)
% for i = 1:100
%     [audio, fs] = audioread(fullfile(cleanSigList(i).folder, cleanSigList(i).name));
%     audio = audio ./ max(abs(audio));
%     [p, q] = rat(fsTarget/fs, 1e-9);
%     audio = resample(audio(:), p, q);
%     nOverlap = windowLen - hopLen;
%     [s, f, t] = spectrogram(audio, windowLen, nOverlap, FFTLen, fsTarget, 'yaxis');
%     s = abs(s).^2;
%     s = filterBank * s;
%     S = 10*log10(max(s, 1e-10));
%     imagesc(t, Fc, s)
%     set(gca, 'YDir', 'normal')
%     ylabel('Frequency (Hz)')
%     xlabel('Bin Center Time Index (s)')
%     colorbar
%     waitforbuttonpress
% end

%% Draw Sequences

% sequence_FileList = dir(fullfile(sequence_Path, '*mat'));
% targetCallDuration = 15;
% interactive_plot_viewer(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur, saturationRange)

%% Draw 'X' (Frames) & 'T' (Targets)

X_T_FileList = dir(fullfile(X_T_Path, '*.mat'));

for i = 1:200
    figure(3)
    load(fullfile(X_T_FileList(i).folder, X_T_FileList(i).name));
    numWin = size(X, 2);
    timeVector = (windowDur/2) + hopDur * (0:numWin-1);
    freqVector = logspace2(bandwidth(1), bandwidth(2), 40);

    yyaxis left
    imagesc(timeVector, freqVector, X)
    set(gca, 'YDir', 'normal')
    ylabel('Frequency (Hz)')
    xlabel('Bin Center Time Index (s)')
    yyaxis right
    plot(timeVector, T, 'r--')
    ylabel('Probability of Song')
    ylim([-0.01, 1.01])
    waitforbuttonpress
end


%% Helper function

function interactive_plot_viewer(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur, saturationRange)
    % Create figure
    figure(3);
    clf; % Clear the figure
    
    % Initialize variables
    currentIndex = 1;
    maxIndex = 100;
    
    % Create the UI button
    btn = uicontrol('Style', 'pushbutton', ...
                    'String', sprintf('Next (%d/%d)', currentIndex, maxIndex), ...
                    'Position', [20, 20, 120, 30], ...
                    'Callback', @advanceLoop);
    
    % Create a text display for current file info
    fileInfoText = uicontrol('Style', 'text', ...
                            'Position', [150, 20, 300, 30], ...
                            'HorizontalAlignment', 'left', ...
                            'BackgroundColor', get(gcf, 'Color'));
    
    % Initial plot
    updatePlot(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur, saturationRange);
    
    function advanceLoop(~, ~)
        if currentIndex < maxIndex
            currentIndex = currentIndex + 1;
            updatePlot(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur, saturationRange);
            set(btn, 'String', sprintf('Next (%d/%d)', currentIndex, maxIndex));
        else
            set(btn, 'String', 'Finished!', 'Enable', 'off');
        end
    end
    
    function updatePlot(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur, saturationRange)
        % Load data for current index
        loaded = load(fullfile(sequence_FileList(currentIndex).folder, sequence_FileList(currentIndex).name));
        audioSequence = loaded.audioSequence;
        mask = loaded.mask;
        snr = loaded.sequenceSNRs;

        windowLen = round(windowDur * fsTarget);
        hopLen = round(hopDur * fsTarget);

        % Process data
        [Xfull, Tfull] = gavdNetPreprocess(audioSequence, sequenceFs, fsTarget, ...
            bandwidth, windowLen, hopLen, saturationRange, mask);
        numWin = size(Xfull, 2);
        timeVector = (windowDur/2) + hopDur * (0:numWin-1);
        freqVector = logspace2(bandwidth(1), bandwidth(2), 40);
        
        % Clear previous plot
        cla;
        
        % Plot spectrogram
        yyaxis left
        imagesc(timeVector, freqVector, Xfull)
        set(gca, 'YDir', 'normal')
        ylabel('Frequency (Hz)')
        xlabel('Bin Center Time Index (s)')
        colorbar
        
        % Plot probability
        yyaxis right
        plot(timeVector, Tfull, 'r--', 'LineWidth', 2)
        ylabel('Probability of Song')
        ylim([-0.01, 1.01])
        
        % Update title and file info
        title(sprintf('File %d/%d: %s - SNR = %.2f', currentIndex, maxIndex, sequence_FileList(currentIndex).name, snr), ...
              'Interpreter', 'none')
        set(fileInfoText, 'String', sprintf('Current file: %s', sequence_FileList(currentIndex).name));
        
        % Refresh the plot
        drawnow;
    end
end