clear all
close all
clc

configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_SORP_BmAntZ.m";
sequence_Path = 'C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\BmAntZ_SORP\sequences';
sequenceFs = 250 ; 

run(configPath) % Load config file
projectRoot = pwd;
[gitRoot, ~, ~] = fileparts(projectRoot);
addpath(fullfile(projectRoot, "Functions"))
addpath(fullfile(gitRoot, "Utilities"))

% Handle multiple model files with a UI dialog:
modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
if isscalar(modelList)
    load(fullfile(modelList.folder, modelList.name))
    fprintf('Loading model: %s\n', modelList.name)
else
    [file, location] = uigetfile(gavdNetDataPath, 'Select a model to load:');
    load(fullfile(location, file))
end

% Get LT & LT scaler post proc parameters
maxDetectionDuration = model.dataSynthesisParams.maxTargetCallDuration;
postProcOptions.LT = model.dataSynthesisParams.minTargetCallDuration .* ...
    postProcOptions.LT_scaler;

% %% Draw model training history
% 
% figure(1)
% plot(model.trainInfo.TrainingHistory.Iteration, model.trainInfo.TrainingHistory.Loss)
% yscale('log')
% hold on
% plot(model.trainInfo.ValidationHistory.Iteration, model.trainInfo.ValidationHistory.Loss)
% ylabel('Binary Cross Entropy')
% xlabel('Iteration')

%% Draw Sequences

sequence_FileList = dir(fullfile(sequence_Path, '*mat'));

figure(2)
interactive_plot_viewer(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur)

% %% Draw 'X' (Frames) & 'T' (Targets)
% X_T_Path = 'C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\BmAntZ_SORP\trainXandT';
% X_T_FileList = dir(fullfile(X_T_Path, '*mat'));
% 
% for i = 1:100
%     figure(3)
%     tiledlayout(1,5)
%     kk = 0;
%     for j = 1:5
%         load(fullfile(X_T_FileList(i+kk).folder, X_T_FileList(i+kk).name));
% 
%         numWin = size(X, 2);
%         timeVector = (windowDur/2) + hopDur * (0:numWin-1);
%         freqVector = logspace2(bandwidth(1), bandwidth(2), 40);
% 
%         nexttile
%         yyaxis left
%         imagesc(timeVector, freqVector, X)
%         set(gca, 'YDir', 'normal')
%         ylabel('Frequency (Hz)')
%         xlabel('Bin Center Time Index (s)')
%         yyaxis right
%         plot(timeVector, T, 'r--')
%         ylabel('Probability of Song')
%         ylim([-0.01, 1.01])
% 
%         kk = kk+1;
%     end
%     waitforbuttonpress
% end


%% Helper function
function interactive_plot_viewer(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur)
    % Create figure
    figure(2);
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
    updatePlot(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur);
    
    function advanceLoop(~, ~)
        if currentIndex < maxIndex
            currentIndex = currentIndex + 1;
            updatePlot(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur);
            set(btn, 'String', sprintf('Next (%d/%d)', currentIndex, maxIndex));
        else
            set(btn, 'String', 'Finished!', 'Enable', 'off');
        end
    end
    
    function updatePlot(sequence_FileList, sequenceFs, fsTarget, bandwidth, windowDur, hopDur)
        % Load data for current index
        loaded = load(fullfile(sequence_FileList(currentIndex).folder, sequence_FileList(currentIndex).name));
        audioSequence = loaded.audioSequence;
        mask = loaded.mask;

        windowLen = round(windowDur * fsTarget);
        hopLen = round(hopDur * fsTarget);

        % Process data
        [Xfull, Tfull] = gavdNetPreprocess(audioSequence, sequenceFs, fsTarget, bandwidth, windowLen, hopLen, mask);
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
        
        % Plot probability
        yyaxis right
        plot(timeVector, Tfull, 'r--', 'LineWidth', 2)
        ylabel('Probability of Song')
        ylim([-0.01, 1.01])
        
        % Update title and file info
        title(sprintf('File %d/%d: %s', currentIndex, maxIndex, sequence_FileList(currentIndex).name), ...
              'Interpreter', 'none')
        set(fileInfoText, 'String', sprintf('Current file: %s', sequence_FileList(currentIndex).name));
        
        % Refresh the plot
        drawnow;
    end
end