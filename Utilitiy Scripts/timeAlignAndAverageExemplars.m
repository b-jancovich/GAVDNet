% createAverageExemplar
clear
close all
clc 

% Directory containing denoised exemplars
% denoisedExemplarsPath = "D:\SORP_BmAntZ_exemplars\Denoised";
denoisedExemplarsPath = 'D:\DGS_Chagos_Exemplars\U1 & U2\Denoised';
% outName = "Bm_Ant_Z_averageExemplar.wav";
outName = "DGS_Chagos_averageExemplar.wav";

fadeDuration = 1; % fade-in/out to apply (s)
bandOfInterest = [20, 40]; % Bandwidth containing the signal of interest (Hz)
targetCallType = 'AM'; % 'tonal' for simple tonal calls, 'pulsed' for noisy, combed AM type calls
VPO = 48; % Voices per octave in wavelet transform filterbank (used in time alignment)

%% Add Paths

here = pwd;
detectorRoot = fileparts(here);
gitPath = here(1:strfind(here,'Git')+2);
addpath(fullfile(detectorRoot, 'Functions'));
addpath(fullfile(gitPath, 'Utilities'));

%% Init

% Set time bandwidth product based on call type.
if strcmp(targetCallType, 'tonal') == 1
    TBP = 70;
else
    TBP = 50;
end

%% Read Files

% Get list of exemplar files
filelist = dir(fullfile(denoisedExemplarsPath, '*.wav'));

% Read Audio Files
nExemplars = length(filelist);
for i = 1:nExemplars
    % Read in audio
    [filelist(i).audio, filelist(i).fs] = audioread(...
        fullfile(denoisedExemplarsPath, filelist(i).name));

    % Normalize audio files
    filelist(i).audio = filelist(i).audio ./ max(abs(filelist(i).audio));
    
    % Get each file's SNR ranks from the filenames
    filelist(i).snrRank = extractBetween(filelist(i).name, '_rank', '_');

    % Convert the ranks from strings to doubles
    filelist(i).snrRank = str2double(filelist(i).snrRank);
end
% 
% % Sort the list by SNR rank, descending
filelist = struct2table(filelist);
% filelist = sortrows(filelist, "snrRank");

%% Preprocessing

% Resample all exemplars to lowest fs in the list
fsOutput = min([filelist.fs]);
for i = 1:nExemplars
    if filelist.fs(i) ~= fsOutput
        % Design a resampler object
        src = designArbitraryAudioResampler(filelist.fs(i), fsOutput);

        % Resample the sigal
        filelist.audio{i} = customAudioResampler(filelist.audio{i}, src);
        
        % Update Fs
        filelist.fs(i) = fsOutput;
    end
end

% Normalize, mean center, and window audio
for i = 1:nExemplars
    % Extract signal
    x = filelist.audio{i};

    % Build the window function
    fadeLength = round(fadeDuration * fsOutput);
    wind = hann(fadeLength * 2);
    wind = wind(1:fadeLength);
    sigLen = length(x);
    middle = ones(sigLen-2*fadeLength, 1);
    windowFull = [wind; middle; flip(wind)];

    % Apply the window to the signals
    y = x .* windowFull;

    % Normalize the signals
    y = y ./ max(abs(y));

    % Subtract the mean
    y = y - mean(y);

    % Pad the signal with 2 seconds of zeros either side
    padding = zeros(2 * fsOutput, 1);
    filelist.audio{i} = [padding; y; padding];
end

%% Time alignment

% Use first exemplar as reference
reference = filelist.audio{1};
referenceLen = length(reference);

% Store aligned signals
alignedAudio = cell(nExemplars, 1);
alignedAudio{1} = reference;

% Compute reference CWT
refWT = cwt(reference, 'morse', fsOutput, 'FrequencyLimits', bandOfInterest, ...
    'VoicesPerOctave', VPO, 'TimeBandwidth', TBP);

for i = 2:nExemplars
    % Get current signal
    signal = filelist.audio{i};
    
    % Compute signal CWT
    sigWT = cwt(signal, 'morse', fsOutput, 'FrequencyLimits', bandOfInterest, ...
        'VoicesPerOctave', VPO, 'TimeBandwidth', TBP);
    
    % 2D cross-correlation of wavelet transforms
    c = xcorr2(abs(refWT), abs(sigWT));
    
    % Find peak correlation
    [~, maxIdx] = max(c, [], 'all');
    [~, col] = ind2sub(size(c), maxIdx);
    
    % Convert to sample lag
    timeLag = col - size(sigWT,2);
    
    % Align signal based on computed lag
    if timeLag > 0
        aligned = [zeros(timeLag,1); signal(1:min(end,referenceLen-timeLag))];
    else
        aligned = [signal(-timeLag+1:min(end,referenceLen-timeLag)); zeros(-timeLag,1)];
    end
    
    % Match reference length
    if length(aligned) > referenceLen
        aligned = aligned(1:referenceLen);
    elseif length(aligned) < referenceLen
        aligned = [aligned; zeros(referenceLen-length(aligned),1)];
    end
    
    
    alignedAudio{i} = aligned;
end

% Update the table
filelist.alignedAudio = alignedAudio;

%% Final Average Exemplar

% Convert cell array to matrix (each column is a signal)
signalMatrix = cell2mat(filelist.alignedAudio');

% Calculate mean signal
meanSignal = mean(signalMatrix, 2);

% Build the window function
fadeLength = round(fadeDuration * fsOutput);
wind = hann(fadeLength * 2);
wind = wind(1:fadeLength);
sigLen = length(meanSignal);
middle = ones(sigLen-2*fadeLength, 1);
windowFull = [wind; middle; flip(wind)];

% Apply the window to the signals
meanSignal = meanSignal .* windowFull;

% Normalize the signal
meanSignal = meanSignal ./ max(abs(meanSignal));

% Save the mean signal
filename = fullfile(denoisedExemplarsPath, outName);
audiowrite(filename, meanSignal, fsOutput);

fprintf('Saved mean signal exemplar to %s\n', denoisedExemplarsPath)

%% Plotting

% Plot all the original signals as waveforms
figure(1)
tiledlayout(nExemplars, 1)
for i = 1:nExemplars
    nexttile
    plot(filelist.audio{i})
end
sgtitle('Original Signal Waveforms')

% Plot all the original signals as spectrograms
figure(2)
tiledlayout(nExemplars, 1)
for i = 1:nExemplars  
    nexttile
    spectrogram(filelist.audio{i}, 125, 120, 4096,...
        filelist.fs(i), 'yaxis')
end
sgtitle('Original Signal Spectrograms')

% Plot all the time aligned signals as waveforms
figure(3)
tiledlayout(nExemplars+1, 1)
for i = 1:nExemplars
    if i == 1
        nexttile
        plot(reference)
    else
        nexttile
        plot(filelist.alignedAudio{i})
    end
end
nexttile
plot(meanSignal)
title("Mean Signal")
sgtitle('Time-Aligned Waveforms')

% Plot all the time aligned signals as spectrograms
figure(4)
tiledlayout(nExemplars+1, 1)
for i = 1:nExemplars
    if i == 1
        nexttile
        spectrogram(reference, 125, 120, 4096, fsOutput, 'yaxis')
    else        
        nexttile
        spectrogram(filelist.alignedAudio{i}, 125, 120, 4096, fsOutput, 'yaxis')
    end
end
nexttile
spectrogram(meanSignal, 125, 120, 4096, fsOutput, 'yaxis')
title("Mean Signal")
sgtitle('Time-Aligned Spectrograms')
