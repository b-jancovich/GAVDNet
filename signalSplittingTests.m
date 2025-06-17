% Test the signal splitting based on extreme event detection

clear  
close all
clc

testNum = 1; 

% File which contains calls + extremely high-power transient noise
audioPath{1} = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_150605-120000_calls+extremelyHighPowerNoise.wav";
comment{1} = 'This file contains both target calls and relatively short duration extreme events.';
extremeGT{1} = true;

% File which contains high dynamic range calls, but no extreme events 
audioPath{2} = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-160000_HighDynamicRangeCalls.wav";
comment{2} = 'This file contains high dynamic range target calls, but no extreme events.';
extremeGT{2} = false;

% An unremarkable file, with no extreme events
audioPath{3} = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-200000_TypicalLotsOfCalls.wav";
comment{3} = 'This file is unremarkable. It contains target calls, and no extreme events.';
extremeGT{3} = false;

% File with an extreme, very long duration event (earthquake or iceberg calving) - also contains calls
audioPath{4} = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-000000_EarthquakeDynamicRangeTest.wav";
comment{4} = 'This file contains target calls, and variety of short and long duration extreme events.';
extremeGT{4} = true;

% Add functions to path
addpath('C:\Users\z5439673\Git\GAVDNet\Functions')

% Preproc Parameters
bandwidth = [10, 50];
windowDur = 1.5;
hopDur = 0.05;
saturationRange = 70;
targetCallDuration = 10;

% Event splitting parametrs
maxTargetCallDuration = 30;
smoothingWindowDuration = maxTargetCallDuration * 4;
eventOverlapDuration = maxTargetCallDuration;

% Read Audio
[audio, fs] = audioread(audioPath{testNum});
% audio = gpuArray(audio);

% Convert durations to sample counts
windowLen = 2 * round((windowDur * fs) / 2);
hopLen = 2 * round((hopDur * fs) / 2);

% normaliza audio
audio = audio ./ max(abs(audio));

% Look for extreme events, and if found, split audio into segments based on
% envelope changepoints
[audioSegments, splitIndices, changepts] = eventSplitter(audio, fs, ...
    smoothingWindowDuration, eventOverlapDuration);

% Run preprocessor on original audio signal
featuresRaw = gavdNetPreprocess(...
    audio, fs, fs, bandwidth, windowLen, hopLen, ...
    saturationRange);

% Run preprocessor on segments
featuresSegments = cell(size(audioSegments));
for i = 1:length(audioSegments)
    featuresSegments{i} = gavdNetPreprocess(...
        audioSegments{i}, fs, fs, bandwidth, windowLen, hopLen, ...
        saturationRange);
end

% Draw envelope & cutpoints
dt = 1/fs;
dur = length(audio) / fs;
t = 0:dt:dur-dt;
changeTimes = changepts / fs;

figure(1)
tiledlayout(2,1)
nexttile
plot(t, audio)
hold on
if ~isempty(changeTimes)
xline(changeTimes, 'r')
end
xlim([t(1), t(end)])
title('Original Signal Waveform + Change Points')
nexttile
imagesc(featuresRaw)
colorbar
set(gca, 'ydir', 'normal')
title('Original Signal Spectrogram')

figure(2)
tiledlayout(2, length(audioSegments))
for i = 1:numel(featuresSegments)
    nexttile
    plot(audioSegments{i})
    title(sprintf('Audio Waveform Segment %d', i))

    nexttile
    imagesc(featuresSegments{i})
    set(gca, 'ydir', 'normal')
    colorbar
    title(sprintf('Audio Spectrogram Segment %d', i))
end
