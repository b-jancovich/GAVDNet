clear 
clc

configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";

% audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\H08S1_071102-200000_TypicalLotsOfCalls_TrimmedTo2.08.39.wav";
% This file has some high energy low frequency, short duration
% peaks, but is mainly pretty uniform. With no framing, global or 
% constant are probably best.

audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\H08S1_150605-120000_calls+extremelyHighPowerNoise.wav";
% For this file, with no framing, robust was definitely not good. 
% Global was probably best.

%% Test 1 - no framing/event splitting

% Add paths
addpath('C:\Users\z5439673\Git\GAVDNet\Functions');
run(configPath)
[audio, fs] = audioread(audioPath);

% Init STFT parameters (ensuring even values)
windowLen = 2 * round((windowDur * fsTarget) / 2);
hopLen = 2 * round((hopDur * fsTarget) / 2);

% Measure audio dynamic range
fileStats = measureFileDynamicRange(audio, fs, fsTarget, bandwidth, windowLen, hopLen);
saturationRanges = [fileStats.robustDynamicRange, fileStats.globalDynamicRange, 70];
metricNames = ["measured, robust", "measured, global", "constant"];


for i = 1:length(saturationRanges)
    features{i} = gavdNetPreprocess(audio, fs, fsTarget, bandwidth, windowLen, hopLen, saturationRanges(i));
end
figure(1)
tiledlayout(length(saturationRanges), 1)
for i = 1:length(saturationRanges)
    nexttile
    imagesc(features{i})
    set(gca, 'ydir', 'normal')
    title(sprintf('Saturation Range Setting: %.2f (%s)', saturationRanges(i), metricNames(i)), Interpreter="none")
    colorbar
    xlim([1e4, 1.5e4])
end

clearvars -except audio fs fsTarget windowLen hopLen bandwidth 

%% Test 2 - Event split 

% Split into events
maxTargetCallDuration = 15;
smoothingWindowDuration = maxTargetCallDuration * 4;
eventOverlapDuration = maxTargetCallDuration;
[audioSegments, splitIndices, ~] = eventSplitter(audio, fs, ...
    smoothingWindowDuration, eventOverlapDuration);

featuresSegments = cell(size(audioSegments));
for i = 1:length(audioSegments)   
    % Normalize segments
    audioSegments{i} = audioSegments{i} ./ max(abs(audioSegments{i}));

    % measure segment dynamic range
    fileStats = measureFileDynamicRange(audioSegments{i}, fs, fsTarget, bandwidth, windowLen, hopLen);
    saturationRanges(i) = fileStats.globalDynamicRange;

    % Run Preprocessing & Feature Extraction on audio
    featuresSegments{i} = gavdNetPreprocess(audioSegments{i}, fs, fsTarget, ...
        bandwidth, windowLen, hopLen, saturationRanges(i));

    % Make a dummy probs to hand to the stitcher
    probs{i} = zeros(1, size(featuresSegments{i}, 2));
end

% stitch features back together
[~, features] = segmentStitcher(probs, splitIndices, hopLen, featuresSegments);
figure(2)
imagesc(features)
set(gca, 'ydir', 'normal')
title(sprintf('Saturation Range Settings: %.2f (measured, global)', saturationRanges(i)), Interpreter="none")
colorbar
xlim([1e4, 1.5e4])
