clear all
close all
clc

% audioFolderPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\TestSubset";
audioFolderPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Detector Test Datasets\AAD_AcousticTrends_BlueFinLibrary\DATA\casey2014\wav";

% Number of files to analyse
fileSampleSize = 50;

% Do not plot results
plotResults = true;

% Spectrogram parameters
fsTarget = 250;
bandwidth = [10, 50];
windowLen = 212;
hopLen = 12;

[datasetStats, fileStats] = measureDatasetDynamicRange(audioFolderPath, fileSampleSize, ...
    fsTarget, bandwidth, windowLen, hopLen, plotResults);