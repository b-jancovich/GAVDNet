% This script is a quick test of the trained detector using a single short
% recording that has 5 high SNR calls in it. Use this as a basis for your
% inference script.

clear all
close all
clc

%% Run test

% Add paths
addpath('C:\Users\z5439673\Git\GAVDNet\Functions');

% Load model
load("C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\chagos_DGS_2025\GAVDNet_trained_28-Apr-2025_18-11.mat")

% Load config
run("C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m")

% Audio Path - File 1: target call
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\Chagos_whale_song_DGS_071102.wav";

% Audio Path - File 2: non-target call
audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\NewZealand_BW_L7910NZ01_002K_Site04_multi_20160225_051500Z_15min.wav";

% Load audio
[audio, fs] = audioread(audioPath);

% Use the "minimum call duration" parameter from the training data synthesis 
% information, stored in the model metadata as the length threshold for 
% post-processing. You can try using smaller values for this if parts of 
% your target call are frequently missing due to propagation effects.
postProcOptions.LT = model.dataSynthesisParams.minTargetCallDuration;

% Run Preprocessing & Feature Extraction on audio
fprintf('Preprocesing audio & extracting features...\n')
[features, ~] = gavdNetPreprocess(...
    audio, ...
    fs, ...
    model.preprocParams.fsTarget, ...
    model.preprocParams.bandwidth, ...
    model.preprocParams.windowLen,...
    model.preprocParams.hopLen);

% Run Model in minibatch mode to save memory
fprintf('Running model...\n')
y = minibatchpredict(model.net, gpuArray(features));

% Run postprocessing to determine decision boundaries. 
fprintf('Postprocesing model outputs...\n')
gavdNetPostprocess(audio, fs, y, model.preprocParams, postProcOptions, features);
 
% Get regions of interest
roi = gavdNetPostprocess(audio, fs, y, model.preprocParams, postProcOptions);
