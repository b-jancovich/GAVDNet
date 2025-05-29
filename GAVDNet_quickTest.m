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
% modelPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\chagos_DGS_2025\GAVDNet_trained_27-May-2025_18-06.mat";
modelPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\BmAntZ_SORP\GAVDNet_trained_28-May-2025_23-29.mat";

% config Path
% configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_SORP_BmAntZ.m";

% Audio Path - File 1: target call
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\Chagos_whale_song_DGS_071102.wav";
audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\BmAnt_ZCall_Casey_2014-03-30_04-00-00.wav";

% Audio Path - File 2: non-target call
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\NewZealand_BW_L7910NZ01_002K_Site04_multi_20160225_051500Z_15min.wav";

% Load Config, Model & audio
load(modelPath)
run(configPath)
[audio, fs] = audioread(audioPath);

% Use the "minimum call duration" parameter from the training data synthesis 
% information, stored in the model metadata as the length threshold for 
% post-processing. You can try using smaller values for this if parts of 
% your target call are frequently missing due to propagation effects.
postProcOptions.LT = model.dataSynthesisParams.minTargetCallDuration * postProcOptions.LT_scaler;

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
