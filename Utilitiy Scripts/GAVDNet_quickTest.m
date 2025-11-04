% This script is a quick test of the trained detector using a single short
% recording that has 5 high SNR calls in it. Use this as a basis for your
% inference script.

clear all
close all
clc

%% Settings

featureFraming = 'event-split'; % 'simple' or 'event-split' or 'none'; 

%% File paths

% Model File Path
modelPath = "D:\GAVDNet\Chagos_DGS\Training & Models\GAVDNet_trained_17-Jun-2025_23-50.mat";

% Config file Path
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";
% configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_SORP_BmAntZ.m";

% Test Audio File Path - File 1: target call
% audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\H08S1_071102-000000_EarthquakeDynamicRangeTest.wav"; 
% audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\H08S1_071102-160000_HighDynamicRangeCalls.wav";
audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\H08S1_150605-120000_calls+extremelyHighPowerNoise.wav";
% audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\H08S1_150605-120000_calls+extremelyHighPowerNoise_TrimmedTo2.08.20.wav";     % 63 calls manually detected 
% audioPath = "D:\GAVDNet\Chagos_DGS\Test Data\H08S1_071102-200000_TypicalLotsOfCalls_TrimmedTo2.08.39.wav";                % 199 calls manually detected
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\Chagos_whale_song_DGS_071102.wav";
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\BmAnt_ZCall_Casey_2014-03-30_04-00-00.wav";

% Audio Path - File 2: non-target call
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\NewZealand_BW_L7910NZ01_002K_Site04_multi_20160225_051500Z_15min.wav";

%% Load Config, Model & audio

% Add paths
addpath('C:\Users\z5439673\Git\GAVDNet\Functions');
load(modelPath)
run(configPath)
[audio, fs] = audioread(audioPath);

% Use the "minimum call duration" parameter from the training data synthesis 
% information, stored in the model metadata as the length threshold for 
% post-processing. You can try using smaller values for this if parts of 
% your target call are frequently missing due to propagation effects.
postProcOptions.LT = model.dataSynthesisParams.minTargetCallDuration * postProcOptions.LT_scaler;

[useGPU, deviceID, ~, bytesAvailable] = gpuConfig();

%% Run Model

% Run Preprocessing and inference 
[probabilities, features, execTime] = gavdNetInference(audio, fs, model, ...
    bytesAvailable, featureFraming);

%% Run postprocessing to determine decision boundaries. 

fprintf('Postprocesing model outputs...\n')
gavdNetPostprocess(audio, fs, probabilities, model.preprocParams, postProcOptions, features);
 
% Get regions of interest
roi = gavdNetPostprocess(audio, fs, probabilities, model.preprocParams, postProcOptions);