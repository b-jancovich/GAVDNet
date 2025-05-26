%% GAVDNet Compare Detector Results with GroundTruth

%% Init

clear
close all
clc
clear persistent

%% **** USER INPUT ****

% Path to the config file:
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";

inferenceResultsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Results\detector_results_postprocessed.mat";

% Path to "groundtruth" file containing date and time stamps of the true 
% detections of the target call in the test audio files:
groundtruthPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\TestSubset\test_dataset_detection_list.mat";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.
%% Set Paths and Load Input Variables

% Add dependencies to path
run(configPath) % Load config file
projectRoot = pwd;
[gitRoot, ~, ~] = fileparts(projectRoot);
addpath(fullfile(projectRoot, "Functions"))

%% Compare results

metrics = compareDetectionsToSubsampledTestDataset(groundtruthPath, inferenceResultsPath, detectionTolerance);
