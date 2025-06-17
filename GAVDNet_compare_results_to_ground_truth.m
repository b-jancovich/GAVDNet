%% GAVDNet Compare Detector Results with GroundTruth

%% Init

clear
close all
clc
clear persistent

%% **** USER INPUT ****

% Path to the config file:
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";
% configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_SORP_BmAntZ.m";

% Path to inference output file:
inferenceResultsPath = "D:\GAVDNet\Chagos_DGS\Test Results\detector_results_postprocessed.mat";
% inferenceResultsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\BmAntZ_SORP\Results\detector_results_postprocessed.mat";

% Path to "groundtruth" file containing date and time stamps of the true 
% detections of the target call in the test audio files:
groundtruthPath = "D:\GAVDNet\Chagos_DGS\Test Data\2007subset_small\test_dataset_detection_list.mat";
% groundtruthPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\2007subset\test_dataset_detection_list.mat";
% groundtruthPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\TestSubset\test_dataset_detection_list.mat";
% groundtruthPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\BmAntZ_SORP\TestSubset\Casey2014.Bm.Ant-Z.selections_SUBSET.txt";
% groundtruthPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Detector Test Datasets\AAD_AcousticTrends_BlueFinLibrary\DATA\casey2014\Casey2014.Bm.Ant-Z.selections.txt";

% Results path for comparison of detector output with groundtruth
gtCompareResultsPath = "D:\GAVDNet\Chagos_DGS\Test Results\groundtruthComparisonResults.xlsx";
% gtCompareResultsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\BmAntZ_SORP\Results\groundtruthComparisonResults.xlsx";

% Test dataset source
gtFormat = 'CTBTO'; % Either "CTBTO" or "SORP"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.
%% Set Paths and Load Input Variables

% Add dependencies to path
run(configPath) % Load config file
projectRoot = pwd;
[gitRoot, ~, ~] = fileparts(projectRoot);
addpath(fullfile(projectRoot, "Functions"))

%% Load model

% Handle multiple model files with a UI dialog:
modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
if isscalar(modelList)
    load(fullfile(modelList.folder, modelList.name))
    fprintf('Loading model: %s\n', modelList.name)
    [~, modelName, ~] = fileparts(fullfile(modelList.folder, modelList.name));

else
    [file, location] = uigetfile(gavdNetDataPath, 'Select a model to load:');
    load(fullfile(location, file))
    [~, modelName, ~] = fileparts(fullfile(location, file));
end

% Get LT & LT scaler post proc parameters
maxDetectionDuration = model.dataSynthesisParams.maxTargetCallDuration*1.5;
postProcOptions.LT = model.dataSynthesisParams.minTargetCallDuration .* ...
    postProcOptions.LT_scaler;

%% Compare Detector Output to Groundtruth

% Run groundtruth comparison
[metrics, FP, FN] = compareDetectionsToSubsampledTestDataset(...
    groundtruthPath, inferenceResultsPath, detectionTolerance, maxDetectionDuration, gtFormat);

results = load(inferenceResultsPath);
inferenceMode = results.inferenceMode;

%% Save Results

% Compile results and test params
testCompleteTime = string(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
[~, dataSetName, ~] = fileparts(fileparts(groundtruthPath));

outTable = struct2table(metrics);
outTable = removevars(outTable, {'roc', 'performanceCurve', ...
    'evaluatedResultCount', 'numResultsExcluded_NoScoreOrTime',...
    'groundtruthSource', 'numResultsExcluded_InferenceFailures',...
    'matchingAlgorithm', 'totalAudioDuration_sec'});
newNames = {'ActivationThreshold', 'DeactivationThreshold', 'AEAVD', ...
    'MergeThreshold', 'LengthThresholdScaler', 'LengthThreshold', ...
    'TestTimeStamp', 'TestDataset', 'ModelName', 'SequenceMode', 'SequenceSNRRange',...
    'InferenceMode'};
outTable = addvars(outTable, postProcOptions.AT, postProcOptions.DT, ...
    postProcOptions.AEAVD, postProcOptions.MT, postProcOptions.LT_scaler, ...
    postProcOptions.LT, testCompleteTime, string(modelName), dataSetName,...
    string(model.dataSynthesisParams.sequenceMode), model.dataSynthesisParams.snrRange,...
    string(inferenceMode), 'NewVariableNames', newNames);

% Write output to CSV
if exist(gtCompareResultsPath, 'file') == 2
    % Append data without headers
    writetable(outTable, gtCompareResultsPath, 'WriteMode', 'append', 'WriteVariableNames', false);
else
    % File does not exist — write with headers
    writetable(outTable, gtCompareResultsPath);
end

% Save disagreements
disagreements = struct('falsePositives', FP, 'falseNegatives', FN);
[resultsFolder, ~, ~] = fileparts(gtCompareResultsPath);
saveNamePath = fullfile(resultsFolder,...
    strcat('detector_vs_GT_disagreements_', testCompleteTime, '.mat'));
save(saveNamePath, 'disagreements', '-v7.3')