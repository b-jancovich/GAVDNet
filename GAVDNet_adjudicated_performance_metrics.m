%% GAVDNet Adjudicated Performance Metrics
%
% This script calculates detector performance metrics using adjudicated 
% disagreements from the GAVDNetAdjudicator app. It applies 3 different 
% decision logics for interpreting the 4-choice analyst decisions 
% (DiscreteCallsPresent, ChorusPresent, DiscreteCallsChorusPresent, 
% CallChorusAbsent), and produces performance metrics and figures identical 
% in format to the pre-adjudication analysis.
%
% The script processes adjudicated disagreements where an analyst has reviewed
% each detector-groundtruth disagreement and classified the acoustic content.
% Different decision logics represent different interpretations of what 
% constitutes a "true positive" detection.
%
% Decision Logics:
%   1. Inclusive: All vocal activity (discrete, chorus, or both) = TP
%   2. Discrete-only: Discrete calls (with or without chorus) = TP
%   3. Strict-discrete: Only pure discrete calls (no chorus) = TP
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%% Init

clear
close all
clc
clear persistent

%% **** USER INPUT ****

% % CHAGOS CONFIGURATION
% adjudicatedDisagreementsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Adjudication\adjudication_results_G.Truong\B.musc.brev_ChagosSong_DiegoGarcia_2007_ADJUDICATED\detector_vs_GT_disagreements_07-Jul-2025_08-57-43.mat";
% inferenceResultsPath = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-10 to 10 Single Exemplar\detector_results_postprocessed.mat";
% gavdNetDataPath = "D:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar";
% groundtruthPath = "D:\GAVDNet\Chagos_DGS\Test Data\2007subset\test_dataset_detection_list.mat";
% gtFormat = 'CTBTO';

% Z-CALL CONFIGURATION
adjudicatedDisagreementsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Adjudication\adjudication_results_G.Truong\B.musc.int_ZCall_CaseyStation_2014_ADJUDICATED\detector_vs_GT_disagreements_07-Jul-2025_10-06-37.mat";
inferenceResultsPath = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-10 to 10\detector_results_postprocessed.mat";
gavdNetDataPath = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-10 to 10";
groundtruthPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Detector Test Datasets\AAD_AcousticTrends_BlueFinLibrary\DATA\casey2014\Casey2014.Bm.Ant-Z.selections.txt";
gtFormat = 'SORP';

% Detection matching parameters (must match settings used to generate the disagreements file)
detectionTolerance = 0.001; % seconds
maxDetectionDuration = 40; % seconds

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.

%% Setup

% Start diary
[outputFolder, ~, ~] = fileparts(adjudicatedDisagreementsPath);
outputFolder = fullfile(outputFolder, 'post_adjudication');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

ts = string(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
logname = sprintf('adjudication_analysis_%s.log', ts);
diary(fullfile(outputFolder, logname));

fprintf('=== GAVDNet Adjudicated Performance Metrics Analysis ===\n');
fprintf('Started: %s\n\n', char(datetime("now")));

% Add Functions to path
projectRoot = pwd;
addpath(fullfile(projectRoot, "Functions"))

%% Load Data

fprintf('Loading adjudicated disagreements from:\n  %s\n', adjudicatedDisagreementsPath);
adjData = load(adjudicatedDisagreementsPath);
disagreements = adjData.disagreements;

fprintf('Loading detector results from:\n  %s\n', inferenceResultsPath);
detectorData = load(inferenceResultsPath);
detectorResults = detectorData.results;
postProcOptions = detectorData.postProcOptions;
featureFraming = detectorData.featureFraming;
frameStandardization = detectorData.frameStandardization;

fprintf('Loading ground truth from:\n  %s\n', groundtruthPath);
[~, dataSetName, ~] = fileparts(fileparts(groundtruthPath));

fprintf('Loading model metadata from:\n  %s\n', gavdNetDataPath);
modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
if isscalar(modelList)
    load(fullfile(modelList.folder, modelList.name), 'model')
    [~, modelName, ~] = fileparts(fullfile(modelList.folder, modelList.name));
else
    error('Expected single model file in %s', gavdNetDataPath);
end

% Get original counts from ground truth
if strcmp(gtFormat, 'CTBTO') == true
    % Load the detection list  
    testDatasetDetectionsList = load(groundtruthPath, "testDatasetDetectionsList");

    % Find the groundtruth file list
    [groundTruthFolder, ~, ~] = fileparts(groundtruthPath);
    groundtruthFolderContents = dir(fullfile(groundTruthFolder, '*.mat'));
    fileListPath = groundtruthFolderContents(contains({groundtruthFolderContents.name}, 'audiofile_list'));

    % Load the file list
    if ~isempty(fileListPath)
        testDatasetFileList = load(fullfile(fileListPath.folder, fileListPath.name), 'testDatasetFileList');
    else
        warning('Could not find the "testDatasetFileList" file. Test dataset audio duration will not be calculated.')
    end

    % Put the GT tables into a new struct together
    gtData = struct('testDatasetDetectionsList', testDatasetDetectionsList.testDatasetDetectionsList,...
        'testDatasetFileList', testDatasetFileList.testDatasetFileList);

    % Remove unused vars
    clearvars testDatasetDetectionsList testDatasetFileList groundTruthFolder groundtruthFolderContents fileListPath

    % Number of positive detections in original groudtruth
    nPositivesGT_original = height(gtData.testDatasetDetectionsList);
    
    % Calculate total audio duration
    if isfield(gtData, 'testDatasetFileList') && ismember('duration', gtData.testDatasetFileList.Properties.VariableNames)
        validDurs = gtData.testDatasetFileList.duration(~isnan(gtData.testDatasetFileList.duration) & gtData.testDatasetFileList.duration > 0);
        totalAudioDurationSeconds = sum(validDurs);
    else
        warning('Could not calculate total audio duration from ground truth file list.');
        totalAudioDurationSeconds = NaN;
    end
elseif strcmp(gtFormat, 'SORP') == true
    gtTable = readtable(groundtruthPath, 'Delimiter', '\t', 'ReadVariableNames', true);
    nPositivesGT_original = height(gtTable);
    
    % Calculate total audio duration from unique files
    uniqueFiles = unique(gtTable.BeginFile);
    totalAudioDurationSeconds = 0;
    [gtPath, ~, ~] = fileparts(groundtruthPath);
    for i = 1:length(uniqueFiles)
        try
            info = audioinfo(fullfile(gtPath, 'wav', uniqueFiles{i}));
            totalAudioDurationSeconds = totalAudioDurationSeconds + info.Duration;
        catch
            warning('Could not read audio file: %s', uniqueFiles{i});
        end
    end
else
    error('Invalid gtFormat: %s', gtFormat);
end

fprintf('\nData Summary:\n');
fprintf('  Ground truth detections: %d\n', nPositivesGT_original);
fprintf('  False positives (pre-adjudication): %d\n', length(disagreements.falsePositives));
fprintf('  False negatives (pre-adjudication): %d\n', length(disagreements.falseNegatives));
fprintf('  Total audio duration: %.2f hours\n\n', totalAudioDurationSeconds/3600);

% Calculate original TP count
nPositivesDetector = length(detectorResults);
nFalsePositives_original = length(disagreements.falsePositives);
nFalseNegatives_original = length(disagreements.falseNegatives);
nTruePositives_original = nPositivesDetector - nFalsePositives_original;

fprintf('Original (pre-adjudication) performance:\n');
fprintf('  True Positives: %d\n', nTruePositives_original);
fprintf('  False Positives: %d\n', nFalsePositives_original);
fprintf('  False Negatives: %d\n', nFalseNegatives_original);
fprintf('  Recall: %.4f\n', nTruePositives_original / nPositivesGT_original);
fprintf('  Precision: %.4f\n\n', nTruePositives_original / nPositivesDetector);

%% Check Adjudication Completeness

fprintf('Checking adjudication completeness...\n');
nUnadjudicated_FP = sum(cellfun(@isempty, {disagreements.falsePositives.analystDecision}));
nUnadjudicated_FN = sum(cellfun(@isempty, {disagreements.falseNegatives.analystDecision}));

if nUnadjudicated_FP > 0 || nUnadjudicated_FN > 0
    warning('Incomplete adjudication detected:');
    if nUnadjudicated_FP > 0
        warning('  %d false positives have no analyst decision', nUnadjudicated_FP);
    end
    if nUnadjudicated_FN > 0
        warning('  %d false negatives have no analyst decision', nUnadjudicated_FN);
    end
    error('All disagreements must be adjudicated before running this analysis.');
end

fprintf('All disagreements have been adjudicated.\n\n');

%% Define Decision Logics

logicNames = {'Inclusive', 'Discrete-only', 'Strict-discrete'};
nLogics = length(logicNames);

fprintf('Processing %d decision logics:\n', nLogics);
for i = 1:nLogics
    fprintf('  %d. %s\n', i, logicNames{i});
end
fprintf('\n');

%% Process Each Decision Logic

metricsArray = cell(nLogics, 1);
adjStatsArray = cell(nLogics, 1);

for iLogic = 1:nLogics
    logicName = logicNames{iLogic};
    fprintf('========================================\n');
    fprintf('Processing Logic %d/%d: %s\n', iLogic, nLogics, logicName);
    fprintf('========================================\n\n');
    
    % Reclassify disagreements according to this logic
    [adjStats, FP_becomes_TP, FN_becomes_TN] = reclassifyDisagreementsByLogic(disagreements, logicName);
    adjStatsArray{iLogic} = adjStats;
    
    % Calculate adjusted metrics
    metrics = calculateAdjudicatedMetrics(...
        nPositivesGT_original, ...
        nTruePositives_original, ...
        nPositivesDetector, ...
        disagreements, ...
        FP_becomes_TP, ...
        FN_becomes_TN, ...
        detectorResults, ...
        totalAudioDurationSeconds, ...
        detectionTolerance, ...
        logicName);
    
    metricsArray{iLogic} = metrics;
    
    fprintf('\nAdjudicated Performance (%s):\n', logicName);
    fprintf('  Ground Truth (adjusted): %d\n', metrics.nPositivesGT_adjusted);
    fprintf('  True Positives: %d\n', metrics.nTruePositives);
    fprintf('  False Positives: %d\n', metrics.nFalsePositives);
    fprintf('  False Negatives: %d\n', metrics.nFalseNegatives);
    fprintf('  True Negatives: %d\n', metrics.nTrueNegatives);
    fprintf('  Recall: %.4f\n', metrics.recall);
    fprintf('  Precision: %.4f\n', metrics.precision);
    fprintf('  F1 Score: %.4f\n', metrics.f1Score);
    fprintf('  AUC: %.4f\n\n', metrics.auc);
end

%% Generate Comparison Figures

fprintf('Generating performance comparison figures...\n');
figPath = fullfile(outputFolder, sprintf('performance_comparison_%s', ts));
plotAdjudicatedPerformanceComparison(metricsArray, logicNames, figPath);

fprintf('Generating confusion matrices...\n');
figPath = fullfile(outputFolder, sprintf('confusion_matrices_%s', ts));
createConfusionMatrices(metricsArray, logicNames, figPath);

%% Export Results to Excel

fprintf('Exporting results to Excel...\n');
excelPath = fullfile(outputFolder, sprintf('adjudicated_performance_metrics_%s.xlsx', ts));

% Build output table
outTableArray = cell(nLogics, 1);
for iLogic = 1:nLogics
    metrics = metricsArray{iLogic};
    adjStats = adjStatsArray{iLogic};
    
    % Get confidence percentiles
    confDist = metrics.confidenceDistribution;
    confPercentile1 = confDist.confPercentiles(1);
    confPercentile50 = confDist.confPercentiles(5);
    confPercentile99 = confDist.confPercentiles(9);
    
    % Get temperature scaling
    if isfield(metrics, 'temperatureScaling') && isfield(metrics.temperatureScaling, 'optimalTemperature')
        tempRecommendation = metrics.temperatureScaling.optimalTemperature;
    else
        tempRecommendation = NaN;
    end
    
    % Create table row
    outTable = struct2table(metrics, 'AsArray', true);
    outTable = removevars(outTable, {'temperatureScaling', 'roc', 'performanceCurve', ...
        'confidenceDistribution'});
    
    % Add additional columns
    outTable = addvars(outTable, ...
        string(logicNames{iLogic}), ...
        confPercentile1, confPercentile50, confPercentile99, ...
        postProcOptions.AT, postProcOptions.DT, postProcOptions.AEAVD, ...
        postProcOptions.MT, postProcOptions.LT_scaler, postProcOptions.LT, ...
        ts, string(modelName), string(dataSetName), ...
        string(model.dataSynthesisParams.snrRange), ...
        string(featureFraming), string(frameStandardization), ...
        tempRecommendation, ...
        adjStats.nDiscreteCallsPresent, adjStats.nChorusPresent, ...
        adjStats.nDiscreteCallsChorusPresent, adjStats.nCallChorusAbsent, ...
        'NewVariableNames', {'DecisionLogic', ...
        '1stPrctileConf', '50thPrctileConf', '99thPrctileConf', ...
        'ActivationThreshold', 'DeactivationThreshold', 'AEAVD', ...
        'MergeThreshold', 'LengthThresholdScaler', 'LengthThreshold', ...
        'TestTimeStamp', 'ModelName', 'TestDataset', 'SequenceSNRRange', ...
        'FeatureFramingMode', 'FrameStandardization', 'TempRecommendation', ...
        'AdjStat_DiscreteCallsPresent', 'AdjStat_ChorusPresent', ...
        'AdjStat_DiscreteCallsChorusPresent', 'AdjStat_CallChorusAbsent'});
    
    outTableArray{iLogic} = outTable;
end

% Concatenate all tables
finalTable = vertcat(outTableArray{:});

% Write to Excel
writetable(finalTable, excelPath);
fprintf('Results saved to:\n  %s\n\n', excelPath);

%% Summary Report

fprintf('========================================\n');
fprintf('ANALYSIS COMPLETE\n');
fprintf('========================================\n\n');
fprintf('Output files saved to:\n  %s\n\n', outputFolder);
fprintf('Completed: %s\n', char(datetime("now")));

diary off

clearvars -except metricsArray logicNames adjStatsArray outputFolder
clear persistent