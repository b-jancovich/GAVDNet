% GAVDNet Parametric Sweep Script
%
% This script combines the GAVNet_Run_Detector and GAVDNet Compare results 
% to GroundTruth scripts to perform systematic parametric sweeps of 
% postprocessing parameters (AT and LT_Scaler). It loads raw detection 
% results, sweeps through parameter combinations, measures performance 
% against groundtruth, and saves results to Excel.
%
% The script loads the raw detection results mat file created in the first 
% part of the GAVNet_Run_Detector script, then systematically sweeps through 
% combinations of values for the AT and LT_Scaler postprocessing parameters, 
% measuring performance against groundtruth for each combination.
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

% Path to the config file:
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";
% configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_SORP_BmAntZ.m";

% Path to the Ground Truth Annotations file
groundtruthPath = "D:\GAVDNet\Chagos_DGS\Test Data\2007subset\test_dataset_detection_list.mat";
% groundtruthPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Detector Test Datasets\AAD_AcousticTrends_BlueFinLibrary\DATA\casey2014\Casey2014.Bm.Ant-Z.selections.txt";

% Test dataset format
gtFormat = 'CTBTO'; % Either "CTBTO" or "SORP"
 
% True known call duration (Only used to set end time for missed detections)
maxDetectionDuration = 40; % (seconds)
% NOT the max duration of the calls in the training data, as these may not
% contain the entire song, but the maximum duration of the real song, as
% observed in real recordings.

% Parameter sweep values
AT_sweep_values = linspace(0.95, 0.005, 30);

LTScaler_sweep_values = 0.5;

% % Different output paths for each version of the detector we are testing:
% many_gavdNetDataPaths{1} = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-10 to 10 Single Exemplar";
% many_gavdNetDataPaths{2} = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-10 to 10";
% many_gavdNetDataPaths{3} = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-6 to 10";
% many_gavdNetDataPaths{4} = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-3 to 10";
% 
% % Output paths for the results from each version of the detector:
% many_inferenceOutputPaths{1} = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-10 to 10 Single Exemplar";
% many_inferenceOutputPaths{2} = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-10 to 10";
% many_inferenceOutputPaths{3} = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-6 to 10";
% many_inferenceOutputPaths{4} = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-3 to 10";

% Different output paths for each version of the detector we are testing:
many_inferenceOutputPaths{1} = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-10 to 10 Single Exemplar";
many_inferenceOutputPaths{2} = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-10 to 10";
many_inferenceOutputPaths{3} = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-6 to 10";
many_inferenceOutputPaths{4} = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-3 to 10";

% Output paths for the results from each version of the detector:
many_gavdNetDataPaths{1} = "D:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar";
many_gavdNetDataPaths{2} = "D:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10";
many_gavdNetDataPaths{3} = "D:\GAVDNet\Chagos_DGS\Training & Models\-6 to 10";
many_gavdNetDataPaths{4} = "D:\GAVDNet\Chagos_DGS\Training & Models\-3 to 10";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.
%% Set Paths and Load Input Variables

for modelNum = 1:length(many_gavdNetDataPaths) 

    % Add dependencies to path
    run(configPath) % Load config file
    projectRoot = pwd;
    [gitRoot, ~, ~] = fileparts(projectRoot);
    addpath(fullfile(projectRoot, "Functions"))
    
    % Overwrite static paths from config with ones for the current loop
    % iteration:
    gavdNetDataPath = many_gavdNetDataPaths{modelNum};
    inferenceOutputPath = many_inferenceOutputPaths{modelNum};
    
    %% Start logging
    
    % Begin logging
    ts = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
    logname = ['parametric_sweep_log_', ts, '.txt'];
    diary(fullfile(inferenceOutputPath, logname));
    
    fprintf('Starting parametric sweep at %s\n', ts);
    fprintf('AT sweep values: [%s]\n', join(string(AT_sweep_values), ', '));
    fprintf('LT_Scaler sweep values: [%s]\n', join(string(LTScaler_sweep_values), ', '));
    
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
    
    %% Load raw detection results
    
    saveNamePathRaw = fullfile(inferenceOutputPath, 'detector_raw_results.mat');
    
    if ~exist(saveNamePathRaw, 'file')
        error(['Raw detection results file not found: %s\n', ...
               'Please run the first part of GAVNet_Run_Detector script first to generate raw results.'], ...
               saveNamePathRaw);
    end
    
    fprintf('Loading raw detection results from: %s\n', saveNamePathRaw);
    rawResults = load(saveNamePathRaw, 'results');
    rawResults = rawResults.results;
    fprintf('Loaded %d files with raw detection probabilities\n', length(rawResults));
    
    %% Set up output Excel file
    
    gtCompareResultsPath = fullfile(inferenceOutputPath, "parametric_sweep_results.xlsx");
    
    %% Calculate total number of parameter combinations
    
    totalCombinations = length(AT_sweep_values) * length(LTScaler_sweep_values);
    fprintf('Total parameter combinations to test: %d\n', totalCombinations);
    
    %% Initialize data collection for plotting
    
    sweepResults = struct();
    sweepResults.AT_values = [];
    sweepResults.LTScaler_values = [];
    sweepResults.precision = [];
    sweepResults.recall = [];
    sweepResults.f1Score = [];
    
    %% Main parametric sweep loop
    
    combinationCounter = 0;
    
    for i = 1:length(AT_sweep_values)
        for j = 1:length(LTScaler_sweep_values)
            
            combinationCounter = combinationCounter + 1;
            currentAT = AT_sweep_values(i);
            currentLTScaler = LTScaler_sweep_values(j);
            
            fprintf('\n=== Combination %d/%d ===\n', combinationCounter, totalCombinations);
            fprintf('Testing AT = %.4f, LT_Scaler = %.3f\n', currentAT, currentLTScaler);
                   
            % Create a copy of the original postProcOptions structure
            currentPostProcOptions = postProcOptions;
            
            % Update the parameters we're sweeping
            currentPostProcOptions.AT = currentAT;
            currentPostProcOptions.LT_scaler = currentLTScaler;
            
            % Calculate the length threshold based on mean training call duration
            currentPostProcOptions.LT = model.dataSynthesisParams.meanTargetCallDuration * currentLTScaler;
            
            % Set maximum expected call duration from the longest signal in the
            % training dataset, with a +20% tolerance
            currentPostProcOptions.maxTargetCallDuration = model.dataSynthesisParams.maxTargetCallDuration * 1.2;
            
            % Run postprocessing for all files with current parameters
            fprintf('Postprocessing model outputs for %d files...\n', length(rawResults));
            
            % Create a copy of results for this parameter combination
            currentResults = rawResults;
            
            for fileIdx = 1:length(currentResults)
                fprintf('Post-processing raw results for file %d\n', fileIdx)
                % Skip files that failed during inference
                if isfield(currentResults(fileIdx), 'failComment')
                    continue;
                end
                
                % Get audio for this file
                try
                    [audioIn, fileFs] = audioread(fullfile(inferenceAudioPath, currentResults(fileIdx).fileName));
                catch ME
                    warning('Could not read audio file: %s. Skipping...', currentResults(fileIdx).fileName);
                    continue;
                end
                
                % Run postprocessing to determine decision boundaries
                [currentResults(fileIdx).eventSampleBoundaries, ~, ...
                    currentResults(fileIdx).confidence] = gavdNetPostprocess(...
                    audioIn, fileFs, currentResults(fileIdx).probabilities, model.preprocParams, ...
                    currentPostProcOptions);
                
                % Get number of detections
                currentResults(fileIdx).nDetections = size(currentResults(fileIdx).eventSampleBoundaries, 1);
                
                % Get the datetime start and end times for each detected event
                if ~isempty(currentResults(fileIdx).eventSampleBoundaries)
                    for detIdx = 1:currentResults(fileIdx).nDetections
                        
                        % Get event boundaries (as sample indices)
                        eventStart = currentResults(fileIdx).eventSampleBoundaries(detIdx, 1);
                        eventEnd = currentResults(fileIdx).eventSampleBoundaries(detIdx, 2);
                        
                        % Convert sample indices to datetime relative to file start
                        currentResults(fileIdx).eventTimesDT(detIdx, 1) = currentResults(fileIdx).sampleDomainTimeVector(eventStart);
                        currentResults(fileIdx).eventTimesDT(detIdx, 2) = currentResults(fileIdx).sampleDomainTimeVector(eventEnd);
                    end
                end
            end
            
            % Flatten detections to one row per detection
            results = flattenDetections(currentResults, model.preprocParams);
    
            % Save temporary postprocessed results for comparison
            tempResultsPath = fullfile(inferenceOutputPath, 'temp_detector_results_postprocessed.mat');
            save(tempResultsPath, 'results', 'featureFraming', 'frameStandardization', 'currentPostProcOptions', '-v7.3');
            
            % Compare detector output to groundtruth
            fprintf('Comparing detections to groundtruth...\n');
            [metrics, FP, FN] = compareDetectionsToSubsampledTestDataset(...
                groundtruthPath, tempResultsPath, detectionTolerance, maxDetectionDuration, gtFormat);
            
            % Report
            fprintf('Performance metrics: Precision=%.3f, Recall=%.3f, F1=%.3f\n', ...
                    metrics.precision, metrics.recall, metrics.f1Score);
            
            % Store results for plotting
            sweepResults.AT_values(end+1) = currentAT;
            sweepResults.LTScaler_values(end+1) = currentLTScaler;
            sweepResults.precision(end+1) = metrics.precision;
            sweepResults.recall(end+1) = metrics.recall;
            sweepResults.f1Score(end+1) = metrics.f1Score;
               
            % Get test time and dataset name
            testCompleteTime = string(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
            [~, dataSetName, ~] = fileparts(fileparts(groundtruthPath));
            
            % Get confidence Metrics
            confDist = metrics.confidenceDistribution;
            confPercentile1 = confDist.confPercentiles(1);
            confPercentile50 = confDist.confPercentiles(5);
            confPercentile99 = confDist.confPercentiles(9);
    
            % Get Temperature Metrics
            tempRecommendation = metrics.temperatureScaling.optimalTemperature;
    
            % Prep results for export to Excel spreadsheet
            outTable = struct2table(metrics);
            outTable = removevars(outTable, {'temperatureScaling', 'roc', 'performanceCurve', ...
                'evaluatedResultCount', 'numResultsExcluded_NoScoreOrTime',...
                'groundtruthSource', 'numResultsExcluded_InferenceFailures',...
                'matchingAlgorithm', 'totalAudioDuration_sec', 'confidenceDistribution',...
                'detectionTolerance_sec', 'sensitivity'});
            newNames = {'1stPrctileConf', '50thPrctileConf', '99thPrctileConf', ...
                'ActivationThreshold', 'DeactivationThreshold', 'AEAVD', ...
                'MergeThreshold', 'LengthThresholdScaler', 'LengthThreshold', ...
                'TestTimeStamp', 'ModelName', 'TestDataset', 'SequenceSNRRange',...
                'FeatureFramingMode', 'FrameStandardization', 'TempRecommendation'};
            outTable = addvars(outTable, confPercentile1, confPercentile50, ...
                confPercentile99, currentPostProcOptions.AT, currentPostProcOptions.DT, ...
                currentPostProcOptions.AEAVD, currentPostProcOptions.MT, currentPostProcOptions.LT_scaler, ...
                currentPostProcOptions.LT, testCompleteTime, string(modelName), dataSetName,...
                model.dataSynthesisParams.snrRange, string(featureFraming), ...
                string(frameStandardization),tempRecommendation, ...
                'NewVariableNames', newNames);
    
            % Write output to Excel
            if exist(gtCompareResultsPath, 'file') == 2
                % Append data without headers
                writetable(outTable, gtCompareResultsPath, 'WriteMode', 'append', 'WriteVariableNames', false);
            else
                % File does not exist — write with headers
                writetable(outTable, gtCompareResultsPath);
            end
            
            % Save disagreements for this parameter combination in a sub-folder
            disagreements = struct('falsePositives', FP, 'falseNegatives', FN);
            disagreementsPath = fullfile(inferenceOutputPath, 'sweep_disagreements');
            if ~isfolder(disagreementsPath)
                mkdir(disagreementsPath);
            end
            saveNamePath = fullfile(disagreementsPath,...
                sprintf('disagreements_AT%.4f_LTS%.3f_%s.mat', currentAT, currentLTScaler, testCompleteTime));
            save(saveNamePath, 'disagreements', '-v7.3');
            
            % Clean up temporary files
            if exist(tempResultsPath, 'file')
                delete(tempResultsPath);
            end
            fprintf('Completed combination %d/%d\n', combinationCounter, totalCombinations);
        end
    end
    
    %% Generate and save recall-precision curve figure
    
    fprintf('\nGenerating recall-precision curve figure...\n');
    figPath = fullfile(inferenceOutputPath, sprintf('recall_precision_curve_%s', ts));
    plotRecallPrecisionCurve(sweepResults, figPath);
    
    %% Summary

    fprintf('\n=== PARAMETRIC SWEEP COMPLETED ===\n');
    fprintf('Total combinations tested: %d\n', totalCombinations);
    fprintf('Results saved to: %s\n', gtCompareResultsPath);
    fprintf('Disagreement files saved to: %s\n', disagreementsPath);
    fprintf('Recall-precision curve saved to: %s\n', figPath);
    
    % Calculate and display time taken
    sweepEndTime = datetime("now");
    fprintf('Parametric sweep completed at: %s\n', char(sweepEndTime));
    
    diary off
    fprintf('Log saved to: %s\n', fullfile(inferenceOutputPath, logname));

    %% Cleanup

    clearvars -except configPath groundtruthPath gtFormat maxDetectionDuration ...
        AT_sweep_values LTScaler_sweep_values many_gavdNetDataPaths...
        many_inferenceOutputPaths
    clear persistent

end
