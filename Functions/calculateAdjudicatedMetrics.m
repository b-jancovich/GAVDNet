function metrics = calculateAdjudicatedMetrics(...
    nPositivesGT_original, ...
    nTruePositives_original, ...
    nPositivesDetector, ...
    disagreements, ...
    FP_becomes_TP, ...
    FN_becomes_TN, ...
    detectorResults, ...
    totalAudioDuration, ...
    detectionTolerance, ...
    logicName)
% calculateAdjudicatedMetrics
% Calculates complete performance metrics using adjudicated disagreements.
%
% This function replicates the metric calculations from 
% compareDetectionsToSubsampledTestDataset but uses the reclassified 
% disagreements based on analyst decisions and decision logic.
%
% Inputs:
%   nPositivesGT_original    - Original ground truth count
%   nTruePositives_original  - Original TP count (before adjudication)
%   nPositivesDetector       - Total detector detections
%   disagreements            - Struct with falsePositives and falseNegatives
%   FP_becomes_TP            - Logical array from reclassification
%   FN_becomes_TN            - Logical array from reclassification
%   detectorResults          - Original detector results for confidence scores
%   totalAudioDuration       - Total audio duration in seconds
%   detectionTolerance       - Detection tolerance in seconds
%   logicName                - Decision logic name
%
% Outputs:
%   metrics - Struct containing all performance metrics
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%% Calculate Adjusted Counts

FP = disagreements.falsePositives;
FN = disagreements.falseNegatives;

nFP_to_TP = sum(FP_becomes_TP);
nFN_to_TN = sum(FN_becomes_TN);

% Adjusted counts
nTruePositives_adj = nTruePositives_original + nFP_to_TP;
nFalsePositives_adj = length(FP) - nFP_to_TP;
nFalseNegatives_adj = length(FN) - nFN_to_TN;
nTrueNegatives_adj = nFN_to_TN;

% Adjusted ground truth (add missed calls, remove false alarms)
% FP→TP means GT missed these calls, so add them
% FN→TN means GT falsely claimed calls, so subtract them
nPositivesGT_adj = nPositivesGT_original + nFP_to_TP - nFN_to_TN;

fprintf('Calculating adjusted metrics...\n');
fprintf('  Adjusted counts: TP=%d, FP=%d, FN=%d, TN=%d\n', ...
    nTruePositives_adj, nFalsePositives_adj, nFalseNegatives_adj, nTrueNegatives_adj);

%% Calculate Core Performance Metrics

if nPositivesGT_adj > 0
    recall = nTruePositives_adj / nPositivesGT_adj;
else
    recall = NaN;
    warning('Adjusted ground truth is zero; recall is NaN');
end

if nPositivesDetector > 0
    precision = nTruePositives_adj / nPositivesDetector;
else
    precision = NaN;
    warning('No detector detections; precision is NaN');
end

if ~isnan(precision) && ~isnan(recall) && (precision + recall) > 0
    f1Score = 2 * (precision * recall) / (precision + recall);
else
    f1Score = NaN;
end

fprintf('  Core metrics: Recall=%.4f, Precision=%.4f, F1=%.4f\n', recall, precision, f1Score);

%% Build Confidence Scores and Labels for ROC Calculation

% We need confidence scores and labels for all evaluated detections
% Start with all detector results
nDetections = length(detectorResults);
confidenceScores = nan(nDetections, 1);
resultLabels = zeros(nDetections, 1);

% Extract confidence scores from all results
for i = 1:nDetections
    if isfield(detectorResults(i), 'confidence') && ...
       isnumeric(detectorResults(i).confidence) && ...
       isscalar(detectorResults(i).confidence)
        confidenceScores(i) = detectorResults(i).confidence;
    end
end

% Build labels: 1 for TP, 0 for FP
% Original TPs (not in disagreements) remain TPs
% Original FPs need to be checked against reclassification

% Strategy: Mark all as TP initially (since most weren't disagreements)
% Then mark FPs according to reclassification
resultLabels(:) = 1;

% Find indices of FPs in detector results and mark appropriately
for i = 1:length(FP)
    % Find this FP in detector results by matching time and filename
    fpStartTime = FP(i).DetectionStartTime;
    fpFilename = FP(i).AudioFilename;
    
    % Find matching result
    for j = 1:nDetections
        if isfield(detectorResults(j), 'eventStartTime') && ...
           isfield(detectorResults(j), 'fileName') && ...
           isdatetime(detectorResults(j).eventStartTime) && ...
           strcmp(detectorResults(j).fileName, fpFilename)
            
            timeDiff = abs(seconds(detectorResults(j).eventStartTime - fpStartTime));
            if timeDiff < 1e-6  % Match found
                if FP_becomes_TP(i)
                    resultLabels(j) = 1;  % Remains TP
                else
                    resultLabels(j) = 0;  % Is FP
                end
                break;
            end
        end
    end
end

% Filter out NaN confidence scores for ROC calculation
validIdx = ~isnan(confidenceScores);
confidenceScores_valid = confidenceScores(validIdx);
resultLabels_valid = resultLabels(validIdx);

fprintf('  Valid confidence scores for ROC: %d/%d\n', sum(validIdx), nDetections);

%% Calculate ROC Curve and AUC

fprintf('Calculating ROC curve and AUC...\n');

X_roc = NaN; Y_tpr = NaN; T_thresholds_roc = NaN; AUC = NaN;

if ~isempty(confidenceScores_valid) && ~isempty(resultLabels_valid)
    uniqueLabels = unique(resultLabels_valid);
    
    if length(uniqueLabels) == 2
        % Normal case: have both TPs and FPs
        [X_roc, Y_tpr, T_thresholds_roc, AUC] = perfcurve(resultLabels_valid, confidenceScores_valid, 1, 'XCrit', 'fpr', 'YCrit', 'tpr');
        fprintf('  AUC: %.4f\n', AUC);
        
    elseif all(uniqueLabels == 1)
        % All TPs
        warning('All evaluated results are TPs. ROC is degenerate. AUC = 1.0');
        AUC = 1.0;
        X_roc = [0; 0];
        Y_tpr = [0; 1];
        T_thresholds_roc = [max(confidenceScores_valid)+eps(max(confidenceScores_valid)); min(confidenceScores_valid)-eps(min(confidenceScores_valid))];
        
    elseif all(uniqueLabels == 0)
        % All FPs
        warning('All evaluated results are FPs. ROC is degenerate. AUC = 0.0');
        AUC = 0.0;
        X_roc = [0; 1];
        Y_tpr = [0; 0];
        T_thresholds_roc = [max(confidenceScores_valid)+eps(max(confidenceScores_valid)); min(confidenceScores_valid)-eps(min(confidenceScores_valid))];
    end
else
    warning('Cannot calculate ROC: insufficient data');
end

%% Apply Temperature Scaling Calibration

fprintf('Applying temperature scaling calibration...\n');

temperatureScaling = struct();
temperatureScaling.optimalTemperature = NaN;
temperatureScaling.originalAUC = AUC;
temperatureScaling.calibratedAUC = NaN;

if ~isempty(confidenceScores_valid) && length(uniqueLabels) == 2
    try
        [optimalTemperature, calibratedConfidences] = calculateOptimalTemperature(confidenceScores_valid, resultLabels_valid);
        
        % Recalculate AUC with calibrated scores
        [~, ~, ~, AUC_cal] = perfcurve(resultLabels_valid, calibratedConfidences, 1, 'XCrit', 'fpr', 'YCrit', 'tpr');
        
        temperatureScaling.optimalTemperature = optimalTemperature;
        temperatureScaling.calibratedAUC = AUC_cal;
        temperatureScaling.calibratedConfidences = calibratedConfidences;
        
        fprintf('  Optimal temperature: %.4f\n', optimalTemperature);
        fprintf('  Calibrated AUC: %.4f (Original: %.4f)\n', AUC_cal, AUC);
    catch ME
        warning('Temperature scaling failed: %s', ME.message);
    end
else
    fprintf('  Skipping temperature scaling: insufficient data diversity\n');
end

%% Calculate Detection Performance Curve (TPR vs FAPS)

fprintf('Calculating detection performance curve (TPR vs FAPS)...\n');

perf_falseAlarmRates = NaN;
perf_detectionRates = NaN;
T_thresholds_perf = NaN;

if totalAudioDuration > 0 && nPositivesGT_adj > 0 && ~isempty(confidenceScores_valid)
    
    uniqueScoresSorted = sort(unique(confidenceScores_valid), 'descend');
    
    if ~isempty(uniqueScoresSorted)
        T_thresholds_perf = [uniqueScoresSorted(1)+eps(uniqueScoresSorted(1)); uniqueScoresSorted; uniqueScoresSorted(end)-eps(uniqueScoresSorted(end))];
        T_thresholds_perf = unique(T_thresholds_perf);
        T_thresholds_perf = sort(T_thresholds_perf, 'descend');
        
        perf_detectionRates = zeros(length(T_thresholds_perf), 1);
        perf_falseAlarmRates = zeros(length(T_thresholds_perf), 1);
        
        for i = 1:length(T_thresholds_perf)
            thr = T_thresholds_perf(i);
            aboveThresholdIdx = (confidenceScores_valid >= thr);
            
            tpAtThreshold = sum(resultLabels_valid(aboveThresholdIdx) == 1);
            fpAtThreshold = sum(resultLabels_valid(aboveThresholdIdx) == 0);
            
            perf_detectionRates(i) = tpAtThreshold / nPositivesGT_adj;
            perf_falseAlarmRates(i) = fpAtThreshold / totalAudioDuration;
        end
        
        fprintf('  Performance curve calculated\n');
    end
else
    warning('Cannot calculate performance curve: missing duration, GT, or scores');
end

%% Analyze Confidence Score Distribution

fprintf('Analyzing confidence distribution...\n');

[confPercentiles, percentiles] = analyseConfidenceDistribution(detectorResults, false);

confidenceDistribution = struct();
confidenceDistribution.confPercentiles = confPercentiles;
confidenceDistribution.percentiles = percentiles;

%% Assemble Output Metrics Structure

metrics = struct();

% Decision logic
metrics.decisionLogic = logicName;

% Counts
metrics.nPositivesGT_original = nPositivesGT_original;
metrics.nPositivesGT_adjusted = nPositivesGT_adj;
metrics.nPositivesDetector = nPositivesDetector;
metrics.nTruePositives = nTruePositives_adj;
metrics.nFalsePositives = nFalsePositives_adj;
metrics.nFalseNegatives = nFalseNegatives_adj;
metrics.nTrueNegatives = nTrueNegatives_adj;

% Performance metrics
metrics.recall = recall;
metrics.sensitivity = recall;
metrics.precision = precision;
metrics.f1Score = f1Score;
metrics.auc = AUC;

% Curves
metrics.roc = struct('fpr', X_roc, 'tpr', Y_tpr, 'thresholds', T_thresholds_roc);
metrics.performanceCurve = struct('faps', perf_falseAlarmRates, 'tpr', perf_detectionRates, 'thresholds', T_thresholds_perf);

% Additional info
metrics.totalAudioDuration_sec = totalAudioDuration;
metrics.detectionTolerance_sec = detectionTolerance;
metrics.temperatureScaling = temperatureScaling;
metrics.confidenceDistribution = confidenceDistribution;
metrics.matchingAlgorithm = 'Hungarian (matchpairs) with adjudication';

fprintf('Metrics calculation complete.\n\n');

end