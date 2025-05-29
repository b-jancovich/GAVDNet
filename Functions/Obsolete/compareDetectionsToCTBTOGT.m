function metrics = compareDetectionsToCTBTOGT(groundtruthPath, inferenceResultsPath, detectionTolerance)
% COMPAREDETECTIONS Compare GAVDNet detector results with ground truth 
% detectios and calculate metrics.
%
% Compares detection results from GAVDNet inference with ground truth detections,
% calculating performance metrics including true positives, false positives,
% false negatives, precision, recall, F1 score, and ROC curve analysis.
%
% This function is designed to use detections from [1] as groundtruth.
% These detections are stored as mat files, containing three matrices;
% detections, snr_mois and snr_sem. For this script, only 'detections' is 
% used. The detections matrix contains a row for every the detected event 
% in the test audio. The columns of this matrix contain the following:
%   Column 1 - Year
%   Column 2 - Day of year (1 to 365 (or 366))
%   Column 3 - Month number (1 to 12)
%   Column 4 - Week number (1 to 52)
%   Column 5 - Detection time (in Matlab serial time aka. datenum)
%   Column 6 - Estiamted SNR of the call
%   Column 7 - Estimated SINR (Signal to Interference plus Noise Ratio)
%   Column 8 - class of SNR. Classes range from -11 (for a SNR = -inf) to 10. 
%
% Inputs:
%   groundtruthPath      - Path to ground truth .mat file
%   inferenceResultsPath - Path to inference results .mat file
%   detectionTolerance   - Time tolerance in seconds for matching detections
%
% Outputs:
%   metrics - Struct containing all calculated performance metrics
%
% Metrics included:
%   - numResultsDetections: Number of detections in results
%   - numGroundtruthDetections: Number of detections in groundtruth
%   - numTruePositives: Number of true positive detections
%   - numFalsePositives: Number of false positive detections
%   - numFalseNegatives: Number of false negative detections
%   - recall: Recall/Sensitivity score
%   - precision: Precision score
%   - f1Score: F1 score
%   - auc: Area under ROC curve
%
% References:
%   [1] Leroy, Emmanuelle C., Jean-Yves Royer, Abigail Alling, Ben Maslen, 
%   and Tracey L. Rogers. “Multiple Pygmy Blue Whale Acoustic Populations 
%   in the Indian Ocean: Whale Song Identifies a Possible New Population.” 
%   Scientific Reports 11, no. 1 (December 2021): 8762. 
%   https://doi.org/10.1038/s41598-021-88062-5.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%% Load data
fprintf('Loading ground truth data from: %s\n', groundtruthPath);
groundtruthData = load(groundtruthPath);

fprintf('Loading inference results from: %s\n', inferenceResultsPath);
resultsData = load(inferenceResultsPath);

%% Extract detections
groundtruthDetections = groundtruthData.detections;
resultsDetections = resultsData.detections;

%% Process ground truth data
% Extract detection times (column 5 is the detection time in datenum format)
gtTimes = groundtruthDetections(:, 5);
numGroundtruthDetections = length(gtTimes);
fprintf('Number of ground truth detections: %d\n', numGroundtruthDetections);

%% Process results data
% Filter out results with failure comments
validResultsIdx = true(size(resultsDetections));
for i = 1:length(resultsDetections)
    if isfield(resultsDetections(i), 'failComment') && ~isempty(resultsDetections(i).failComment)
        validResultsIdx(i) = false;
        fprintf('Skipping result with failure: %s\n', resultsDetections(i).failComment);
    end
end
validResults = resultsDetections(validResultsIdx);

% Extract detection times and confidence scores
resultsTimes = [];
confidenceScores = [];

for i = 1:length(validResults)
    if isfield(validResults(i), 'eventStartTime') && ~isempty(validResults(i).eventStartTime)
        % Convert datetime to datenum for consistent comparison
        resultsTimes(end+1) = datenum(validResults(i).eventStartTime);
        
        if isfield(validResults(i), 'confidence') && ~isempty(validResults(i).confidence)
            confidenceScores(end+1) = validResults(i).confidence;
        else
            confidenceScores(end+1) = 1.0; % Default confidence if not available
        end
    end
end

resultsTimes = resultsTimes(:);  % Ensure column vector
confidenceScores = confidenceScores(:);  % Ensure column vector
numResultsDetections = length(resultsTimes);

fprintf('Number of valid result detections: %d\n', numResultsDetections);

%% Match detections using time tolerance
% Convert tolerance from seconds to datenum format
toleranceInDays = detectionTolerance / (24 * 60 * 60);

% Initialize arrays for tracking matches
truePositives = false(numGroundtruthDetections, 1);
matchedResults = false(numResultsDetections, 1);

% For each ground truth detection, find the closest result within tolerance
for i = 1:numGroundtruthDetections
    gtTime = gtTimes(i);
    timeDifferences = abs(resultsTimes - gtTime);
    
    % Find the closest match within tolerance
    [minDiff, minIdx] = min(timeDifferences);
    
    if minDiff <= toleranceInDays && ~matchedResults(minIdx)
        truePositives(i) = true;
        matchedResults(minIdx) = true;
    end
end

% Count metrics
numTruePositives = sum(truePositives);
numFalsePositives = numResultsDetections - numTruePositives;
numFalseNegatives = numGroundtruthDetections - numTruePositives;

% Calculate performance metrics
recall = numTruePositives / (numTruePositives + numFalseNegatives);
sensitivity = recall; % Sensitivity is the same as recall
precision = numTruePositives / (numTruePositives + numFalsePositives);
f1Score = 2 * (precision * recall) / (precision + recall);

fprintf('\nPerformance Metrics:\n');
fprintf('True Positives: %d\n', numTruePositives);
fprintf('False Positives: %d\n', numFalsePositives);
fprintf('False Negatives: %d\n', numFalseNegatives);
fprintf('Recall/Sensitivity: %.4f\n', recall);
fprintf('Precision: %.4f\n', precision);
fprintf('F1 Score: %.4f\n', f1Score);

%% Calculate ROC curve and AUC
% For ROC calculation, we need to vary the confidence threshold and record TPR and FPR
% First, sort detection confidences
[sortedConfidences, sortIdx] = sort(confidenceScores, 'descend');
sortedTimes = resultsTimes(sortIdx);

% Calculate TPR and FPR at different thresholds
% Use unique confidence values as thresholds plus 0 and 1
uniqueConfidences = unique(sortedConfidences);
thresholds = [max(1.0, max(uniqueConfidences)+0.1); uniqueConfidences; -0.1];
numThresholds = length(thresholds);
tpr = zeros(numThresholds, 1);
fpr = zeros(numThresholds, 1);

for i = 1:numThresholds
    threshold = thresholds(i);
    
    % Determine which detections pass this threshold
    aboveThreshold = sortedConfidences >= threshold;
    currentResultTimes = sortedTimes(aboveThreshold);
    
    % For this threshold, which ground truth detections are matched?
    currMatchedGT = false(numGroundtruthDetections, 1);
    
    % Create a more efficient matching algorithm using vectorization
    for j = 1:numGroundtruthDetections
        gtTime = gtTimes(j);
        if ~isempty(currentResultTimes)
            timeDifferences = abs(currentResultTimes - gtTime);
            minDiff = min(timeDifferences);
            if minDiff <= toleranceInDays
                currMatchedGT(j) = true;
            end
        end
    end
    
    % Calculate metrics for this threshold
    currTP = sum(currMatchedGT);
    currFP = sum(aboveThreshold) - currTP;
    
    % True positive rate (Recall/Sensitivity)
    if numGroundtruthDetections > 0
        tpr(i) = currTP / numGroundtruthDetections;
    else
        tpr(i) = 0;
    end
    
    % Calculate false positive rate
    % Since we're dealing with event detection, we need to estimate the total number of 
    % potential false alarm opportunities. We'll use the total number of inference results
    % as a denominator, which is a common approach in detection tasks.
    if numResultsDetections > 0
        fpr(i) = currFP / numResultsDetections;
    else
        fpr(i) = 0;
    end
end

% Calculate AUC using trapezoidal integration
auc = trapz(fpr, tpr);
fprintf('Area Under ROC Curve (AUC): %.4f\n', auc);

%% Plot ROC curve
figure;
plot(fpr, tpr, 'b-', 'LineWidth', 2);
hold on;
plot([0, 1], [0, 1], 'r--'); % Random classifier line
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Receiver Operating Characteristic (ROC) Curve');
legend(['AUC = ' num2str(auc, '%.4f')], 'Random Classifier');
grid on;
hold off;

%% Prepare output metrics structure
metrics = struct();
metrics.numResultsDetections = numResultsDetections;
metrics.numGroundtruthDetections = numGroundtruthDetections;
metrics.numTruePositives = numTruePositives;
metrics.numFalsePositives = numFalsePositives;
metrics.numFalseNegatives = numFalseNegatives;
metrics.recall = recall;
metrics.sensitivity = sensitivity;
metrics.precision = precision;
metrics.f1Score = f1Score;
metrics.auc = auc;
metrics.roc = struct('fpr', fpr, 'tpr', tpr);
end