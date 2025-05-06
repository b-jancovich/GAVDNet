function metrics = compareDetectionsToSubsampledTestDataset(groundtruthPath, inferenceResultsPath, detectionTolerance)
% compareDetectionsToSubsampledTestDataset_Hungarian
% Compare GAVDNet detector results with ground truth using OPTIMAL ASSIGNMENT
% (Hungarian algorithm via matchpairs) and calculate metrics.
%
% Compares detection results from GAVDNet inference with ground truth detections,
% calculating performance metrics including true positives, false positives,
% false negatives, precision, recall, F1 score, and ROC curve analysis.
% Uses optimal assignment (matchpairs) for matching detections.
%
% This function is designed to use detections from either testDatasetDetectionsList
% or testDatasetFileList tables created by the test_dataset_subsampling.m script.
%
% Inputs:
%   groundtruthPath      - Path to ground truth .mat file (either test_dataset_detection_list.mat
%                          or test_dataset_audiofile_list.mat)
%   inferenceResultsPath - Path to inference results .mat file
%   detectionTolerance   - Time tolerance in seconds for matching detections
%
% Outputs:
%   metrics - Struct containing all calculated performance metrics
%
% Metrics included:
%   - numResultsDetections: Number of detections in results (before score filtering)
%   - numGroundtruthDetections: Number of detections in groundtruth
%   - numTruePositives: Number of true positive detections (based on optimal matching)
%   - numFalsePositives: Number of false positive detections (based on optimal matching)
%   - numFalseNegatives: Number of false negative detections (based on optimal matching)
%   - recall: Recall/Sensitivity score
%   - precision: Precision score
%   - f1Score: F1 score
%   - auc: Area under ROC curve
%   - roc: Struct containing ROC curve data (fpr, tpr, thresholds)
%   - performanceCurve: Struct containing detection performance curve data (faps, tpr, thresholds)
%   - evaluatedResultCount: Number of results actually used (with valid scores)
%   - matchingAlgorithm: Indicates the matching method used ('Hungarian (matchpairs)')
%
% Ben Jancovich, 2025 (Revised based on feedback May 2024, Hungarian implementation, Fix May 2024)
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%% Load data
fprintf('Loading ground truth data from: %s\n', groundtruthPath);
groundtruthData = load(groundtruthPath);

fprintf('Loading inference results from: %s\n', inferenceResultsPath);
resultsData = load(inferenceResultsPath);

%% Process results data and identify files with failures

% Get list of files for which the inference script reported audio read failures
failedFiles = {};
validResultsIdx = true(size(resultsData.results));
fprintf('Checking %d inference results for failures...\n', ...
    length(resultsData.results));
numFailures = 0;
for i = 1:length(resultsData.results)
    if isfield(resultsData.results(i), 'failComment') && ...
            ~isempty(resultsData.results(i).failComment)
        validResultsIdx(i) = false;
        numFailures = numFailures + 1;
        % Store the filename of the failed file
        if isfield(resultsData.results(i), 'fileName')
            failedFiles{end+1} = resultsData.results(i).fileName;
            fprintf('Excluding failed file: %s - Reason: %s\n',...
                resultsData.results(i).fileName, resultsData.results(i).failComment);
        else
            fprintf('Excluding result %d with failure (no filename available): %s\n', ...
                i, resultsData.results(i).failComment);
        end
    end
end
validResults = resultsData.results(validResultsIdx);
if numFailures > 0
    fprintf('Excluded %d results due to processing failures.\n', numFailures);
end
if ~isempty(failedFiles)
    fprintf('List of unique failed filenames:\n');
    uniqueFailed = unique(failedFiles);
    for k=1:length(uniqueFailed)
        fprintf('  %s\n', uniqueFailed{k});
    end
end

% Extract detection times and confidence scores
resultsTimes = [];
confidenceScores = [];
resultsFileNames = {};
resultsFileDurations = [];
for i = 1:length(validResults)
    % Check if it's a valid detection entry (not just file info)
    if isfield(validResults(i), 'eventStartTime') && ...
            ~isempty(validResults(i).eventStartTime)

        % Convert datetime to datenum for consistent comparison
        resultsTimes(end+1) = datenum(validResults(i).eventStartTime);

        % Get confidence scores for valid detections
        if isfield(validResults(i), 'confidence') && ...
                ~isempty(validResults(i).confidence) && ...
                isnumeric(validResults(i).confidence)
            confidenceScores(end+1) = validResults(i).confidence;
        else
            % Use NaN if confidence is missing or invalid
            confidenceScores(end+1) = NaN; 
            warning('Missing or invalid confidence score for result index %d (within valid results). Assigning NaN.', i);
        end

        % Store file info for duration calculation, associate with this *detection* index temporarily
        if isfield(validResults(i), 'fileName') && ...
            ~isempty(validResults(i).fileName) && ...
            isfield(validResults(i), 'fileDuration') && ...
            ~isempty(validResults(i).fileDuration) && ...
            isnumeric(validResults(i).fileDuration)
           
            % Get filenames
            resultsFileNames{end+1} = validResults(i).fileName;
            resultsFileDurations(end+1) = validResults(i).fileDuration;
        else
            % Need placeholders if file info isn't available for a detection
            resultsFileNames{end+1} = '';
            resultsFileDurations(end+1) = NaN;
        end
    end
end

% Ensure column vectors
resultsTimes = resultsTimes(:);  
confidenceScores = confidenceScores(:);
resultsFileNames = resultsFileNames(:);
resultsFileDurations = resultsFileDurations(:);

% Remove any detections lacking a confidence score
validConfidenceIdx = ~isnan(confidenceScores);
numResultsOriginal = length(confidenceScores);
numRemoved_NoScore = 0;
if ~all(validConfidenceIdx)
    numRemoved_NoScore = sum(~validConfidenceIdx);
    fprintf('Warning: Removing %d detections with missing/invalid confidence scores.\n', ...
        numRemoved_NoScore);
    resultsTimes = resultsTimes(validConfidenceIdx);
    confidenceScores = confidenceScores(validConfidenceIdx);
    resultsFileNames = resultsFileNames(validConfidenceIdx);
    resultsFileDurations = resultsFileDurations(validConfidenceIdx);
else
    fprintf('All %d valid result detections have confidence scores.\n', ...
        length(resultsTimes));
end
% Get final results count for matching
numResultsDetections = length(resultsTimes); 
fprintf('Number of valid result detections for matching (with scores): %d\n', ...
    numResultsDetections);

%% Extract detections from groundtruth based on available format and filter failed files

% Determine which type of groundtruth file was loaded
if isfield(groundtruthData, 'testDatasetDetectionsList')
    gtTable = groundtruthData.testDatasetDetectionsList;
    gtSourceDescription = 'testDatasetDetectionsList';

    % Filter out entries from failed files if table contains filename information
    if ismember('fileName', gtTable.Properties.VariableNames) && ...
            ~isempty(failedFiles)
        originalGtCount = height(gtTable);
        validGtIdx = ~ismember(gtTable.fileName, failedFiles);
        gtTable = gtTable(validGtIdx, :);
        numRemovedGt = originalGtCount - height(gtTable);
        if numRemovedGt > 0
            fprintf('Removed %d ground truth entries corresponding to audio files that could not be read.\n', ...
                numRemovedGt);
        end
    end

    % Extract detection times from datenum field
    if ismember('datenum', gtTable.Properties.VariableNames)
        gtTimes = gtTable.datenum;
    else
        error('Ground truth table ''testDatasetDetectionsList'' is missing the required ''datenum'' column.');
    end

elseif isfield(groundtruthData, 'testDatasetFileList')
    gtTable = groundtruthData.testDatasetFileList;
    gtSourceDescription = 'testDatasetFileList';

    % Filter out entries from failed files if table contains filename information
    if ismember('filenames', gtTable.Properties.VariableNames) && ...
            ~isempty(failedFiles)

        % Original number of ground truth files
        originalGtFileCount = height(gtTable);

        % Check each cell entry in 'filenames' against the list of failed files
        validGtFileIdx = true(height(gtTable), 1);
        for i = 1:height(gtTable)

            % Handle various types for filename - cell, char, string etc.
            if iscell(gtTable.filenames) && ...
                    ~isempty(gtTable.filenames{i}) && ...
                    ismember(gtTable.filenames{i}, failedFiles)
                validGtFileIdx(i) = false;    

            elseif ischar(gtTable.filenames) && ...
                    size(gtTable.filenames,1) == 1 && ...
                    ismember(gtTable.filenames, failedFiles)
                validGtFileIdx(i) = false;

            elseif ischar(gtTable.filenames) && ...
                    size(gtTable.filenames,1) > 1 && ...
                    ismember(deblank(gtTable.filenames(i,:)), failedFiles)
                validGtFileIdx(i) = false;
            end
        end

        % Strip rows from ground truth table that had unreadable audio files
        gtTable = gtTable(validGtFileIdx, :);
        numRemovedGtFiles = originalGtFileCount - height(gtTable);
        if numRemovedGtFiles > 0
            fprintf('Removed %d ground truth file entries corresponding to failed processing files.\n', ...
                numRemovedGtFiles);
        end
    end

    % Extract detection times from the datenum_all cell array field
    if ismember('datenum_all', gtTable.Properties.VariableNames)
        % Extract and concatenate all detection times
        gtTimes = [];
        for i = 1:height(gtTable)
            if ~isempty(gtTable.datenum_all{i})

                % Ensure the content of the cell is numeric before concatenating
                if isnumeric(gtTable.datenum_all{i})
                    gtTimes = [gtTimes; gtTable.datenum_all{i}(:)];
                else
                    warning('Non-numeric data found in datenum_all for file index %d. Skipping.', i);
                end
            end
        end
    else
        error('Ground truth table ''testDatasetFileList'' is missing the required ''datenum_all'' column.');
    end

elseif isfield(groundtruthData, 'detections') &&...
        isnumeric(groundtruthData.detections) && ...
        size(groundtruthData.detections, 2) >= 5

    fprintf('Processing ground truth format: CTBTO matrix (assuming column 5 is datenum)\n');
    gtSourceDescription = 'CTBTO matrix';

    % Extract detection times (column 5 is the detection time in datenum format)
    gtTimes = groundtruthData.detections(:, 5);
    if ~isempty(failedFiles)
        fprintf('Warning: Cannot filter out failed files from CTBTO matrix format as filename info is not available in this structure.\n');
    end
else
    error('Unrecognized or invalid ground truth data format. Expected testDatasetDetectionsList, testDatasetFileList, or a numeric matrix named ''detections''.');
end

% Ensure gtTimes is a column vector
gtTimes = gtTimes(:);
numGroundtruthDetections = length(gtTimes);
fprintf('Number of ground truth detections after filtering: %d (from %s)\n',...
    numGroundtruthDetections, gtSourceDescription);

% Handle no results/GTs case:
if numResultsDetections == 0 && numGroundtruthDetections == 0
    error('Both results and ground truth contain zero detections for matching. Cannot calculate metrics.');
end

%% Optimal Matching Logic (Hungarian Algorithm via matchpairs)

fprintf('Matching results to ground truth using Optimal Assignment (Hungarian/matchpairs) with %.2f s tolerance...\n', ...
    detectionTolerance);
toleranceInDays = detectionTolerance / (24 * 60 * 60);
numTruePositives = 0;
matchedResultIndices = []; % Indices of results that are matched (TP)
if numResultsDetections > 0 && numGroundtruthDetections > 0
    % 1. Construct the Cost Matrix
    % Rows: Results (numResultsDetections), Columns: Ground Truth (numGroundtruthDetections)
    % Cost is time difference if within tolerance, Inf otherwise.
    costMatrix = Inf(numResultsDetections, numGroundtruthDetections);

    for i = 1:numResultsDetections
        % Calculate time differences for result 'i' against all GTs
        timeDiffs = abs(resultsTimes(i) - gtTimes);

        % Find GT indices within tolerance
        validGtIndices = find(timeDiffs <= toleranceInDays);

        % Assign the actual time difference as cost for valid pairs
        if ~isempty(validGtIndices)
            costMatrix(i, validGtIndices) = timeDiffs(validGtIndices);
        end
    end

    % 2. Run matchpairs
    % Find the minimum cost matching. Any assignment with cost > toleranceInDays
    % is disallowed (costOfNonAssignment). This ensures only pairs within
    % tolerance are matched, and it maximizes the number of such pairs
    % while minimizing the sum of their time differences.
    costOfNonAssignment = toleranceInDays;
    try
        % matchedRows: indices (1-based) of results (rows) that were matched
        % matchedCols: indices (1-based) of ground truth (cols) that were matched
        [matchedRows, ~, ~] = matchpairs(costMatrix, costOfNonAssignment);
        numTruePositives = length(matchedRows);
        matchedResultIndices = matchedRows; 

        fprintf('Optimal assignment found %d matches (True Positives).\n', ...
            numTruePositives);

    catch ME
        warning(ME.identifier, 'matchpairs encountered an error: %s', ...
            ME.message);
        % If matchpairs fails, metrics will be calculated assuming 0 TPs.
        numTruePositives = 0;
        matchedResultIndices = [];
    end

else
    fprintf('Skipping matching: Zero results or zero ground truth detections.\n');
    % numTruePositives remains 0
end

% 3. Determine Labels and Final Metrics
% Initialize all results as False Positives (label 0)
% Size this based on the number of results actually used in matching
resultLabels = zeros(numResultsDetections, 1);

% Set matched results as True Positives (label 1)
if ~isempty(matchedResultIndices)
    % matchedResultIndices contains the row indices of matching pairs
    % between GT and results - col1 is results indices and col2 is GT.
    resultLabels(matchedResultIndices(:, 1)) = 1;
end

% Calculate final metrics based on the optimal matching
numFalsePositives = numResultsDetections - numTruePositives;
numFalseNegatives = numGroundtruthDetections - numTruePositives;

% Calculate Performance Metrics based on the optimal matching
% Handle division by zero cases for metrics
if (numTruePositives + numFalseNegatives) > 0
    recall = numTruePositives / (numTruePositives + numFalseNegatives);
else
    recall = NaN; % If no GT (TP+FN=0), recall is undefined.
    if numGroundtruthDetections == 0 && numTruePositives == 0
        fprintf('Warning: Recall is NaN because there are no ground truth detections.\n');
    elseif numTruePositives > 0 % TP > 0 implies GT > 0, so denominator > 0. This case shouldn't be NaN.
        recall = 1.0; % Should only happen if FN=0 and TP>0
    end
end
sensitivity = recall; % Sensitivity is the same as recall

if (numTruePositives + numFalsePositives) > 0
    precision = numTruePositives / (numTruePositives + numFalsePositives);
else
    precision = NaN; % If no Results (TP+FP=0), precision is undefined.
    if numResultsDetections == 0 && numTruePositives == 0
        fprintf('Warning: Precision is NaN because there are no result detections for matching.\n');
    elseif numTruePositives > 0 % TP > 0 implies Results > 0, so denominator > 0. This case shouldn't be NaN.
        precision = 1.0; % Should only happen if FP=0 and TP>0
    end
end

if (precision + recall) > 0 && ~isnan(precision) && ~isnan(recall)
    f1Score = 2 * (precision * recall) / (precision + recall);
else
    f1Score = NaN; % If precision or recall is NaN, F1 is NaN.
    if isnan(precision) || isnan(recall)
        fprintf('Warning: F1 Score is NaN because Precision or Recall is NaN.\n');
    end
end

fprintf('\n--- Performance Metrics (using Optimal Matching) ---\n');
fprintf('Total Ground Truth Detections: %d\n', numGroundtruthDetections);
fprintf('Total Result Detections (input): %d\n', numResultsOriginal);
fprintf('Result Detections Evaluated (with score): %d\n', numResultsDetections);
fprintf('True Positives (TP):  %d\n', numTruePositives);
fprintf('False Positives (FP): %d\n', numFalsePositives);
fprintf('False Negatives (FN): %d\n', numFalseNegatives);
fprintf('---------------------------\n');
fprintf('Recall / Sensitivity (TPR): %.4f\n', recall);
fprintf('Precision          (PPV): %.4f\n', precision);
fprintf('F1 Score                : %.4f\n', f1Score);
fprintf('---------------------------\n');


%% Calculate ROC curve and AUC using perfcurve
% Use the 'resultLabels' (0 for FP, 1 for TP) and 'confidenceScores'
% Both should now have size [numResultsDetections, 1]

fprintf('Calculating ROC curve and AUC...\n');

% Initialize ROC outputs
X_roc = NaN; Y_tpr = NaN; T_thresholds_roc = NaN; AUC = NaN;

if numResultsDetections > 0 && numGroundtruthDetections > 0
    uniqueLabels = unique(resultLabels);

    if length(uniqueLabels) == 2 % Both TP (1) and FP (0) exist

        % Call perfcurve to calculate ROC and AUC
        [X_roc, Y_tpr, T_thresholds_roc, AUC] = perfcurve(resultLabels, confidenceScores, 1);
        fprintf('AUC calculated: %.4f\n', AUC);

        % Plot ROC curve
        figure;
        plot(X_roc, Y_tpr, 'b-', 'LineWidth', 2);
        hold on;
        plot([0, 1], [0, 1], 'r--'); % Random classifier line
        xlabel('False Positive Rate (FPR)');
        ylabel('True Positive Rate (TPR)');
        title(sprintf('Receiver Operating Characteristic (ROC) Curve (AUC = %.4f)', AUC));
        legend('Detector ROC', 'Random Classifier', 'Location', 'southeast');
        grid on;
        axis square; % Often helpful for ROC plots
        axis([0 1 0 1]); % Ensure axes are fixed
        hold off;

    elseif all(uniqueLabels == 1) && ~isempty(uniqueLabels) % All results are TPs
        warning('All %d evaluated results were True Positives based on matching. ROC curve is degenerate (point at FPR=0, TPR=1). Setting AUC = 1.0.', numResultsDetections);
        AUC = 1.0;
        X_roc = [0; 0]; % Represents point (0,1) pathologically
        Y_tpr = [0; 1];
        if ~isempty(confidenceScores)
            T_thresholds_roc = [max(confidenceScores)+eps; min(confidenceScores)]; 
        else 
            T_thresholds_roc = NaN;
        end
    elseif all(uniqueLabels == 0) && ~isempty(uniqueLabels) % All results are FPs
        warning('All %d evaluated results were False Positives based on matching. ROC curve is degenerate (point at FPR=1, TPR=0). Setting AUC = 0.0.', numResultsDetections);
        AUC = 0.0;
        X_roc = [0; 1]; % Represents point (1,0) pathologically
        Y_tpr = [0; 0];
        if ~isempty(confidenceScores) 
            T_thresholds_roc = [max(confidenceScores) + eps; min(confidenceScores)]; 
        else 
            T_thresholds_roc=NaN;
        end
    elseif isempty(uniqueLabels) % Should only happen if numResultsDetections is 0, handled earlier but check anyway
        warning('Cannot calculate ROC curve: No results detections to evaluate.');
        AUC = NaN; X_roc = NaN; Y_tpr = NaN; T_thresholds_roc = NaN;
    else % Only one label type present, but not 0 or 1? Should not happen.
        warning('Could not calculate ROC curve. Unexpected labels found: %s. Check resultLabels.', ...
            mat2str(uniqueLabels));
        AUC = NaN; X_roc = NaN; Y_tpr = NaN; T_thresholds_roc = NaN;
    end

    % Handle cases where ROC calculation is not possible due to lack of GT or Results
elseif numResultsDetections == 0
    warning('Cannot calculate ROC curve: No results detections with scores.');
    AUC = NaN; X_roc = NaN; Y_tpr = NaN; T_thresholds_roc = NaN;
elseif numGroundtruthDetections == 0
    warning('Cannot calculate ROC curve: No ground truth detections. AUC is undefined.');
    AUC = NaN; X_roc = NaN; Y_tpr = NaN; T_thresholds_roc = NaN; % Strictly undefined
end


%% --- Calculate and Plot Detection Performance Curve (TPR vs False Alarms Per Second) ---

fprintf('Calculating Detection Performance Curve (TPR vs FAPS)...\n');

% Initialize performance curve outputs
totalAudioDuration = 0;

% Calculate total audio duration robustly
% Use unique file durations associated with the *results* that were actually processed and scored
uniqueFileDurations = containers.Map('KeyType','char','ValueType','double');
validDurationCount = 0;
if ~isempty(resultsFileNames)
    for i = 1:length(resultsFileNames) % Iterate through the potentially filtered list
        fileName = resultsFileNames{i};
        fileDuration = resultsFileDurations(i);
        if ~isempty(fileName) && ~isnan(fileDuration) && fileDuration > 0
            % Store duration, keyed by filename. Overwrites duplicates, keeps last seen duration for a file.
            uniqueFileDurations(fileName) = fileDuration;
            validDurationCount = validDurationCount + 1;
        end
    end
    if ~isempty(uniqueFileDurations)
        totalAudioDuration = sum(cell2mat(values(uniqueFileDurations)));
        fprintf('Calculated total duration from %d unique files found in evaluated results: %.2f sec\n', ...
            length(uniqueFileDurations), totalAudioDuration);
    else
        fprintf('Warning: Could not determine total duration from results file info.\n');
    end

    if validDurationCount < numResultsDetections && numResultsDetections > 0
        fprintf('Warning: Duration info was missing or invalid for %d out of %d evaluated results.\n', ...
            numResultsDetections - validDurationCount, numResultsDetections);
    end
end

% Fallback to ground truth file list if needed
if totalAudioDuration <= 0 && isfield(groundtruthData, 'testDatasetFileList')
    fprintf('Attempting to calculate total duration from ground truth file list (testDatasetFileList).\n');
    gtTable = groundtruthData.testDatasetFileList; % Already loaded
    gtTableDuration = gtTable; % Start with full table
    % Filter out failed files if needed (re-applying filter logic here for duration specifically)
    if ismember('filenames', gtTable.Properties.VariableNames) && ~isempty(failedFiles)
        originalGtFileCount = height(gtTable);
        validGtFileIdx = true(height(gtTable), 1);
        for i = 1:height(gtTable)
            if iscell(gtTable.filenames) && ...
                ~isempty(gtTable.filenames{i}) && ...
                    ismember(gtTable.filenames{i}, failedFiles)
                validGtFileIdx(i) = false;
                % Add checks for char arrays if necessary, as in GT loading
            elseif ischar(gtTable.filenames) && ...
                    size(gtTable.filenames,1) == 1 && ...
                    ismember(gtTable.filenames, failedFiles) % Handle single char entry
                validGtFileIdx(i) = false;
            elseif ischar(gtTable.filenames) && ...
                    size(gtTable.filenames,1) > 1 && ...
                    ismember(deblank(gtTable.filenames(i,:)), failedFiles) % Handle char array
                validGtFileIdx(i) = false;
            end
        end
        gtTableDuration = gtTable(validGtFileIdx, :); % Use filtered table for duration sum
        numRemovedGtFiles = originalGtFileCount - height(gtTableDuration);
        if numRemovedGtFiles > 0
            fprintf('(Duration calculation): Ignored %d GT file entries corresponding to failed processing files.\n', numRemovedGtFiles);
        end
    end

    % Sum durations if the column exists and is valid
    if ismember('duration', gtTableDuration.Properties.VariableNames) && ...
            isnumeric(gtTableDuration.duration)
        validDurations = gtTableDuration.duration(gtTableDuration.duration > 0 & ...
            ~isnan(gtTableDuration.duration));
        totalAudioDuration = sum(validDurations);
        fprintf('Calculated total duration from ground truth file list: %.2f sec\n', ...
            totalAudioDuration);
    else
        fprintf('Warning: Could not find a valid numeric ''duration'' column in testDatasetFileList.\n');
        totalAudioDuration = 0; % Ensure it remains 0 if calculation failed
    end
end

% Proceed only if we have valid duration, GT, and results
if totalAudioDuration > 0 && numGroundtruthDetections > 0 && ...
        numResultsDetections > 0

    uniqueScores = unique(confidenceScores);
    if isempty(uniqueScores)
        warning('No unique scores found among evaluated results, cannot generate performance curve.');
        perf_falseAlarmRates = NaN; 
        perf_detectionRates = NaN; 
        T_thresholds_perf = NaN;
    else
        minScore = min(uniqueScores);
        maxScore = max(uniqueScores);
        % Thresholds: from slightly above max down to slightly below min
        T_thresholds_perf = sort([maxScore + eps; uniqueScores; minScore - eps], 'descend');

        perf_detectionRates = zeros(length(T_thresholds_perf), 1); % TPR
        perf_falseAlarmRates = zeros(length(T_thresholds_perf), 1); % FAPS

        for i = 1:length(T_thresholds_perf)
            thr = T_thresholds_perf(i);
            % Find results with scores >= threshold
            aboveThresholdIdx = (confidenceScores >= thr);

            % Count TP and FP *at this threshold* using the definitive resultLabels
            tpAtThreshold = sum(resultLabels(aboveThresholdIdx) == 1);
            fpAtThreshold = sum(resultLabels(aboveThresholdIdx) == 0);

            % Calculate rates
            perf_detectionRates(i) = tpAtThreshold / numGroundtruthDetections; % TPR = TP / (TP + FN) = TP / TotalGT
            perf_falseAlarmRates(i) = fpAtThreshold / totalAudioDuration; % FAPS = FP / TotalDuration
        end

        % --- Plot detection rate vs false alarms per second (semi-log plot) ---
        figure;
        % Use markers especially if points are sparse
        semilogx(perf_falseAlarmRates, perf_detectionRates, 'b.-', ...
            'LineWidth', 1.5, 'MarkerSize', 10);
        grid on;
        xlabel('False Alarms Per Second (FAPS) [log scale]');
        ylabel('Detection Rate (TPR)');
        title('Detection Performance Curve (TPR vs. FAPS)');
        ylim([0 1.05]); % Ensure full TPR range is visible
        % Add operating point (overall TP/FP rate, implies a single threshold was used for summary stats)
        op_FAPS = numFalsePositives / totalAudioDuration;
        op_TPR = recall; % Recall is TPR at the implicit operating point
        hold on;
        if ~isnan(op_FAPS) && ~isnan(op_TPR)
            plot(op_FAPS, op_TPR, 'ro', 'MarkerSize', 8, 'LineWidth', 2, ...
                'MarkerFaceColor', 'r', 'DisplayName', 'Operating Point');
            legend('Performance Curve', 'Operating Point', 'Location', ...
                'southeast');
        else
            legend('Performance Curve', 'Location', 'southeast');
        end
        hold off;

        % Report false alarm rate at operating point (using overall FP count)
        if ~isnan(op_FAPS)
            fprintf('False Alarms Per Second at operating point: %.6f\n', ...
                op_FAPS);
        else
            fprintf('False Alarms Per Second at operating point: NaN (likely zero duration or zero results)\n');
        end
    end % End check for empty uniqueScores

else
    warning('Cannot calculate or plot Detection Performance Curve.');
    if totalAudioDuration <= 0
        fprintf('Reason: Total audio duration is zero or negative.\n');
    elseif numGroundtruthDetections == 0
        fprintf('Reason: Number of ground truth detections is zero.\n');
    elseif numResultsDetections == 0
        fprintf('Reason: Number of result detections for matching is zero.\n');
    end
    % Ensure outputs are NaN if not calculated
    perf_falseAlarmRates = NaN;
    perf_detectionRates = NaN;
    T_thresholds_perf = NaN;
    fprintf('False Alarms Per Second at operating point: NaN\n');
end
fprintf('Total Audio Duration Analyzed: %.2f seconds (%.2f hours)\n', ...
    totalAudioDuration, totalAudioDuration/3600);


%% Prepare output metrics structure
metrics = struct();
metrics.numResultsDetections = numResultsOriginal; % Report original count before score filtering
metrics.numGroundtruthDetections = numGroundtruthDetections;
metrics.numTruePositives = numTruePositives;
metrics.numFalsePositives = numFalsePositives;
metrics.numFalseNegatives = numFalseNegatives;
metrics.recall = recall;
metrics.sensitivity = sensitivity; % Same as recall
metrics.precision = precision;
metrics.f1Score = f1Score;
metrics.auc = AUC;
metrics.roc = struct('fpr', X_roc, 'tpr', Y_tpr, 'thresholds', ...
    T_thresholds_roc); % Store actual ROC data from perfcurve
metrics.performanceCurve = struct('faps', perf_falseAlarmRates, ...
    'tpr', perf_detectionRates, 'thresholds', T_thresholds_perf); % Store FAPS vs TPR data
metrics.totalAudioDuration_sec = totalAudioDuration;
metrics.detectionTolerance_sec = detectionTolerance;
metrics.evaluatedResultCount = numResultsDetections; % Count actually used in matching/curves
metrics.groundtruthSource = gtSourceDescription; % Record where GT data came from
metrics.numResultsExcluded_Failures = numFailures; % Track how many original results struct entries were skipped
metrics.numResultsExcluded_NoScore = numRemoved_NoScore; % Track how many valid results were skipped due lack of score
metrics.matchingAlgorithm = 'Hungarian (matchpairs)'; % Indicate algorithm used

end % End of function