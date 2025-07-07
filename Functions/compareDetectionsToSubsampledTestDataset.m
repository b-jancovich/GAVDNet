function [metrics, falsePositives, falseNegatives] = compareDetectionsToSubsampledTestDataset(...
    groundtruthPath, inferenceResultsPath, detectionTolerance, maxDetectionDuration, gtFormat)
% compareDetectionsToSubsampledTestDataset
% Compare GAVDNet detector results with ground truth (detections from 
% SORP or CTBTO data) using Hungarian algorithm via matchpairs and calculate 
% performance metrics.
%
% Inputs:
%   groundtruthPath      - Path to ground truth file (format depends on gtFormat)
%   inferenceResultsPath - Path to inference results .mat file (containing 'results' struct array)
%   detectionTolerance   - Time tolerance in seconds for matching detections
%   maxDetectionDuration - Maximum detection duration in seconds (used for false negatives)
%   gtFormat             - String/char indicating ground truth format: 'SORP' or 'CTBTO'
%                          'SORP': tab-delimited txt file with SORP format
%                          'CTBTO': .mat file containing 'testDatasetDetectionsList' table
%
% Outputs:
%   metrics - Struct containing all calculated performance metrics
%   falsePositives - Struct array (1xN or Nx1) containing details of unmatched detector results
%   falseNegatives - Struct array (1xN or Nx1) containing details of unmatched ground truth detections
%
% Metrics included:
%   - nPositivesDetector: Number of detections in results (before score/time filtering)
%   - nPositivesGT: Number of detections in groundtruth
%   - nTruePositives: Number of true positive detections (based on optimal matching)
%   - nFalsePositives: Number of false positive detections (based on optimal matching)
%   - nFalseNegatives: Number of false negative detections (based on optimal matching)
%   - recall: Recall/Sensitivity score
%   - precision: Precision score
%   - f1Score: F1 score
%   - auc: Area under ROC curve
%   - roc: Struct containing ROC curve data (fpr, tpr, thresholds)
%   - performanceCurve: Struct containing detection performance curve data (faps, tpr, thresholds)
%   - evaluatedResultCount: Number of results actually used for matching (after all filtering)
%   - numResultsExcluded_InferenceFailures: (Now always 0)
%   - numResultsExcluded_NoScoreOrTime: Number of results excluded due to invalid time or score
%   - matchingAlgorithm: Indicates the matching method used ('Hungarian (matchpairs)')
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%% Validate gtFormat input
if ~ismember(upper(gtFormat), {'SORP', 'CTBTO'})
    error('gtFormat must be either ''SORP'' or ''CTBTO''');
end
gtFormat = upper(gtFormat); % Normalize to uppercase

%% Load data
fprintf('Loading ground truth data from: %s (Format: %s)\n', groundtruthPath, gtFormat);

fprintf('Loading inference results from: %s\n', inferenceResultsPath);
resultsData = load(inferenceResultsPath);

%% Process results data

% All entries in resultsData.results are considered for processing.
inferenceResults = resultsData.results;

% Process all 'validResults'
nPositivesDetector = length(inferenceResults);

% Initialize arrays for all potential detections, using NaN for easy filtering later
resultsTimes = NaN(nPositivesDetector, 1);
confidenceScores = NaN(nPositivesDetector, 1);
resultsFileNames = cell(nPositivesDetector, 1);
resultsFileDurations = NaN(nPositivesDetector, 1);
originalIndicesInValidResults = (1:nPositivesDetector)'; % Maps back to 'validResults' (which is resultsData.results)

% Diagnostic counters
diag_missingEventStartTimeField = 0;
diag_emptyEventStartTime = 0;
diag_nonScalarOrNonDatetimeEventStartTime = 0;
diag_NaTEventStartTime = 0;
diag_missingConfidenceField = 0;
diag_emptyConfidence = 0;
diag_nonNumericOrNonScalarConfidence = 0;

% Init output struct
metrics = struct();

for i = 1:nPositivesDetector
    currentEntry = inferenceResults(i);

    % Process eventStartTime
    if ~isfield(currentEntry, 'eventStartTime')
        diag_missingEventStartTimeField = diag_missingEventStartTimeField + 1;
    else
        est = currentEntry.eventStartTime;
        if isempty(est)
            diag_emptyEventStartTime = diag_emptyEventStartTime + 1;
        elseif ~isdatetime(est) || ~isscalar(est)
            diag_nonScalarOrNonDatetimeEventStartTime = diag_nonScalarOrNonDatetimeEventStartTime + 1;
            warning('Entry %d: eventStartTime is not a scalar datetime object (Type: %s, Size: %s). Assigning NaN time.', i, class(est), mat2str(size(est)));
        else
            if isnat(est)
                diag_NaTEventStartTime = diag_NaTEventStartTime + 1;
            end
            resultsTimes(i) = datenum(est);
        end
    end

    % Process confidence
    if ~isfield(currentEntry, 'confidence')
        diag_missingConfidenceField = diag_missingConfidenceField + 1;
    else
        conf = currentEntry.confidence;
        if isempty(conf)
            diag_emptyConfidence = diag_emptyConfidence + 1;
        elseif ~isnumeric(conf) || ~isscalar(conf)
            diag_nonNumericOrNonScalarConfidence = diag_nonNumericOrNonScalarConfidence + 1;
            warning('Entry %d: confidence is not a numeric scalar (Type: %s, Size: %s). Assigning NaN score.', i, class(conf), mat2str(size(conf)));
        else
            confidenceScores(i) = conf;
        end
    end

    % Process fileName
    if isfield(currentEntry, 'fileName') && ischar(currentEntry.fileName)
        resultsFileNames{i} = currentEntry.fileName;
    else
        resultsFileNames{i} = ''; 
        if ~isfield(currentEntry, 'fileName')
            warning('Entry %d: Missing fileName field.',i);
        else
            warning('Entry %d: fileName is not char (Type: %s).',i, class(currentEntry.fileName));
        end
    end
    
    % Process fileDuration
    if isfield(currentEntry, 'fileDuration') && isnumeric(currentEntry.fileDuration) && isscalar(currentEntry.fileDuration)
        resultsFileDurations(i) = currentEntry.fileDuration;
    else
        resultsFileDurations(i) = NaN;
        if ~isfield(currentEntry, 'fileDuration')
            warning('Entry %d: Missing fileDuration field.',i);
        elseif ~isnumeric(currentEntry.fileDuration) || ~isscalar(currentEntry.fileDuration)
             warning('Entry %d: fileDuration is not numeric scalar (Type: %s, Size: %s).',i, class(currentEntry.fileDuration), mat2str(size(currentEntry.fileDuration)));
        end
    end
end

% Report diagnostic information
if diag_missingEventStartTimeField > 0
    fprintf('  %d entries: Missing ''eventStartTime'' field.\n', diag_missingEventStartTimeField)
end
if diag_emptyEventStartTime > 0
    fprintf('  %d entries: ''eventStartTime'' was empty (e.g., 0x0 datetime).\n', diag_emptyEventStartTime)
end
if diag_nonScalarOrNonDatetimeEventStartTime > 0
    fprintf('  %d entries: ''eventStartTime'' not scalar datetime.\n', diag_nonScalarOrNonDatetimeEventStartTime)
end
if diag_NaTEventStartTime > 0
    fprintf('  %d entries: ''eventStartTime'' was NaT (Not-a-Time), resulting in NaN time.\n', diag_NaTEventStartTime)
end
if diag_missingConfidenceField > 0
    fprintf('  %d entries: Missing ''confidence'' field.\n', diag_missingConfidenceField)
end
if diag_emptyConfidence > 0
    fprintf('  %d entries: ''confidence'' was empty.\n', diag_emptyConfidence)
end
if diag_nonNumericOrNonScalarConfidence > 0
    fprintf('  %d entries: ''confidence'' not numeric scalar.\n', diag_nonNumericOrNonScalarConfidence)
end

% Remove entries with NaN time or NaN confidence score
validDataIdx = ~isnan(resultsTimes) & ~isnan(confidenceScores);
numResultsOriginal_beforeNaNFilter = nPositivesDetector; 
numRemoved_NoScoreOrTime = sum(~validDataIdx);

if numRemoved_NoScoreOrTime > 0
    fprintf('Warning: Removing %d result entries due to invalid/missing time or confidence score (see diagnostics above).\n', ...
        numRemoved_NoScoreOrTime);
end

resultsTimes = resultsTimes(validDataIdx);
confidenceScores = confidenceScores(validDataIdx);
resultsFileNames = resultsFileNames(validDataIdx);
resultsFileDurations = resultsFileDurations(validDataIdx);
filteredOriginalIndicesInValidResults = originalIndicesInValidResults(validDataIdx);

nPositivesDetector_Evaluated = length(resultsTimes);

fprintf('Total results entries initially loaded: %d\n', length(resultsData.results));
fprintf('Results entries considered (nPositivesDetector metric source): %d\n', numResultsOriginal_beforeNaNFilter);
fprintf('Results entries after time/score validation (evaluatedResultCount metric): %d\n', nPositivesDetector_Evaluated);

%% Extract detections from groundtruth (format-specific)

if strcmp(gtFormat, 'SORP')
    %% SORP Format Processing
    % Read the tab-delimited txt file
    try
        gtTable = readtable(groundtruthPath, 'Delimiter', '\t', 'ReadVariableNames', true);
        fprintf('Successfully loaded SORP groundtruth file with %d entries\n', height(gtTable));
    catch ME
        error('Failed to read SORP groundtruth file %s: %s', groundtruthPath, ME.message);
    end

    gtSourceDescription = 'SORP tab-delimited file';

    % Check for required columns
    requiredCols = {'Selection', 'BeginFile', 'EndFile', 'BegFileSamp_samples_', 'EndFileSamp_samples_', 'BeginDateTime'};
    missingCols = {};
    for i = 1:length(requiredCols)
        if ~ismember(requiredCols{i}, gtTable.Properties.VariableNames)
            missingCols{end+1} = requiredCols{i};
        end
    end

    if ~isempty(missingCols)
        error('SORP groundtruth file is missing required columns: %s', strjoin(missingCols, ', '));
    end

    % Parse BeginDateTime column (format: "YYYY/MM/DD HH:mm:SS.SSS")
    fprintf('Parsing BeginDateTime column...\n');
    beginDateTimeStrs = gtTable.BeginDateTime;
    gtTimes = NaN(height(gtTable), 1);

    for i = 1:height(gtTable)
        dateTimeStr = beginDateTimeStrs{i};
        try
            % Parse the datetime string with format "YYYY/MM/DD HH:mm:SS.SSS"
            dt = datetime(dateTimeStr, 'InputFormat', 'yyyy/M/d HH:mm:ss.SSS');
            gtTimes(i) = datenum(dt);
        catch ME
            warning('Could not parse BeginDateTime "%s" for row %d: %s', dateTimeStr, i, ME.message);
            % gtTimes(i) remains NaN
        end
    end

    % Remove entries with unparseable datetime
    validGtIdx = ~isnan(gtTimes);
    numInvalidDateTime = sum(~validGtIdx);
    if numInvalidDateTime > 0
        fprintf('Warning: Removing %d groundtruth entries with invalid BeginDateTime format.\n', numInvalidDateTime);
        gtTable = gtTable(validGtIdx, :);
        gtTimes = gtTimes(validGtIdx);
    end

    % Extract filenames from BeginFile column
    gtFilenames = gtTable.BeginFile;

    % Convert gtTimes to column vector
    gtTimes = gtTimes(:);
    gtFilenames = gtFilenames(:);

    % Handle sample rates - get from first unique file
    fprintf('Determining sample rates from audio files...\n');
    uniqueFilenames = unique(gtFilenames);
    if ~isempty(uniqueFilenames)
        try
            % Get directory path from groundtruth file
            [gtDir, ~, ~] = fileparts(groundtruthPath);
            firstAudioFile = fullfile(gtDir, 'wav', uniqueFilenames{1});
            
            % Try to read audio info
            if exist(firstAudioFile, 'file')
                audioFileInfo = audioinfo(firstAudioFile);
                firstFileFs = audioFileInfo.SampleRate;
            else
                % If file not in same directory, just use a common sample rate
                warning('Could not find audio file %s. Using default sample rate of 2000 Hz.', firstAudioFile);
                firstFileFs = 2000;
            end
        catch ME
            warning(ME.identifier, 'Could not get sample rate from audio file: %s. Using default of 2000 Hz.', ME.message);
            firstFileFs = 2000;
        end
    else
        firstFileFs = 2000;
    end

    % Assume all files have same sample rate
    gtFs = repmat(firstFileFs, length(gtTimes), 1);

elseif strcmp(gtFormat, 'CTBTO')
    %% CTBTO Format Processing
    groundtruthData = load(groundtruthPath);
    
    if ~isfield(groundtruthData, 'testDatasetDetectionsList')
        error('Ground truth .mat file must contain a table named ''testDatasetDetectionsList''.');
    end
    gtTable = groundtruthData.testDatasetDetectionsList;
    gtSourceDescription = 'testDatasetDetectionsList';

    if ismember('datenum', gtTable.Properties.VariableNames)
        gtTimes = gtTable.datenum;
    else
        error('Ground truth table ''testDatasetDetectionsList'' is missing ''datenum'' column.');
    end

    if ismember('filenames', gtTable.Properties.VariableNames)
        gtFilenames = gtTable.filenames;
    else
        gtFilenames = repmat({''}, height(gtTable), 1);
    end

    if ismember('fileFs', gtTable.Properties.VariableNames)
        gtFs = gtTable.fileFs;
    else
        % Get sample rate of first file (assume all same)
        disp('Sample rates not in groundtruth table. Pulling sample rate from first file in GT folder, assuming Fs is uniform.')
        [gtPath, ~, ~] = fileparts(groundtruthPath);
        audioFileInfo = audioinfo(fullfile(gtPath, gtFilenames{1}));
        firstFileFs = audioFileInfo.SampleRate;
        gtFs = repmat(firstFileFs, height(gtTable), 1);
    end

    gtTimes = gtTimes(:);
    gtFilenames = gtFilenames(:);
    gtFs = gtFs(:);
end

nPositivesGT = length(gtTimes);
fprintf('Number of ground truth detections: %d (from %s)\n', ...
    nPositivesGT, gtSourceDescription);

if nPositivesDetector_Evaluated == 0 && nPositivesGT == 0
    warning('Both evaluated results and ground truth have zero detections. Metrics will be NaN/empty.');
    metrics.nPositivesGT = 0;
    metrics.nPositivesDetector = numResultsOriginal_beforeNaNFilter;
    metrics.nTruePositives = 0;
    metrics.nFalsePositives = 0;
    metrics.nFalseNegatives = 0;
    metrics.recall = NaN;
    metrics.sensitivity = NaN;
    metrics.precision = NaN;
    metrics.f1Score = NaN;
    metrics.auc = NaN;
    metrics.roc = struct('fpr', NaN, 'tpr', NaN, 'thresholds', NaN);
    metrics.performanceCurve = struct('faps', NaN, 'tpr', NaN, 'thresholds', NaN);
    metrics.totalAudioDuration_sec = 0;
    metrics.detectionTolerance_sec = detectionTolerance;
    metrics.evaluatedResultCount = 0;
    metrics.groundtruthSource = gtSourceDescription;
    metrics.numResultsExcluded_InferenceFailures = 0; % Always 0 now
    metrics.numResultsExcluded_NoScoreOrTime = numRemoved_NoScoreOrTime;
    metrics.matchingAlgorithm = 'Hungarian (matchpairs)';

    falsePositives = struct('DetectionStartTime', {}, ...
        'DetectionEndTime', {}, 'DetectionStartSamp', [], ...
        'DetectionEndSamp', [], 'AudioFilename', {}, ...
        'AudioFs', [], 'Confidence', [], ...
        'probabilities', []);
    falseNegatives = struct('DetectionStartTime', {}, ...
        'DetectionEndTime', {}, 'DetectionStartSamp', [], ...
        'DetectionEndSamp', [], 'AudioFilename', {}, 'AudioFs', []);
    return;
end

%% Optimal Matching Logic (Hungarian Algorithm via matchpairs)
fprintf('Matching results to ground truth using Optimal Assignment (Hungarian/matchpairs) with %.2f s tolerance...\n', ...
    detectionTolerance);
toleranceInDays = detectionTolerance / (24 * 60 * 60);
nTruePositives = 0;

if nPositivesDetector_Evaluated > 0 && nPositivesGT > 0
    costMatrix = Inf(nPositivesDetector_Evaluated, nPositivesGT);
    for i = 1:nPositivesDetector_Evaluated
        timeDiffs = abs(resultsTimes(i) - gtTimes);
        validGtIndicesForMatch = find(timeDiffs <= toleranceInDays);
        if ~isempty(validGtIndicesForMatch)
            costMatrix(i, validGtIndicesForMatch) = timeDiffs(validGtIndicesForMatch);
        end
    end
    costOfNonAssignment = toleranceInDays; 
    [matches, ~, ~] = matchpairs(costMatrix, costOfNonAssignment);
    
    if ~isempty(matches)
        nTruePositives = size(matches, 1); 
        matchedResultIndices = matches(:, 1); 
        matchedGtIndices = matches(:, 2);   
    else
        nTruePositives = 0;
        matchedResultIndices = [];
        matchedGtIndices = [];
    end
    fprintf('Optimal assignment found %d matches (True Positives).\n', nTruePositives);
else
    fprintf('Skipping matching: Zero evaluated results or zero ground truth detections.\n');
    matchedResultIndices = []; 
    matchedGtIndices = [];   
end
resultLabels = zeros(nPositivesDetector_Evaluated, 1); 

if ~isempty(matchedResultIndices) 
    resultLabels(matchedResultIndices) = 1; 
end
nFalsePositives = nPositivesDetector_Evaluated - nTruePositives;
nFalseNegatives = nPositivesGT - nTruePositives;

fprintf('\nPerformance Metrics (Optimal Matching)\n');
fprintf('Total Ground Truth Detections: %d\n', nPositivesGT);
fprintf('Total Result Entries Considered: %d\n', numResultsOriginal_beforeNaNFilter);
fprintf('Result Detections Evaluated (with valid score/time): %d\n', nPositivesDetector_Evaluated);
fprintf('True Positives (TP):  %d\n', nTruePositives);
fprintf('False Positives (FP): %d\n', nFalsePositives);
fprintf('False Negatives (FN): %d\n', nFalseNegatives);
fprintf('---------------------------\n');

if nPositivesGT > 0 
    recall = nTruePositives / nPositivesGT;
else
    recall = NaN; 
    if nTruePositives == 0, fprintf('Recall is NaN (no ground truth).\n'); end
end
sensitivity = recall;

if nPositivesDetector_Evaluated > 0 
    precision = nTruePositives / nPositivesDetector_Evaluated;
else
    precision = NaN;
    if nTruePositives == 0, fprintf('Precision is NaN (no evaluated results).\n'); end
end

if (precision + recall) > 0 && ~isnan(precision) && ~isnan(recall)
    f1Score = 2 * (precision * recall) / (precision + recall);
else
    f1Score = NaN;
    if isnan(precision) || isnan(recall), fprintf('F1 Score is NaN.\n'); end
end
fprintf('Recall / Sensitivity (TPR): %.4f\n', recall);
fprintf('Precision          (PPV): %.4f\n', precision);
fprintf('F1 Score                : %.4f\n', f1Score);
fprintf('---------------------------\n');

%% Create False Positives Structure

fprintf('Creating false positives structure...\n');

if nFalsePositives > 0
    fpResultIndices_in_evaluated_list = find(resultLabels == 0); 
    fpCount = length(fpResultIndices_in_evaluated_list);

    fpFieldNames = {'DetectionStartTime', 'DetectionEndTime', 'DetectionStartSamp', 'DetectionEndSamp', 'AudioFilename', 'AudioFs', 'Confidence', 'probabilities'};
    fpCellArray = cell(fpCount, length(fpFieldNames));
    falsePositives = cell2struct(fpCellArray, fpFieldNames, 2);
    
    for i = 1:fpCount 
        idxInEvaluatedList = fpResultIndices_in_evaluated_list(i); 
        originalIdxInValidResultsArray = filteredOriginalIndicesInValidResults(idxInEvaluatedList);
        currentResultStruct = inferenceResults(originalIdxInValidResultsArray); % validResults is resultsData.results

        if isfield(currentResultStruct, 'eventStartTime') && isdatetime(currentResultStruct.eventStartTime) && isscalar(currentResultStruct.eventStartTime)
            falsePositives(i).DetectionStartTime = currentResultStruct.eventStartTime; 
        else 
            falsePositives(i).DetectionStartTime = NaT;
        end
        if isfield(currentResultStruct, 'eventEndTime') && isdatetime(currentResultStruct.eventEndTime) && isscalar(currentResultStruct.eventEndTime)
            falsePositives(i).DetectionEndTime = currentResultStruct.eventEndTime; 
        else 
            falsePositives(i).DetectionEndTime = NaT; 
        end
        if isfield(currentResultStruct, 'eventSampleStart') && isnumeric(currentResultStruct.eventSampleStart) && isscalar(currentResultStruct.eventSampleStart)
            falsePositives(i).DetectionStartSamp = currentResultStruct.eventSampleStart; 
        else 
            falsePositives(i).DetectionStartSamp = NaN; 
        end
        if isfield(currentResultStruct, 'eventSampleEnd') && isnumeric(currentResultStruct.eventSampleEnd) && isscalar(currentResultStruct.eventSampleEnd)
            falsePositives(i).DetectionEndSamp = currentResultStruct.eventSampleEnd; 
        else 
            falsePositives(i).DetectionEndSamp = NaN; 
        end
        if isfield(currentResultStruct, 'fileName') && ischar(currentResultStruct.fileName)
            falsePositives(i).AudioFilename = currentResultStruct.fileName; 
        else 
            falsePositives(i).AudioFilename = ''; 
        end
        if isfield(currentResultStruct, 'fileFs') && isnumeric(currentResultStruct.fileFs) && isscalar(currentResultStruct.fileFs)
            falsePositives(i).AudioFs = currentResultStruct.fileFs; 
        else 
            falsePositives(i).AudioFs = NaN; 
        end
        if isfield(currentResultStruct, 'confidence') && isnumeric(currentResultStruct.confidence) && isscalar(currentResultStruct.confidence)
            falsePositives(i).Confidence = currentResultStruct.confidence; 
        else 
            falsePositives(i).Confidence = NaN; 
        end
        if isfield(currentResultStruct, 'probabilities') && isnumeric(currentResultStruct.probabilities)
            falsePositives(i).probabilities = currentResultStruct.probabilities; 
        else 
            falsePositives(i).probabilities = []; 
        end
    end
    fprintf('Created false positives struct array with %d entries.\n', fpCount);
else
    falsePositives = struct('DetectionStartTime', {}, 'DetectionEndTime', {}, 'DetectionStartSamp', [], 'DetectionEndSamp', [], 'AudioFilename', {}, 'AudioFs', [], 'Confidence', [], 'probabilities', []);
    fprintf('No false positives found (nFalsePositives calculated as %d).\n', nFalsePositives);
end

%% Create False Negatives Structure
fprintf('Creating false negatives structure...\n');
if nFalseNegatives > 0
    allGtIndices = 1:nPositivesGT;
    fnGtIndices = setdiff(allGtIndices, matchedGtIndices); 
    fnCount = length(fnGtIndices);

    fnFieldNames = {'DetectionStartTime', 'DetectionEndTime', 'DetectionStartSamp', 'DetectionEndSamp', 'AudioFilename', 'AudioFs'};
    fnCellArray = cell(fnCount, length(fnFieldNames));
    falseNegatives = cell2struct(fnCellArray, fnFieldNames, 2); 
    
    for i = 1:fnCount
        idxInGt = fnGtIndices(i);
        
        fnStartTime = datetime(gtTimes(idxInGt), 'ConvertFrom', 'datenum');
        falseNegatives(i).DetectionStartTime = fnStartTime;
        falseNegatives(i).DetectionEndTime = fnStartTime + seconds(maxDetectionDuration);
        
        currentGtFilename = '';
        if idxInGt <= length(gtFilenames) && ~isempty(gtFilenames{idxInGt})
            currentGtFilename = gtFilenames{idxInGt};
        end
        falseNegatives(i).AudioFilename = currentGtFilename;
        
        currentGtFs = NaN;
        if idxInGt <= length(gtFs) && ~isnan(gtFs(idxInGt)) && gtFs(idxInGt) > 0
            currentGtFs = gtFs(idxInGt);
        else
            if ~isempty(currentGtFilename)
                try
                    [gtAudioPath, ~, ~] = fileparts(groundtruthPath); 
                    audioFileInfo = audioinfo(fullfile(gtAudioPath, currentGtFilename));
                    currentGtFs = audioFileInfo.SampleRate;
                    warning('FN %d (GT index %d): Used Fs from audioinfo for %s as GT table Fs was missing/invalid.', i, idxInGt, currentGtFilename);
                catch ME_audioinfo
                    warning('FN %d (GT index %d): Could not get Fs from audioinfo for %s (Error: %s). Samples will be NaN.', i, idxInGt, currentGtFilename, ME_audioinfo.message);
                end
            else
                 warning('FN %d (GT index %d): GT Fs missing/invalid and no filename to get Fs from audioinfo. Samples will be NaN.', i, idxInGt);
            end
        end
        falseNegatives(i).AudioFs = currentGtFs;
        
        % Format-specific sample calculation
        if strcmp(gtFormat, 'SORP')
            % Use sample indices directly from SORP data if available
            if idxInGt <= height(gtTable)
                try
                    begSamp = gtTable.BegFileSamp_samples_(idxInGt);
                    endSamp = gtTable.EndFileSamp_samples_(idxInGt);
                    if isnumeric(begSamp) && isnumeric(endSamp) && ~isnan(begSamp) && ~isnan(endSamp)
                        falseNegatives(i).DetectionStartSamp = begSamp;
                        falseNegatives(i).DetectionEndSamp = endSamp;
                    else
                        falseNegatives(i).DetectionStartSamp = NaN;
                        falseNegatives(i).DetectionEndSamp = NaN;
                    end
                catch
                    falseNegatives(i).DetectionStartSamp = NaN;
                    falseNegatives(i).DetectionEndSamp = NaN;
                end
            else
                falseNegatives(i).DetectionStartSamp = NaN;
                falseNegatives(i).DetectionEndSamp = NaN;
            end
        elseif strcmp(gtFormat, 'CTBTO')
            % Use extractDatetimeFromFilename for CTBTO format
            falseNegatives(i).DetectionStartSamp = NaN;
            falseNegatives(i).DetectionEndSamp = NaN;

            if ~isempty(currentGtFilename) && ~isnan(currentGtFs)
                try
                    % Assumes extractDatetimeFromFilename function is available on MATLAB path
                    fileStartTime = extractDatetimeFromFilename(currentGtFilename, 'datetime'); 
                    if ~isnat(fileStartTime)
                        timeDiffSec = seconds(fnStartTime - fileStartTime);
                        if timeDiffSec >= 0 
                            startSamp = round(timeDiffSec * currentGtFs) + 1; 
                            falseNegatives(i).DetectionStartSamp = startSamp;
                            falseNegatives(i).DetectionEndSamp = startSamp + round(maxDetectionDuration * currentGtFs) -1;
                        else
                            warning('False Negative %d (GT index %d): Detection start time is before extracted file start time for %s. Samples set to NaN.', i, idxInGt, currentGtFilename);
                        end
                    else
                        warning('False Negative %d (GT index %d): Could not extract datetime from filename %s. Samples set to NaN.', i, idxInGt, currentGtFilename);
                    end
                catch ME
                    warning('False Negative %d (GT index %d): Error processing filename %s for sample calculation: %s. Samples set to NaN.', i, idxInGt, currentGtFilename, ME.message);
                end
            end
        end
    end
    fprintf('Created false negatives struct array with %d entries.\n', fnCount);
else
    falseNegatives = struct('DetectionStartTime', {}, 'DetectionEndTime', {}, 'DetectionStartSamp', [], 'DetectionEndSamp', [], 'AudioFilename', {}, 'AudioFs', []);
    fprintf('No false negatives found.\n');
end

%% Calculate ROC curve and AUC using perfcurve
fprintf('\nCalculating ROC curve and AUC...\n');
X_roc = NaN; Y_tpr = NaN; T_thresholds_roc = NaN; AUC = NaN;

if nPositivesDetector_Evaluated > 0 && nPositivesGT > 0 && ~isempty(confidenceScores) && ~isempty(resultLabels)
    uniqueLabels = unique(resultLabels);
    if length(uniqueLabels) == 2 
        [X_roc, Y_tpr, T_thresholds_roc, AUC] = perfcurve(resultLabels, confidenceScores, 1, 'XCrit', 'fpr', 'YCrit', 'tpr');
        fprintf('AUC calculated: %.4f\n', AUC);
    elseif all(uniqueLabels == 1) && ~isempty(uniqueLabels)
        warning('All %d evaluated results were TPs. ROC is degenerate (0,1). AUC = 1.0.', nPositivesDetector_Evaluated);
        AUC = 1.0; X_roc = [0; 0]; Y_tpr = [0; 1]; 
        if ~isempty(confidenceScores)
            T_thresholds_roc = [max(confidenceScores)+eps(max(confidenceScores)); min(confidenceScores)-eps(min(confidenceScores))];
        else
            T_thresholds_roc = [1;0]; 
        end
    elseif all(uniqueLabels == 0) && ~isempty(uniqueLabels)
        warning('All %d evaluated results were FPs. ROC is degenerate (1,0). AUC = 0.0.', nPositivesDetector_Evaluated);
        AUC = 0.0; X_roc = [0; 1]; Y_tpr = [0; 0]; 
        if ~isempty(confidenceScores)
            T_thresholds_roc = [max(confidenceScores)+eps(max(confidenceScores)); min(confidenceScores)-eps(min(confidenceScores))];
        else
            T_thresholds_roc = [1;0];
        end
    else
        warning('Cannot calculate ROC: Not enough diversity in labels or data. Labels: %s', mat2str(uniqueLabels));
    end
elseif nPositivesDetector_Evaluated == 0
    warning('Cannot calculate ROC: No evaluated results detections with scores.');
elseif nPositivesGT == 0
    warning('Cannot calculate ROC: No ground truth detections. AUC is undefined.');
end

%% Apply Temperature Scaling Calibration
fprintf('\nApplying temperature scaling calibration...\n');

if nPositivesDetector_Evaluated > 0 && any(resultLabels == 1) && any(resultLabels == 0)

    % Apply temperature scaling calibration
    [optimalTemperature, calibratedConfidences] = calculateOptimalTemperature(confidenceScores, resultLabels);
    
    % Update confidence scores with calibrated values
    calibratedConfidenceScores = calibratedConfidences;
    
    % Recalculate ROC curve and AUC with calibrated scores
    fprintf('Recalculating metrics with calibrated confidence scores...\n');
    if ~isempty(calibratedConfidenceScores) && ~isempty(resultLabels)
        uniqueLabels = unique(resultLabels);
        if length(uniqueLabels) == 2 
            [~, ~, ~, AUC_cal] = perfcurve(resultLabels, calibratedConfidenceScores, 1, 'XCrit', 'fpr', 'YCrit', 'tpr');
            fprintf('Calibrated AUC: %.4f (Original: %.4f)\n', AUC_cal, AUC);
        end
    end
    
    % Store calibration results in metrics
    metrics.temperatureScaling.optimalTemperature = optimalTemperature;
    metrics.temperatureScaling.originalAUC = AUC;
    metrics.temperatureScaling.calibratedAUC = AUC_cal;
    metrics.temperatureScaling.calibratedConfidences = calibratedConfidenceScores;
else
    fprintf('Skipping temperature scaling: insufficient data diversity for calibration.\n');
end

%% Calculate and Plot Detection Performance Curve (TPR vs False Alarms Per Second)

fprintf('\nCalculating Detection Performance Curve (TPR vs FAPS)...\n');
totalAudioDuration = 0;
uniqueFileDurationsMap = containers.Map('KeyType','char','ValueType','double');
if ~isempty(resultsFileNames) 
    for i = 1:length(resultsFileNames)
        fileName = resultsFileNames{i};
        fileDuration = resultsFileDurations(i);
        if ~isempty(fileName) && ~isnan(fileDuration) && fileDuration > 0
            uniqueFileDurationsMap(fileName) = fileDuration;
        end
    end
    if ~isempty(keys(uniqueFileDurationsMap)) 
        totalAudioDuration = sum(cell2mat(values(uniqueFileDurationsMap)));
        fprintf('Total duration from %d unique files in evaluated results: %.2f sec (%.2f hr)\n', ...
            length(uniqueFileDurationsMap), totalAudioDuration, totalAudioDuration/3600);
    end
end

% Fallback duration calculation for CTBTO format
if totalAudioDuration <= 0 && strcmp(gtFormat, 'CTBTO') && exist('groundtruthData', 'var') && isfield(groundtruthData, 'testDatasetFileList')
    fprintf('Attempting total duration from ground truth file list (testDatasetFileList).\n');
    gtTableForDuration = groundtruthData.testDatasetFileList; 
    if ismember('duration', gtTableForDuration.Properties.VariableNames) && isnumeric(gtTableForDuration.duration)
        validDurs = gtTableForDuration.duration(~isnan(gtTableForDuration.duration) & gtTableForDuration.duration > 0);
        totalAudioDuration = sum(validDurs);
        fprintf('Total duration from ground truth file list: %.2f sec (%.2f hr)\n', totalAudioDuration, totalAudioDuration/3600);
    else
        fprintf('Warning: No valid ''duration'' column in testDatasetFileList for fallback duration calculation.\n');
    end
end

perf_falseAlarmRates = NaN; perf_detectionRates = NaN; T_thresholds_perf = NaN;
if totalAudioDuration > 0 && nPositivesGT > 0 && nPositivesDetector_Evaluated > 0 && ~isempty(confidenceScores) && ~isempty(resultLabels)
    
    if length(unique(resultLabels)) < 1 
        warning('Not enough valid score/label pairs for performance curve calculation.');
    else
        uniqueScoresSorted = sort(unique(confidenceScores), 'descend'); 
        if isempty(uniqueScoresSorted)
             warning('No unique non-NaN scores for performance curve.');
        else
            T_thresholds_perf = [uniqueScoresSorted(1)+eps(uniqueScoresSorted(1)); uniqueScoresSorted; uniqueScoresSorted(end)-eps(uniqueScoresSorted(end))];
            T_thresholds_perf = unique(T_thresholds_perf); 
            T_thresholds_perf = sort(T_thresholds_perf, 'descend');

            perf_detectionRates = zeros(length(T_thresholds_perf), 1); 
            perf_falseAlarmRates = zeros(length(T_thresholds_perf), 1); 

            for i = 1:length(T_thresholds_perf)
                thr = T_thresholds_perf(i);
                aboveThresholdIdx = (confidenceScores >= thr);
                
                tpAtThreshold = sum(resultLabels(aboveThresholdIdx) == 1);
                fpAtThreshold = sum(resultLabels(aboveThresholdIdx) == 0);

                if nPositivesGT > 0
                    perf_detectionRates(i) = tpAtThreshold / nPositivesGT;
                else
                    perf_detectionRates(i) = NaN;
                end
                perf_falseAlarmRates(i) = fpAtThreshold / totalAudioDuration; 
            end
            fprintf('Calculated FAPS vs TPR curve.\n');
        end
    end
else
    warning('Cannot calculate Detection Performance Curve (TPR vs FAPS) due to missing data (duration, GT, evaluated results, scores, or labels).');
end
fprintf('Total Audio Duration for FAPS: %.2f seconds (%.2f hours)\n', ...
    totalAudioDuration, totalAudioDuration/3600);

%% Analyse Confidence Score Distribution

% Plot confidence distribution
[confPercentiles, percentiles] = analyseConfidenceDistribution( ...
    resultsData.results, false);

% Print confidence distribution percentiles
fprintf('\nConfidence Percentiles:\n');
for i = 1:length(percentiles)
    fprintf('%d%%: %.6f\n', percentiles(i), confPercentiles(i));
end

%% Prepare output metrics structure

metrics.nPositivesGT = nPositivesGT;
metrics.nPositivesDetector = numResultsOriginal_beforeNaNFilter; 
metrics.nTruePositives = nTruePositives;
metrics.nFalsePositives = nFalsePositives;
metrics.nFalseNegatives = nFalseNegatives;
metrics.recall = recall;
metrics.sensitivity = sensitivity;
metrics.precision = precision;
metrics.f1Score = f1Score;
metrics.auc = AUC;
metrics.roc = struct('fpr', X_roc, 'tpr', Y_tpr, 'thresholds', T_thresholds_roc);
metrics.performanceCurve = struct('faps', perf_falseAlarmRates, 'tpr', perf_detectionRates, 'thresholds', T_thresholds_perf);
metrics.totalAudioDuration_sec = totalAudioDuration;
metrics.detectionTolerance_sec = detectionTolerance;
metrics.evaluatedResultCount = nPositivesDetector_Evaluated; 
metrics.numResultsExcluded_NoScoreOrTime = numRemoved_NoScoreOrTime; 
metrics.groundtruthSource = gtSourceDescription;
metrics.numResultsExcluded_InferenceFailures = 0; % Always 0 now
metrics.confidenceDistribution.confPercentiles = confPercentiles;
metrics.confidenceDistribution.percentiles = percentiles;
metrics.matchingAlgorithm = 'Hungarian (matchpairs)';

end
