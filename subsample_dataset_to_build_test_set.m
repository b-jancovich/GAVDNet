%% subsample_dataset_to_build_test_set.m
% Create a stratified test dataset for automated animal call detector.
%
% This script creates a balanced, representative test dataset for evaluating 
% an automated animal call detector by stratified subsampling from a 
% larger dataset. This larger dataset is itself a single year of data from a 
% decade-scale study of the Chagos pygmy blue whale song [1]. The dataset 
% takes the form of MATLAB files containing a list of whale song detections
% with timestamps and metadata, isolated from audio recordings 
% captured by the Comprehensive Nuclear Test Ban Treaty Organisation's
% International Monitoring System. The dataset is open access.
% Access to the hydroacoustic data from the CTBTO hydrophone station HA08 
% operated by USA at Chagos Archipelago was made available to the authors 
% by the CTBTO's Virtual Data Exploitation Centre under contract.
%
% The original detections in [1] were made using a Sparse Representation 
% Detector, as described in [2]. False positives were removed manually by 
% the authors of [1]. The detector in [2] returns estimates of SNR and 
% SINR for each detection, and these are used in the subsampling process.
% 
% Subsampling process:
%
% 1. LOADS SOURCE DATA:
%    - Loads the detections the existing dataset from [1]
%    - EITHER:
%           - Matches detections to their corresponding audio files
%           - Extracts metadata (SNR, SINR, time of day, month, etc.)
%    - OR:
%           - Loads detections with pre-retrieved metadata saved in a
%           previous execution of the script
%
% 2. FILTERS THE DATASET:
%    - Removes detections whose audio files cannot be read, or do not
%      contain valid audio data.
%    - Filters out files with duration shorter than 20% of mean file duration
%    - Identifies files with and without detections
%
% 3. CREATES STRATIFIED BINS based on multiple dimensions:
%    - Signal-to-Noise Ratio (SNR)
%    - Signal-to-Interference-plus-Noise Ratio (SINR)
%    - Hour of day (temporal coverage)
%    - Month of year (seasonal coverage)
%
% 4. PERFORMS STRATIFIED SAMPLING:
%    - Samples proportionally from each bin to maintain dataset distribution
%    - Prioritizes bins with higher representation in the original dataset
%    - Samples until target duration for files with detections is reached
%    - Adds some audio files without detections based on propNoDetections 
%      input argument.
%
% 5. CREATES TEST DATASET:
%    - Builds detection-level and file-level tables containing detection
%      list and metadata
%    - Saves tables to the output directory
%    - Copies audio files to the output directory
%
% PARAMETERS:
%   audioPath         - Directory containing source audio files
%   detectionsPath    - Path to the detections .mat file
%   outputDir         - Directory where the test dataset will be saved
%   targetDuration    - Desired duration of the test dataset in hours
%   propNoDetections  - Proportion of dataset that should contain no detections
%   snrBins           - Number of bins for stratifying by SNR
%   sinrBins          - Number of bins for stratifying by SINR
%
% The stratified sampling ensures the test dataset is representative of the
% original dataset's SNR/SINR distribution, temporal patterns, and seasonal
% patterns, making it suitable for robust detector performance evaluation.
%
% References:
%   [1] Leroy, Emmanuelle C., Jean-Yves Royer, Abigail Alling, Ben Maslen, 
%       and Tracey L. Rogers. “Multiple Pygmy Blue Whale Acoustic Populations 
%       in the Indian Ocean: Whale Song Identifies a Possible New Population.” 
%       Scientific Reports 11, no. 1 (December 2021): 8762. 
%       https://doi.org/10.1038/s41598-021-88062-5.
%   [2] Socheleau, F.-X., & Samaran, F. (2017). Detection of Mysticete Calls: 
%       A Sparse Representation-Based Approach (Research Report RR-2017-04-SC). 
%       Dépt. Signal et Communications (Institut Mines-Télécom-IMT Atlantique-UBL);
%       https://hal.archives-ouvertes.fr/hal-01736178
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%%
clear 
close all
clc

%% Inputs

% Audio directory
audioPath = "D:\Diego Garcia South\DiegoGarcia2007\wav";

% Detections directory
detectionsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\Manue_Chagos_RawData\DGS\detections_H08S1_DiegoGarciaS_2007.mat";

% Directory to save the test dataset
outputDir = "D:\GAVDNet\Chagos_DGS\Test Data\2007subset_small";

% Test dataset target duration (hours)
targetDuration = 50; 

% Proportion of dataset audio files with no detections (e.g., 10%)
propNoDetections = 0.1;

% Number of bins for SNR sorting
snrBins = 5; 

% Number of bins for SINR sorting
sinrBins = 5; 

%% Add dependencies

addpath(fullfile(pwd, "Functions"));

%% Execute the dataset creation process

% Get input data and metadata (if retrieving fresh metadata, this will take > 4 hours)
detections = loadOrCreateMetadata(detectionsPath, audioPath);

% Filter the dataset
[detections, uniqueDetectionFiles, filesWithoutDetections, meanaudioDuration] = filterDataset(....
    detections, audioPath);

% Calculate required durations
[secondsWithDetections, secondsWithoutDetections] = calculateRequiredDurations(...
    targetDuration, propNoDetections);

% Create table of unique files with metadata
uniqueFiles = createUniqueFilesTable(detections, uniqueDetectionFiles);

% Sort detections into bins for SNR, SINR, Hour, Month
[strataBins, binProps, binCounts] = createStrataAndBins(uniqueFiles, snrBins, sinrBins);

% Sample files with detections from bins proportionally
[selectedFilesWithDetections, selectedDetections, currentDuration] = sampleFilesFromBins(...
    strataBins, binProps, binCounts, uniqueFiles, detections, secondsWithDetections);

% Sample files without detections
selectedFilesWithoutDetections = selectFilesWithoutDetections(...
    filesWithoutDetections, audioPath, secondsWithoutDetections, meanaudioDuration);

% Create detection-level and file-level tables
[testDatasetDetectionsList, testDatasetFileList, totalDurationHours] = createDatasetTables(...
    selectedDetections, selectedFilesWithDetections, selectedFilesWithoutDetections, targetDuration);

% Save dataset and copy files
saveAndCopyFiles(testDatasetFileList, testDatasetDetectionsList, outputDir, audioPath);

fprintf('\nTest dataset creation complete.\n');

%% Helper Functions

function detections = loadOrCreateMetadata(detectionsPath, audioPath)
%LOADORCREATEMETADATA Load existing metadata or create it from scratch

    [filePath, fileName, ext] = fileparts(detectionsPath);
    saveName = strcat('metadata_and_', fileName, ext);

    % Get list of audio files
    audioList = dir(fullfile(audioPath, '*.wav'));

    % If the detections table with metadata has already been built and saved,
    if exist(fullfile(filePath, saveName), 'file') == 2
        % load it:
        load(fullfile(filePath, saveName))
    else % retrieve metadata for every detection (this will take 4+ hours)
        % Load detections
        load(detectionsPath, "detections");

        % Convert detections to table
        detections = array2table(detections, 'VariableNames', ...
            {'year', 'day', 'month', 'week', 'datenum', 'snr', 'sinr', 'snrClass'});

        fprintf('Starting detection information retrieval...\n')

        % Go through detections and match up the detections with their
        % corresponding audio files using filename time stamps:
        nDetections = height(detections);
        datenums = detections.datenum;
        dateTimeString = cell(nDetections, 1);
        filenames = cell(nDetections, 1);
        audioValid = zeros(nDetections, 1);
        audioDuration = zeros(nDetections, 1);
        hourBin = zeros(nDetections, 1);
        parfor i = 1:nDetections
            % Get the date time string from the datenum
            dateTimeString{i} = datestr(datenums(i), 'yymmdd-HHMMSS');

            % Get the audio file name whose timestamp most closely precedes
            % the detection datenum:
            filenames{i} = findClosestWav(audioList, dateTimeString{i});

            try
                % Read audio
                [audio, fs] = audioread(fullfile(audioPath, filenames{i}));

                % Check wav file contains valid audio data
                if isValidAudio(audio)
                    % Record audio as valid
                    audioValid(i) = true;

                    % Record file duration
                    audioDuration(i) = length(audio) / fs;

                    % Get time of day
                    % Extract time of day as fraction of a day (0-1)
                    timeFractionalDay = mod(datenums(i), 1);

                    % Put in one of 24 'hour of day' bins
                    hourBin(i) = floor(timeFractionalDay * 24);
                end
            catch
                audioValid(i) = false;
                audioDuration(i) = NaN;
                hourBin(i) = NaN;
            end

            if mod(i, 1000) == 0
                fprintf('Finished retrieving info for detection %g of %g.\n', i, nDetections)
            end
        end

        % Add the new data to the table
        detections = addvars(detections, ...
            filenames, dateTimeString, hourBin, audioValid, audioDuration);

        save(fullfile(filePath, saveName), 'detections');
    end
end

function [detections, uniqueDetectionFiles, filesWithoutDetections, meanaudioDuration] = filterDataset(detections, audioPath)
%FILTERDATASET Filter the dataset to remove invalid entries

    % After the loop, filter out invalid entries
    fprintf('Full dataset size: %d detections\n', height(detections));

    validIndices = table2array(detections(:, 'audioValid') == true);
    detections = detections(validIndices, :);

    % Calculate mean audio duration
    meanaudioDuration = mean(detections.audioDuration, 'omitnan');
    fprintf('Mean audio duration: %.2f seconds\n', meanaudioDuration);

    % Filter out files that are significantly shorter than the mean (< 20% of mean)
    validDurationIndices = detections.audioDuration >= 0.2 * meanaudioDuration;
    detections = detections(validDurationIndices, :);
    fprintf('Dataset size after filtering short/unreadable files: %d detections\n', height(detections));

    % Get unique filenames from the detections
    uniqueDetectionFiles = unique(detections.filenames);
    fprintf('Number of unique files with detections: %d\n', length(uniqueDetectionFiles));

    % Get all audio files
    audioList = dir(fullfile(audioPath, '*.wav'));
    allAudioFiles = {audioList.name}';

    % Find files that don't have detections
    filesWithoutDetections = setdiff(allAudioFiles, uniqueDetectionFiles);
    fprintf('Number of files without detections: %d\n', length(filesWithoutDetections));
end

function [secondsWithDetections, secondsWithoutDetections] = calculateRequiredDurations(targetDuration, propNoDetections)
%CALCULATEREQUIREDDURATION Calculate the required durations for the dataset

    totalHoursNeeded = targetDuration;
    hoursWithDetections = totalHoursNeeded * (1 - propNoDetections);
    hoursWithoutDetections = totalHoursNeeded * propNoDetections;

    % Convert to seconds
    secondsWithDetections = hoursWithDetections * 3600;
    secondsWithoutDetections = hoursWithoutDetections * 3600;
end

function uniqueFiles = createUniqueFilesTable(detections, uniqueDetectionFiles)
%CREATEUNIQUEFILESTABLE Create a table of unique files with their metadata

    uniqueFiles = table();
    uniqueFiles.filenames = uniqueDetectionFiles;
    uniqueFiles.duration = zeros(length(uniqueDetectionFiles), 1);
    uniqueFiles.month = zeros(length(uniqueDetectionFiles), 1);
    uniqueFiles.hourBin = zeros(length(uniqueDetectionFiles), 1);
    uniqueFiles.meanSNR = zeros(length(uniqueDetectionFiles), 1);
    uniqueFiles.meanSINR = zeros(length(uniqueDetectionFiles), 1);
    uniqueFiles.numDetections = zeros(length(uniqueDetectionFiles), 1);

    % Populate the unique files table
    for i = 1:length(uniqueDetectionFiles)
        fileDetIdx = strcmp(detections.filenames, uniqueDetectionFiles{i});
        fileDetections = detections(fileDetIdx, :);
        uniqueFiles.duration(i) = fileDetections.audioDuration(1);
        uniqueFiles.month(i) = mode(fileDetections.month);
        uniqueFiles.hourBin(i) = mode(fileDetections.hourBin);
        uniqueFiles.meanSNR(i) = mean(fileDetections.snr);
        uniqueFiles.meanSINR(i) = mean(fileDetections.sinr);
        uniqueFiles.numDetections(i) = height(fileDetections);
    end
end

function [strataBins, binProps, binCounts] = createStrataAndBins(uniqueFiles, snrBins, sinrBins)
%CREATESTRATAANDBINS Sort detections into bins for SNR, SINR, Hour, Month

    % First, sort files into bins based on these properties
    months = unique(uniqueFiles.month);
    hours = unique(uniqueFiles.hourBin);

    % Create SNR and SINR bin edges (handling Inf, -Inf, NaN values)
    snrValues = uniqueFiles.meanSNR;
    sinrValues = uniqueFiles.meanSINR;

    % Filter out non-finite values for edge calculation
    snrFinite = snrValues(isfinite(snrValues));
    sinrFinite = sinrValues(isfinite(sinrValues));

    % Create SNR edges
    snrEdges = createBinEdges(snrFinite, snrBins);
    
    % Create SINR edges
    sinrEdges = createBinEdges(sinrFinite, sinrBins);

    % Assign each file to a bin, handling special values
    uniqueFiles.snrBin = discretize(uniqueFiles.meanSNR, snrEdges);
    uniqueFiles.sinrBin = discretize(uniqueFiles.meanSINR, sinrEdges);

    % Handle NaN values manually (assign them to a separate bin)
    uniqueFiles.snrBin(isnan(uniqueFiles.meanSNR)) = snrBins + 1;
    uniqueFiles.sinrBin(isnan(uniqueFiles.meanSINR)) = sinrBins + 1;

    % Update the number of bins to account for the additional NaN bin
    snrBins = snrBins + 1;
    sinrBins = sinrBins + 1;

    % Create stratified bins
    strataBins = cell(length(months), length(hours), snrBins, sinrBins);
    for i = 1:height(uniqueFiles)
        m = find(months == uniqueFiles.month(i));
        h = find(hours == uniqueFiles.hourBin(i));
        s = uniqueFiles.snrBin(i);
        si = uniqueFiles.sinrBin(i);

        if ~isnan(s) && ~isnan(si) % Skip if SNR or SINR couldn't be binned
            bin = strataBins{m, h, s, si};
            strataBins{m, h, s, si} = [bin; i];
        end
    end

    % Calculate the proportion of files in each bin
    binCounts = cellfun(@length, strataBins);
    binProps = binCounts ./ sum(binCounts);
end

function [selectedFilesWithDetections, selectedDetections, currentDuration] = sampleFilesFromBins(strataBins, binProps, binCounts, uniqueFiles, detections, secondsWithDetections)
%SAMPLEFILESFROMBINS Sample files from each bin proportionally

    % Initialize selected files and current duration
    selectedIndices = [];
    currentDuration = 0;

    % Sample from each bin proportionally
    nonEmptyBins = find(binCounts > 0);
    [~, sortedBins] = sort(binProps(nonEmptyBins), 'descend'); % Sort by proportion
    nonEmptyBins = nonEmptyBins(sortedBins);

    for binIdx = nonEmptyBins(:)'
        % Calculate number of files to sample from this bin
        binProp = binProps(binIdx);
        binDuration = secondsWithDetections * binProp;

        % Get indices for this bin
        [m, h, s, si] = ind2sub(size(strataBins), binIdx);
        binIndices = strataBins{m, h, s, si};

        % Calculate how many files to sample from this bin
        fileDurations = uniqueFiles.duration(binIndices);
        filesToSample = min(length(binIndices), ...
                            ceil(binDuration / mean(fileDurations)));

        % Randomly sample from this bin
        if filesToSample > 0
            sampledIndices = binIndices(randperm(length(binIndices), filesToSample));
            selectedIndices = [selectedIndices; sampledIndices];
            currentDuration = currentDuration + sum(uniqueFiles.duration(sampledIndices));
        end

        % Check if we've reached the target
        if currentDuration >= secondsWithDetections
            break;
        end
    end

    % Select files with detections
    selectedFilesWithDetections = uniqueFiles(selectedIndices, :);

    % Keep track of the original detections for each selected file
    selectedDetections = table();
    for i = 1:height(selectedFilesWithDetections)
        fileDetIdx = strcmp(detections.filenames, selectedFilesWithDetections.filenames{i});
        if sum(fileDetIdx) > 0
            fileDetections = detections(fileDetIdx, :);

            % Add to the table of selected detections
            selectedDetections = [selectedDetections; fileDetections];
        end
    end
end

function selectedFilesWithoutDetections = selectFilesWithoutDetections(filesWithoutDetections, audioPath, secondsWithoutDetections, meanaudioDuration)
%SELECTFILESWITHOUTDETECTIONS Select files without detections

    selectedFilesWithoutDetections = table();
    currentNoDetectionDuration = 0;

    if ~isempty(filesWithoutDetections) && secondsWithoutDetections > 0
        % Randomly shuffle files without detections
        shuffledFiles = filesWithoutDetections(randperm(length(filesWithoutDetections)));

        % Initialize table for files without detections
        maxNoDetectionFiles = length(shuffledFiles);
        selectedFilesWithoutDetections = table();
        selectedFilesWithoutDetections.filenames = cell(maxNoDetectionFiles, 1);
        selectedFilesWithoutDetections.duration = zeros(maxNoDetectionFiles, 1);
        selectedFilesWithoutDetections.month = zeros(maxNoDetectionFiles, 1);
        selectedFilesWithoutDetections.hourBin = zeros(maxNoDetectionFiles, 1);

        % Try to read files until we reach the target duration
        numSelected = 0;
        for i = 1:length(shuffledFiles)
            try
                [audio, fs] = audioread(fullfile(audioPath, shuffledFiles{i}));
                if isValidAudio(audio) == true
                    fileDuration = length(audio) / fs;

                    % Skip files that are too short
                    if fileDuration < 0.2 * meanaudioDuration
                        continue;
                    end

                    % Add to selected files
                    numSelected = numSelected + 1;
                    selectedFilesWithoutDetections.filenames{numSelected} = shuffledFiles{i};
                    selectedFilesWithoutDetections.duration(numSelected) = fileDuration;

                    % Extract time information from filenames if possible
                    % Assuming format like 'yymmdd-HHMMSS.wav'
                    nameParts = split(shuffledFiles{i}, '-');
                    if length(nameParts) >= 2
                        % Extract month from the date part
                        dateStr = nameParts{1};
                        if length(dateStr) >= 4
                            monthStr = dateStr(3:4);
                            selectedFilesWithoutDetections.month(numSelected) = str2double(monthStr);
                        end

                        % Extract hour from the time part
                        timeStr = nameParts{2};
                        if length(timeStr) >= 2
                            hourStr = timeStr(1:2);
                            selectedFilesWithoutDetections.hourBin(numSelected) = str2double(hourStr);
                        end
                    end

                    currentNoDetectionDuration = currentNoDetectionDuration + fileDuration;

                    % Check if we've reached the target duration
                    if currentNoDetectionDuration >= secondsWithoutDetections
                        break;
                    end
                end
            catch
                % Skip files that can't be read
            end
        end

        % Trim the table to the actual number of selected files
        selectedFilesWithoutDetections = selectedFilesWithoutDetections(1:numSelected, :);
    end
end

function [testDatasetDetectionsList, testDatasetFileList, totalDurationHours] = createDatasetTables(selectedDetections, selectedFilesWithDetections, selectedFilesWithoutDetections, targetDuration)
%CREATEDATASETTABLES Create detection-level and file-level tables

    % Add hasDetections field to the original detections table
    selectedDetections.hasDetections = true(height(selectedDetections), 1);

    % First, let's get information about the structure of the table
    varNames = selectedDetections.Properties.VariableNames;
    varTypes = cell(1, length(varNames));
    for i = 1:length(varNames)
        if iscell(selectedDetections.(varNames{i}))
            varTypes{i} = 'cell';
        elseif islogical(selectedDetections.(varNames{i}))
            varTypes{i} = 'logical';
        else
            varTypes{i} = 'double';
        end
    end

    % Create a new empty table with all the right variable types
    testDatasetDetectionsList = selectedDetections;

    % Create matrix for the file-level table
    testDatasetFileList = table();
    testDatasetFileList.filenames = [selectedFilesWithDetections.filenames; selectedFilesWithoutDetections.filenames];
    testDatasetFileList.duration = [selectedFilesWithDetections.duration; selectedFilesWithoutDetections.duration];
    testDatasetFileList.month = [selectedFilesWithDetections.month; selectedFilesWithoutDetections.month];
    testDatasetFileList.hourBin = [selectedFilesWithDetections.hourBin; selectedFilesWithoutDetections.hourBin];
    testDatasetFileList.hasDetections = [true(height(selectedFilesWithDetections), 1); false(height(selectedFilesWithoutDetections), 1)];
    testDatasetFileList.meanSNR = [selectedFilesWithDetections.meanSNR; nan(height(selectedFilesWithoutDetections), 1)];
    testDatasetFileList.meanSINR = [selectedFilesWithDetections.meanSINR; nan(height(selectedFilesWithoutDetections), 1)];
    testDatasetFileList.numDetections = [selectedFilesWithDetections.numDetections; zeros(height(selectedFilesWithoutDetections), 1)];

    % Add all the original fields as cell arrays in the file list
    for i = 1:length(varNames)
        varName = varNames{i};
        if ~ismember(varName, {'hasDetections', 'filenames', 'duration', 'month', 'hourBin'})
            testDatasetFileList.([varName '_all']) = cell(height(testDatasetFileList), 1);
        end
    end

    % Populate the cell arrays in the file list
    for i = 1:height(testDatasetFileList)
        filename = testDatasetFileList.filenames{i};

        % Get all corresponding detections
        fileDetIdx = strcmp(testDatasetDetectionsList.filenames, filename);
        fileDetections = testDatasetDetectionsList(fileDetIdx, :);

        % Store each field's values
        for j = 1:length(varNames)
            varName = varNames{j};
            if ~ismember(varName, {'hasDetections', 'filenames', 'duration', 'month', 'hourBin'})
                if ~isempty(fileDetections)
                    testDatasetFileList.([varName '_all']){i} = fileDetections.(varName);
                else
                    % Empty arrays with the right type
                    if strcmp(varTypes{j}, 'cell')
                        testDatasetFileList.([varName '_all']){i} = {};
                    elseif strcmp(varTypes{j}, 'logical')
                        testDatasetFileList.([varName '_all']){i} = false(0, 1);
                    else
                        testDatasetFileList.([varName '_all']){i} = zeros(0, 1);
                    end
                end
            end
        end
    end

    % Calculate total duration
    totalDuration = sum(testDatasetFileList.duration);
    totalDurationHours = totalDuration / 3600;

    % Print summary
    fprintf('\nTest dataset summary:\n');
    fprintf('  - Total files: %d\n', height(testDatasetFileList));
    fprintf('  - Files with detections: %d (%.1f%%)\n', ...
            sum(testDatasetFileList.hasDetections), 100*sum(testDatasetFileList.hasDetections)/height(testDatasetFileList));
    fprintf('  - Files without detections: %d (%.1f%%)\n', ...
            sum(~testDatasetFileList.hasDetections), 100*sum(~testDatasetFileList.hasDetections)/height(testDatasetFileList));
    fprintf('  - Total duration: %.2f hours (target: %.2f hours)\n', ...
            totalDurationHours, targetDuration);
    fprintf('  - Total detections: %d\n', sum(testDatasetFileList.numDetections));
end

function saveAndCopyFiles(testDatasetFileList, testDatasetDetectionsList, outputDir, audioPath)
%SAVEANDCOPYFILES Save the tables and copy the files

    % Save both tables
    save(fullfile(outputDir, 'test_dataset_audiofile_list.mat'), 'testDatasetFileList');
    save(fullfile(outputDir, 'test_dataset_detection_list.mat'), 'testDatasetDetectionsList');

    % Copy files
    fprintf('\nCopying selected files to %s...\n', outputDir);
    uniqueFilenames = testDatasetFileList.filenames;
    for i = 1:length(uniqueFilenames)
        sourceFile = fullfile(audioPath, uniqueFilenames{i});
        destFile = fullfile(outputDir, uniqueFilenames{i});
        try
            copyfile(sourceFile, destFile);
            fprintf('  Copied file %d of %d: %s\n', i, length(uniqueFilenames), uniqueFilenames{i});
        catch e
            fprintf('  Error copying file %s: %s\n', uniqueFilenames{i}, e.message);
        end
    end
end

function edges = createBinEdges(values, numBins)
%CREATEBINEDGES Create bin edges for the data, handling edge cases

    % If we don't have enough finite values, create default bins
    if length(values) < 2
        edges = [-Inf, -20, -10, 0, 10, Inf]; % Default bins if not enough data
    else
        minVal = min(values);
        maxVal = max(values);
        % Add a small buffer to avoid edge cases
        range = maxVal - minVal;
        if range < 1e-6 % If all values are very close together
            minVal = minVal - 1;
            maxVal = maxVal + 1;
        end
        edges = linspace(minVal, maxVal, numBins+1);
        % Add -Inf and Inf as first and last edges to capture all values
        edges = [-Inf, edges(2:end-1), Inf];
    end
end
