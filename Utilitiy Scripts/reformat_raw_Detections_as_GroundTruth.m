%% Groundtruth Cleanup Script
%
% This script reformats detection data from a previous study [1] into a 
% format compatible with the current detector's test system. These
% detections were produced using a sparse representation detector [2]
% and false positives manually removed by E. C. Leroy as part of [1].
%
% The script processes raw detection .mat files, matches each detection 
% with it's corresponding audio file, and handles cases where detections
% span multiple files. The resulting output is intended to be used as
% ground truth.
%
% References:
%   [1] Leroy, E. C., Royer, J.-Y., Alling, A., Maslen, B., & Rogers, T. L. 
%   (2021). Multiple pygmy blue whale acoustic populations in the Indian 
%   Ocean: Whale song identifies a possible new population. Scientific 
%   Reports, 11(1), 8762. https://doi.org/10.1038/s41598-021-88062-5
%
%   [2] F.-X. Socheleau, F. Samaran, "Detection of Mysticete Calls: a Sparse 
%   Representation-Based Approach", IMT Atlantique research report 
%   RR-2017-04-SC, Oct. 2017.
% 
% INPUT:
%   - Raw detections .mat file containing a 'detections' matrix with columns:
%     1: detection year (e.g., 2002)
%     2: detection day (1-365/366)
%     3: detection month (1-12)
%     4: detection week (1-52)
%     5: detection timestamp (MATLAB serial date number)
%     6: SNR estimate of the call
%     7: SINR estimate (Signal to Interference plus Noise Ratio)
%     8: SNR class (floor(SNR/5))
%
% OUTPUT:
%   - Ground truth .mat file containing a 'groundTruth' table with variables:
%     1: 'Selection' - sequential detection number
%     2: 'Fs' - sample rate of audio file in Hz
%     3: 'BeginFile' - filename containing start of the detection
%     4: 'EndFile' - filename containing end of the detection
%     5: 'beginTimeFileSec' - time index of detection start relative to file start (seconds)
%     6: 'endTimeFileSec' - time index of detection end relative to file start (seconds)
%     7: 'detStartDateTimeDT' - datetime of detection start
%     8: 'Dur90__s_' - duration of detection in seconds
%     9: 'audioAvailable' - boolean indicating whether corresponding audio file(s) exist
%
% CONFIGURATION:
%   - rawDetectionsPath: Path to the input .mat file
%   - rawAudioPath: Path to the directory containing raw audio (.wav) files
%   - outputPath: Path where the cleaned ground truth file will be saved
%   - meanCallDuration: Average duration of call in seconds
%   - callDurationVariance: Variation in call duration (± seconds)
%
% REQUIREMENTS:
%   - Requires external helper functions:
%     * extractDatetimeFromFilename() - Extracts datetime from audio filename
%     * find_closest_wav() - Finds the wav file that best matches a detection time
%
% NOTES:
%   - The script employs parallel processing for improved performance
%   - Handles multiple failure modes including missing files and corrupted audio
%   - Output filename is automatically generated from the input filename
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%        
%% Init
clear 
close all
clc

%% User Inputs

% File paths
rawDetectionsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\Manue_Chagos_RawData\DGS\detections_H08S1_DiegoGarciaS_2015.mat";
rawAudioPath = "D:\Diego Garcia South\DiegoGarcia2015\wav";
outputPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\Manue_Chagos_RawData\DGS";

% Target call duration
meanCallDuration = 34.38; % (seconds)
callDurationVariance = 0.4; % (+/- seconds)

%% Process all detections:

% Load detections
rawDetections = load(rawDetectionsPath);

% Set max call duration
estimatedMaxDuration = meanCallDuration + callDurationVariance;

% Extract the detections matrix
if isfield(rawDetections, 'detections')
    detectionsMatrix = rawDetections.detections;
else
    error('The loaded .mat file does not contain a "detections" variable.');
end

% Create empty table with required columns for validateDetectionsFromGroundtruth
groundTruth = table();

% Process detections
numDetections = size(detectionsMatrix, 1);
fprintf('Processing %d detections from ground truth data...\n', numDetections);

% Create arrays to store data for table
Selection = zeros(numDetections, 1);
Fs = NaN(numDetections, 1);  % Changed from zeros to NaN
BeginFile = cell(numDetections, 1);
EndFile = cell(numDetections, 1);
beginTimeFileSec = NaN(numDetections, 1);
endTimeFileSec = NaN(numDetections, 1);
detStartDateTimeDT = NaT(numDetections, 1);
Dur90__s_ = NaN(numDetections, 1);
audioAvailable = false(numDetections, 1);

% Get list of all wav files
wavFiles = dir(fullfile(rawAudioPath, '*.wav'));

% Sort filenames alphanumerically
sortedFilenames = sort({wavFiles.name});

% If we can't find one, clean the raw detections data.

fprintf('Retrieving groundtruth detection times as DateTime objects...\n')
for i = 1:numDetections

    % Set detection index
    Selection(i) = i;

    % Convert MATLAB serial date number to datetime
    detTime = detectionsMatrix(i, 5);
    detStartDateTimeDT(i) = datetime(detTime, 'ConvertFrom', 'datenum');  
end

% Get the average time interval between consecutive wav files
fileStartTimesDT = NaT(length(wavFiles), 1);
for i = 1:length(wavFiles)
    fileStartTimesDT(i) = extractDatetimeFromFilename(wavFiles(i).name);
end
meanInterval = mean(diff(fileStartTimesDT));

% Go get a cup of tea. This bit takes forever...
parfor i = 1:numDetections
    fprintf('Retrieving other groundtruth info for detection# %g...\n', i)

    % Convert the DT to a string
    detectionDateTimeString = string(detStartDateTimeDT(i), 'yyMMdd-HHmmSS');

    % Find the wav file with filename that is closest match to timeString
    matchingFile = find_closest_wav(wavFiles, detectionDateTimeString);

    % Check that the matching file's start time doesn't differ from the
    % target timestring by more than the mean interval beteen files:
    matchingFileStartDT = extractDatetimeFromFilename(matchingFile);
    offset = abs(matchingFileStartDT - detStartDateTimeDT(i));

    % Failure mode: Could not find a wav file with a close wav date time 
    % stamp, or the nearest file was > meanInterval from detection datetime.
    if isempty(matchingFile) || offset > meanInterval
        warning('No matching wav file found for detection at %s', string(detStartDateTimeDT(i)));
        % Use null values
        BeginFile{i} = '';
        EndFile{i} = ''; 
        beginTimeFileSec(i) = NaN;
        endTimeFileSec(i) = NaN;
        Fs(i) = NaN;
        audioAvailable(i) = false;
        Dur90__s_(i) = NaN;
    else
        % Found a matching file
        BeginFile{i} = matchingFile;

        % Calculate time from start of file
        fileStartTimeDT = extractDatetimeFromFilename(matchingFile);
        if ~isnat(fileStartTimeDT)
            timeDiff = seconds(detStartDateTimeDT(i) - fileStartTimeDT);
            beginTimeFileSec(i) = max(0, timeDiff);

            % Estimate end time based on max possible target duration
            endTimeFileSec(i) = beginTimeFileSec(i) + estimatedMaxDuration;
            
            % Set duration
            Dur90__s_(i) = endTimeFileSec(i) - beginTimeFileSec(i);

            % Read the wav file info:
            try
                audioInfo = audioinfo(fullfile(rawAudioPath, matchingFile));
                Fs(i) = audioInfo.SampleRate;
                fileDuration = audioInfo.Duration;
                audioAvailable(i) = true;
            catch ME
                warning('Could not read audio file %s: %s', matchingFile, ME.message);
                fileDuration = 0;
                retry = 1;
                maxTries = 3;
                success = 0;
                while success == 0 && retry <= maxTries
                    fprintf('\tRetry # %d\n', retry)
                    try
                        audioInfo = audioinfo(fullfile(rawAudioPath, matchingFile));
                        Fs(i) = audioInfo.SampleRate;
                        fileDuration = audioInfo.Duration;
                        success = 1;
                        audioAvailable(i) = true;
                    catch
                        success = 0;
                        pause(3)
                        retry = retry+1;
                    end
                end

                if success == 0 % Failure mode: Could not read the begin 
                                % file's audio info. Begin file may be
                                % corrupted.
                    warning('Could not read audio file after %d attempts %s: %s', ...
                        maxTries, matchingFile, ME.message);
                    Fs(i) = NaN;
                    audioAvailable(i) = false;
                    EndFile{i} = '';
                    continue
                end
            end

            % Check whether the detection spans two files
            if endTimeFileSec(i) <= fileDuration
                % Spans a single file
                EndFile{i} = BeginFile{i};

            else % The detection end time is after the file end; 
                % detection spans 2 files:
                % Find the index of the beginFile in the wavFile list:
                beginIdx = find(strcmp(sortedFilenames, BeginFile{i}));
                
                % Get the next file if it exists
                if ~isempty(beginIdx) && beginIdx < length(sortedFilenames)
                    nextFileName = sortedFilenames{beginIdx + 1};

                else % Failure mode: The end file is missing. (There are 
                    % no more wav files in the list)
                    nextFileName = '';
                    EndFile{i} = '';
                    audioAvailable(i) = false;
                    endTimeFileSec(i) = NaN;
                    continue
                end

                % Get the start datetime for that file:
                nextFileStartDT = extractDatetimeFromFilename(nextFileName);

                % Make sure that the next file's start time is not after
                % the detection end time:
                endTimeDT = detStartDateTimeDT(i) + seconds(estimatedMaxDuration);
                validMultiFileDetection = nextFileStartDT < endTimeDT;

                % If we have valid time gaps, use this filename
                if validMultiFileDetection == true
                    EndFile{i} = nextFileName;
                    
                    % Calculate the remaining detection time in the second file
                    % This is the key change needed
                    remainingDuration = seconds(endTimeDT - nextFileStartDT);
                    endTimeFileSec(i) = remainingDuration;
                
                else % Failure mode: The end file is missing (next one in 
                    % the alphanumerically sorted wav file list starts too 
                    % far in the future to contain the end of the
                    % detection.
                    EndFile{i} = '';
                    audioAvailable(i) = false;
                end

            end
            
        else % Failure mode: No valid date time in filename
            warning('Could not extract datetime from filename: %s', matchingFile);
            beginTimeFileSec(i) = NaN;
            endTimeFileSec(i) = NaN;
            EndFile{i} = '';
            Fs(i) = NaN;
            audioAvailable(i) = false;
        end
    end
end

%% Save results to disk

% Create the ground truth table
groundTruth.Selection = Selection;
groundTruth.Fs = Fs;
groundTruth.BeginFile = BeginFile;
groundTruth.EndFile = EndFile;
groundTruth.beginTimeFileSec = beginTimeFileSec;
groundTruth.endTimeFileSec = endTimeFileSec;
groundTruth.detStartDateTimeDT = detStartDateTimeDT;
groundTruth.Dur90__s_ = Dur90__s_;
groundTruth.audioAvailable = audioAvailable;

fprintf('Ground truth data processed. %d valid detections found.\n', ...
    height(groundTruth));

% Set filename
[~, Fn, ~] = fileparts(rawDetectionsPath);
saveNameGT = strcat(Fn, '_cleaned.mat');

% Save the cleaned groundtruth table
save(fullfile(outputPath, saveNameGT), 'groundTruth');





% %% Groundtruth Cleanup Script
% %
% % This Script is designed to reformat the detection (.mat) files generated 
% % by Dr Leroy for her 2019 paper so they can be used made compatible with
% % the detector system in this study. 
% %
% % Input raw detections .mat file structure:
% %     'detections' matrix with columns:
% %         1: detection year (e.g., 2002)
% %         2: detection day (1-365/366)
% %         3: detection month (1-12)
% %         4: detection week (1-52)
% %         5: detection timestamp (MATLAB serial date number)
% %         6: SNR estimate of the call
% %         7: SINR estimate (Signal to Interference plus Noise Ratio)
% %         8: SNR class (floor(SNR/5))
% % 
% % Output ground truth .mat file structure:
% %     'groundTruth' table with variables:
% %         1: 'Selection' - detection number (e.g., 1)
% %         2: 'Fs' - sample rate of file in Hz (e.g., 250)
% %         3: 'BeginFile' - name of file containing start of the detection (e.g., '200_2014-01-10_12-00-00.wav')
% %         4: 'EndFile' - name of file containing end of the detection (e.g., '200_2014-01-10_13-00-00.wav')
% %         5: 'beginTimeFileSec' - time index of detection start re. start of file, in seconds (e.g., 451.73)
% %         6: 'endTimeFileSec' - time index of the detection end re. start of file, in seconds (e.g., 457.82)
% %         7: 'detStartDateTimeDT' - 1x1 datetime array containing time and date of detection start (e.g., '10-Jan-2014 12:59:09')
% %         8: 'Dur90__s_' - duration from 'beginTimeFileSec' to 'endTimeFileSec' in seconds (e.g., 6.273)
% %         9: 'audioAvailable' - true or false, indicating whether the corresponding wav(s) file exist.
% %          
% %%
% clear 
% close all
% clc
% 
% %% Inputs
% 
% % File paths
% rawDetectionsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\Manue_Chagos_RawData\DGS\detections_H08S1_DiegoGarciaS_2015.mat";
% rawAudioPath = "D:\Diego Garcia South\DiegoGarcia2015\wav";
% outputPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\Manue_Chagos_RawData\DGS";
% 
% % Target call duration
% meanCallDuration = 34.38; % (seconds)
% callDurationVariance = 0.4; % (+/- seconds)
% 
% % save name is set automatically from original raw detections file name.
% 
% %% Begin:
% 
% % Load detections
% rawDetections = load(rawDetectionsPath);
% 
% % Set max call duration
% estimatedMaxDuration = meanCallDuration + callDurationVariance;
% 
% % Extract the detections matrix
% if isfield(rawDetections, 'detections')
%     detectionsMatrix = rawDetections.detections;
% else
%     error('The loaded .mat file does not contain a "detections" variable.');
% end
% 
% % Create empty table with required columns for validateDetectionsFromGroundtruth
% groundTruth = table();
% 
% % Process detections
% numDetections = size(detectionsMatrix, 1);
% fprintf('Processing %d detections from ground truth data...\n', numDetections);
% 
% % Create arrays to store data for table
% Selection = zeros(numDetections, 1);
% Fs = zeros(numDetections, 1);
% BeginFile = cell(numDetections, 1);
% EndFile = cell(numDetections, 1);
% beginTimeFileSec = zeros(numDetections, 1);
% endTimeFileSec = zeros(numDetections, 1);
% detStartDateTimeDT = NaT(numDetections, 1);
% Dur90__s_ = zeros(numDetections, 1);
% audioAvailable = true(numDetections, 1);
% 
% % Get list of all wav files
% wavFiles = dir(fullfile(rawAudioPath, '*.wav'));
% 
% % Sort filenames alphanumerically
% sortedFilenames = sort({wavFiles.name});
% 
% % If we can't find one, clean the raw detections data.
% % Go get a cup of tea. This is slow...
% for i = 1:numDetections
%     fprintf('retrieving groundtruth info for detection# %d\n', i)
% 
%     % Set detection index
%     Selection(i) = i;
% 
%     % Extract date information
%     year = detectionsMatrix(i, 1);
%     day = detectionsMatrix(i, 2);
%     month = detectionsMatrix(i, 3);
% 
%     % Convert MATLAB serial date number to datetime
%     detTime = detectionsMatrix(i, 5);
%     detStartDateTimeDT(i) = datetime(detTime, 'ConvertFrom', 'datenum');
% 
%     % Find matching wav file
%     timeString = datestr(detTime, 'yymmdd-HHMMSS');
% 
%     % Try to find the matching wav file - 'yymmdd-HHMMSS'
%     matchingFile = find_closest_wav(wavFiles, timeString);
% 
%     if isempty(matchingFile)
%         warning('No matching wav file found for detection at %s', char(detTime));
%         % Use null values
%         BeginFile{i} = '';
%         EndFile{i} = ''; 
%         beginTimeFileSec(i) = 0;
%         endTimeFileSec(i) = 0;
%         Fs(i) = 0;
%         audioAvailable(i) = false;
%         Dur90__s_(i) = 0;
%     else
%         % Found a matching file
%         BeginFile{i} = matchingFile;
% 
%         % Calculate time from start of file
%         fileStartTimeDT = extractDatetimeFromFilename(matchingFile);
%         if ~isnat(fileStartTimeDT)
%             timeDiff = seconds(detStartDateTimeDT(i) - fileStartTimeDT);
%             beginTimeFileSec(i) = max(0, timeDiff);
% 
%             % Estimate end time based on max possible target duration
%             endTimeFileSec(i) = beginTimeFileSec(i) + estimatedMaxDuration;
% 
%             % Set duration
%             Dur90__s_(i) = endTimeFileSec(i) - beginTimeFileSec(i);
% 
%             try
%                 audioInfo = audioinfo(fullfile(rawAudioPath, matchingFile));
%                 Fs(i) = audioInfo.SampleRate;
%                 fileDuration = audioInfo.Duration;
%             catch ME
%                 warning('Could not read audio file %s: %s', matchingFile, ME.message);
%                 fileDuration = 0;
%                 retry = 1;
%                 maxTries = 3;
%                 success = 0;
%                 while success == 0 && retry <= maxTries
%                     fprintf('\tRetry # %d\n', retry)
%                     try
%                         audioInfo = audioinfo(fullfile(rawAudioPath, matchingFile));
%                         Fs(i) = audioInfo.SampleRate;
%                         success = 1;
%                     catch
%                         success = 0;
%                         pause(3)
%                         retry = retry+1;
%                     end
%                 end
% 
%                 if success == 0
%                     warning('Could not read audio file after %d attempts %s: %s', ...
%                         maxTries, matchingFile, ME.message);
%                     Fs(i) = 0;
%                     audioAvailable(i) = false;
%                     EndFile{i} = [];
%                 end
%             end
% 
%             % Check whether the detection spans two files
%             if endTimeFileSec(i) <= fileDuration
%                 % Spans a single file
%                 EndFile{i} = BeginFile{i};
%             else
% 
%                 % Find the index of the beginFile
%                 beginIdx = find(strcmp(sortedFilenames, BeginFile{i}));
% 
%                 % Get the next file if it exists
%                 if ~isempty(beginIdx) && beginIdx < length(sortedFilenames)
%                     nextFileName = sortedFilenames{beginIdx + 1};
%                 else
%                     nextFileName = '';  % Return empty string if no next file exists
%                     EndFile{i} = [];
%                     audioAvailable(i) = false;
%                 end
% 
%                 % Get the start datetime for that file:
%                 nextFileStartDT = extractDatetimeFromFilename(nextFileName);
% 
%                 % Make sure that the next file's start time is not after
%                 % the detection end time:
%                 endTimeDT = detStartDateTimeDT(i) + seconds(estimatedMaxDuration);
%                 validMultiFileDetection = nextFileStartDT < endTimeDT;
% 
%                 % If we have valid time gaps, use this filename
%                 if validMultiFileDetection == true
%                     EndFile{i} = nextFileName;
%                 else
%                     EndFile{i} = [];
%                     audioAvailable(i) = false;
%                 end
% 
%             end
% 
%         else
%             warning('Could not extract datetime from filename: %s', matchingFile);
%             beginTimeFileSec(i) = 0;
%             endTimeFileSec(i) = 0;
%             EndFile{i} = [];
%             Fs(i) = NaN;
%             audioAvailable(i) = false;
%         end
%     end
% end
% 
% % Create the ground truth table
% groundTruth.Selection = Selection;
% groundTruth.Fs = Fs;
% groundTruth.BeginFile = BeginFile;
% groundTruth.EndFile = EndFile;
% groundTruth.beginTimeFileSec = beginTimeFileSec;
% groundTruth.endTimeFileSec = endTimeFileSec;
% groundTruth.detStartDateTimeDT = detStartDateTimeDT;
% groundTruth.Dur90__s_ = Dur90__s_;
% groundTruth.audioAvailable = audioAvailable;
% 
% % Remove entries with empty filenames
% groundTruth = groundTruth(~cellfun(@isempty, groundTruth.BeginFile), :);
% 
% fprintf('Ground truth data processed. %d valid detections found.\n', ...
%     height(groundTruth));
% 
% % Set filename
% [~, Fn, ~] = fileparts(rawDetectionsPath);
% saveNameGT = strcat(Fn, '_cleaned.mat');
% 
% % Save the cleaned groundtruth table
% save(fullfile(outputPath, saveNameGT), 'groundTruth');
