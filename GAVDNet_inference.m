% GAVDNet Inference
%
% Run audio files throught the trained model to detect the target animal
% call.
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
configPath = "C:\Users\z5439673\Git\GAVDNet\config_DGS_chagos.m";

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
else
    [file, location] = uigetfile(gavdNetDataPath, 'Select a model to load:');
    load(fullfile(location, file))
end

% Set the length threshold parameter for the post-processing equal to the 
% duration of the shortest call in the training dataset.
postProcOptions.LT = model.dataSynthesisParams.minTargetCallDuration;

%% Set up for GPU or CPU processing

[useGPU, gpuDeviceID, ~] = gpuConfig();

%% Set up test audio datastore

% Build audioDataStore containing audio to test
ads_test = audioDatastore(testAudioPath, "IncludeSubfolders", false,...
    "FileExtensions",".wav","LabelSource","foldernames", ...
    OutputDataType="single");

if useGPU == true
    % If we are using the GPU, set audioDatastore to output there:
    ads_test.OutputEnvironment = 'gpu';
else
    ads_test.OutputEnvironment = 'cpu';
end

%% Load Groundtruth

load(groundtruthPath)

%% Run Model

fileIdx = 1;
reset(ads_test)
while hasdata(ads_test)
    % Announce start
    fprintf('Running inference on file %d of %d...\n', fileIdx, length(ads_test.Files))
    
    % Clear GPU memory from previous iteration
    if useGPU
        wait(gpuDevice(gpuDeviceID)); % Wait for operations on the selected GPU
        reset(gpuDevice(gpuDeviceID)); % Reset the selected GPU
    end

    % Read audio file
    fprintf('Reading audio...\n')
    try
        [audioIn, fileInfo] = read(ads_test);
    catch ME
        warning('Could not read file: %d\nError: %s. Skipping...\n', fileIdx, ME.message)
        fileIdx = fileIdx + 1;
        continue
    end

    % Write filename, Fs and file info to detections struct
    [~, fileName, fileExt] = fileparts(fileInfo.FileName);
    detections(fileIdx).fileName = [fileName, fileExt];
    detections(fileIdx).fileFs = fileInfo.SampleRate;
    detections(fileIdx).fileSamps = length(audioIn);
    detections(fileIdx).fileDuration = detections(fileIdx).fileSamps / detections(fileIdx).fileFs;

    % Extract datetime from filename
    fprintf('Extracting datetime stamp from audio filename...\n')
    detections(fileIdx).fileStartDateTime = extractDatetimeFromFilename(fileInfo.FileName, 'datetime');

    % Skip this file if it's name doesn't contain a valid start date
    if isempty(detections(fileIdx).fileStartDateTime) ||...
            isnat(detections(fileIdx).fileStartDateTime)
        warning('Could not extract datetime from filename: %s. Skipping...\n',...
            detections(fileIdx).fileName)
        fileIdx = fileIdx + 1;
        continue
    end

    % Skip this file if it doesn't contain valid audio
    if isValidAudio(audioIn) == false
        warning('File %s did not contain valid audio. Skipping...\n', detections(fileIdx).fileName)
        fileIdx = fileIdx + 1;
        continue
    end

    % Construct Datetime vector for the audio (sample-domain)
    detections(fileIdx).sampleDomainTimeVector = createDateTimeVector(...
        detections(fileIdx).fileStartDateTime,...
        detections(fileIdx).fileSamps, ...
        detections(fileIdx).fileFs);

    % Run Preprocessing & Feature Extraction on audio
    fprintf('Preprocesing audio & extracting features...\n')
    [features, ~] = gavdNetPreprocess(...
        audioIn, ...
        detections(fileIdx).fileFs, ...
        model.preprocParams.fsTarget, ...
        model.preprocParams.bandwidth, ...
        model.preprocParams.windowLen,...
        model.preprocParams.hopLen);

    % Run Model in minibatch mode to save memory
    fprintf('Running model...\n')
    probabilities = minibatchpredict(model.net, gpuArray(features));

    % Run postprocessing to determine decision boundaries. 
    fprintf('Postprocesing model outputs...\n')
    [detections(fileIdx).eventSampleBoundaries, probabilities, ...
        detections(fileIdx).confidence, fig] = gavdNetPostprocess(...
        audioIn, detections(fileIdx).fileFs, probabilities, model.preprocParams, ...
        postProcOptions);

    % Get number of detections
    detections(fileIdx).nDetections = length(detections(fileIdx).eventSampleBoundaries);

    % Get the datetime start and end times for each detected event using 
    if ~isempty(detections(fileIdx).eventSampleBoundaries)
        for detIdx = 1:detections(fileIdx).nDetections

            % Get event boundaries (as sample indices)
            eventStart = detections(fileIdx).eventSampleBoundaries(detIdx, 1);
            eventEnd = detections(fileIdx).eventSampleBoundaries(detIdx, 2);

            % Convert samples indices to datetime relative to file start.
            detections(fileIdx).eventTimesDT(detIdx, 1) = detections(fileIdx).sampleDomainTimeVector(eventStart);
            detections(fileIdx).eventTimesDT(detIdx, 2) = detections(fileIdx).sampleDomainTimeVector(eventEnd);
        end
    end

     % Increment the file index for the next iteration
     fileIdx = fileIdx + 1;
end

% Detections are one row per audio file, potentially multiple detections per row.
% Flatten detections to one row per detection.
flatDetections = flattenDetections(detections);