% Test VADNet

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

%% Set up for GPU or CPU processing

numGPUs = gpuDeviceCount("available");
% Check if GPU is available
if numGPUs > 0
    % GPU is available
    fprintf('Found %d GPU device(s)\n', numGPUs);

    % Initialize variables to store max memory and corresponding device ID
    maxMemory = 0;
    maxMemoryDeviceID = 1;

    % Loop through each GPU to find the one with maximum available memory
    for i = 1:numGPUs
        % Get current GPU info
        gpuInfo = gpuDevice(i);

        % Get available memory in bytes and convert to GB for display
        availableMemory = gpuInfo.AvailableMemory;
        availableMemoryGB = availableMemory / (1024^3);

        fprintf('GPU %d: %s - Available Memory: %.2f GB\n', ...
            i, gpuInfo.Name, availableMemoryGB);

        % Check if this GPU has more available memory
        if availableMemory > maxMemory
            maxMemory = availableMemory;
            maxMemoryDeviceID = i;
        end
    end

    % Select the GPU with the most available memory
    fprintf('Selecting GPU %d with %.2f GB available memory\n', ...
        maxMemoryDeviceID, maxMemory / (1024^3));
    gpu = gpuDevice(maxMemoryDeviceID);
    reset(gpu);

    useGPU = true;
    disp('Audio datastore will output to GPU.')
else
    % No GPU available
    disp('AudioDatastore will output to CPU.');
    useGPU = false;
end
clear gpuInfo


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
while hasdata(ads_test)
    % Announce start
    fprintf('\nRunning pre-processing file %d of %d...\n', fileIdx, length(ads_test.Files))

    % Clear GPU memory from previous iteration
    if useGPU
        wait(gpuDevice(maxMemoryDeviceID)); % Wait for operations on the selected GPU
        reset(gpuDevice(maxMemoryDeviceID)); % Reset the selected GPU
    end

    % Read audio file
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
    detections(fileIdx).fileStartDateTime = extractDatetimeFromFilename(fileInfo.FileName);

    % Skip if file doesn't contain valid audio
    if isValidAudio(audioIn) == false
        warning('File %s did not contain valid audio. Skipping...\n', detections(fileIdx).fileName)
        fileIdx = fileIdx + 1;
        continue
    end

    % Skip if file name doesn't contain valid date
    if isempty(detections(fileIdx).fileStartDateTime) ||...
            isnat(detections(fileIdx).fileStartDateTime)
        warning('Could not extract datetime from filename: %s. Skipping...\n',...
            detections(fileIdx).fileName)
        fileIdx = fileIdx + 1;
        continue
    end

    % Run Model 
    y = predict(model.net, gpuArray(XValidation));

    % Run VADNet Postprocessing
    boundaries = vadnetPostprocess(audioValidation,fs,y);
    
    YValidationPerSample = double(sigroi2binmask(boundaries,size(audioValidation,1)));
    
    XValidation = vadnetPreprocess(audioValidation,fs);

