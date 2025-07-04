% GAVDNet Inference System
%
% Run audio files throught the trained model to detect the target animal
% call.
%
% Model inference runs on audio files in a loop, then saves raw results to
% disk (raw results = probabilities for target call presence per STFT time 
% bin). The post processing procedure that converts these raw probabilities
% into discrete detection boundaries is run in a second loop after
% reloading the raw data. The script is built this way so that post
% processing parameters can be iteratively fine tuned without having to run
% all the audio through the model again, which is computationally
% expensive, time consuming and energy inefficient.
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

plotting = false;

% Path to the config file:
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_SORP_BmAntZ.m";
% configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";

% Different output paths for each version of the Z-Call detector:
many_gavdNetDataPaths{1} = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-10 to 10 Single Exemplar";
many_gavdNetDataPaths{2} = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-10 to 10";
many_gavdNetDataPaths{3} = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-6 to 10";
many_gavdNetDataPaths{4} = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-3 to 10";

% many_gavdNetDataPaths{1} = "D:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar";
% many_gavdNetDataPaths{2} = "D:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10";
% many_gavdNetDataPaths{3} = "D:\GAVDNet\Chagos_DGS\Training & Models\-6 to 10";
% many_gavdNetDataPaths{4} = "D:\GAVDNet\Chagos_DGS\Training & Models\-3 to 10";

% Results path for inference
many_inferenceOutputPaths{1} = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-10 to 10 Single Exemplar";
many_inferenceOutputPaths{2} = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-10 to 10";
many_inferenceOutputPaths{3} = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-6 to 10";
many_inferenceOutputPaths{4} = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-3 to 10";

% many_inferenceOutputPaths{1} = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-10 to 10 Single Exemplar";
% many_inferenceOutputPaths{2} = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-10 to 10";
% many_inferenceOutputPaths{3} = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-6 to 10";
% many_inferenceOutputPaths{4} = "D:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-3 to 10";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.

nModels = length(many_gavdNetDataPaths);
for modelNumber = 1:nModels
    %% Set Paths and Load Input Variables
    
    % Add dependencies to path
    run(configPath) % Load config file
    projectRoot = pwd;
    [gitRoot, ~, ~] = fileparts(projectRoot);
    addpath(fullfile(projectRoot, "Functions"))   

    % Get paths for current loop iteration & overwrite config file ones:
    inferenceOutputPath = many_inferenceOutputPaths{modelNumber};
    gavdNetDataPath = many_gavdNetDataPaths{modelNumber};

    fprintf('\tCurrent GAVDNet data path: %s\n', gavdNetDataPath)
    fprintf('\tCurrent inference output path: %s\n', inferenceOutputPath)

    %% Start logging
    % Begin logging
    ts = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
    logname = ['detector_inference_log_', ts, '.txt'];
    diary(fullfile(inferenceOutputPath, logname));

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
    
    % Set the length threshold parameter for the post-processing.
    % Use the mean training call length, multiplied by the scaling
    % factor set in config:
    postProcOptions.LT = model.dataSynthesisParams.meanTargetCallDuration .* ...
        postProcOptions.LT_scaler;
    
    % Set maximum expected call duration from the longest signal in the
    % training dataset, with a +20% tolerance.
    postProcOptions.maxTargetCallDuration = model.dataSynthesisParams.maxTargetCallDuration * 1.2;
    
    %% Set up for GPU or CPU processing
    
    [useGPU, gpuDeviceID, ~, bytesAvailable] = gpuConfig();
    
    %% Set up test audio datastore
    
    % Build audioDataStore containing audio to test
    ads_test = audioDatastore(inferenceAudioPath, "IncludeSubfolders", false,...
        "FileExtensions",".wav","LabelSource","foldernames", ...
        OutputDataType="single");
    
    if useGPU == true
        % If we are using the GPU, set audioDatastore to output there:
        ads_test.OutputEnvironment = 'gpu';
    else
        ads_test.OutputEnvironment = 'cpu';
    end
    
    % Set output file name for raw results (probabilities)
    saveNamePathRaw = fullfile(inferenceOutputPath, 'detector_raw_results.mat');
    
    %% Run Model
    
    % If raw results have been saved, load raw results, if not, run inference now:
    if ~exist(saveNamePathRaw, 'file')
    
        % Run inference...
        fileIdx = 1;
        reset(ads_test)
        while hasdata(ads_test)
            % Announce start
            fprintf('Running inference on file %d of %d...\n', fileIdx, length(ads_test.Files))
            
            % Clear GPU memory from previous iteration
            if useGPU
                wait(gpuDevice(gpuDeviceID));
                reset(gpuDevice(gpuDeviceID));
            end
        
            % Read audio file
            fprintf('\tReading audio...\n')
            try
                [audioIn, fileInfo] = read(ads_test);
            catch ME
                warning('\tCould not read file: %d\nError: %s. Skipping...\n', fileIdx, ME.message)
                results(fileIdx).failComment = 'Could not read valid audio from file';
                fileIdx = fileIdx + 1;
                continue
            end
        
            % Write filename, Fs and file info to detections struct
            [~, fileName, fileExt] = fileparts(fileInfo.FileName);
            results(fileIdx).fileName = [fileName, fileExt];
            results(fileIdx).fileFs = fileInfo.SampleRate;
            results(fileIdx).fileSamps = length(audioIn);
            results(fileIdx).fileDuration = results(fileIdx).fileSamps / results(fileIdx).fileFs;
            results(fileIdx).probabilities = [];
    
            % Extract datetime from filename
            fprintf('\tExtracting datetime stamp from audio filename...\n')
            results(fileIdx).fileStartDateTime = extractDatetimeFromFilename(fileInfo.FileName, 'datetime');
        
            % Skip this file if it's name doesn't contain a valid start date
            if isempty(results(fileIdx).fileStartDateTime) ||...
                    isnat(results(fileIdx).fileStartDateTime)
                warning('\tCould not extract datetime from filename: %s. Skipping...\n',...
                    results(fileIdx).fileName)
                results(fileIdx).failComment = 'Could not read valid recording start data-time from filename';
                fileIdx = fileIdx + 1;
                continue
            end
        
            % Skip this file if it doesn't contain valid audio
            if isValidAudio(audioIn) == false
                warning('\tFile %s did not contain valid audio. Skipping...\n', results(fileIdx).fileName)
                results(fileIdx).failComment = 'Could not read valid audio from file';
                fileIdx = fileIdx + 1;
                continue
            end
        
            % Construct Datetime vector for the audio (sample-domain)
            results(fileIdx).sampleDomainTimeVector = createDateTimeVector(...
                results(fileIdx).fileStartDateTime,...
                results(fileIdx).fileSamps, ...
                results(fileIdx).fileFs);
        
            % Run preprocessing and inference 
            [results(fileIdx).probabilities, ~, execTime, ...
                results(fileIdx).silenceMask, numAudioSegments] = gavdNetInference(...
                audioIn, fileInfo.SampleRate, model, bytesAvailable, ...
                featureFraming, frameStandardization, minSilenceDuration, plotting);
    
            % Report execution time and seconds of audio with high probability
            numTimeBinsProbHigh = sum(results(fileIdx).probabilities > postProcOptions.AT);
            secondsBinsProbHigh = numTimeBinsProbHigh * windowDur;
            fprintf('\tInference Completed in %.2f seconds\n', execTime)
            fprintf('\tTotal audio duration: %.2f seconds\n', results(fileIdx).fileDuration)
            fprintf('\tDuration with raw detection probability > Activation Threshold%%: %.2f seconds.\n\n', secondsBinsProbHigh)
            
            % Write diagnostic information to the results struct
            results(fileIdx).probsAllNan = all(isnan(results(fileIdx).probabilities));
            results(fileIdx).probsAnyNan = any(isnan(results(fileIdx).probabilities));
            results(fileIdx).audioAllSilence = all(results(fileIdx).silenceMask == true);
            results(fileIdx).audioAnySilence = any(results(fileIdx).silenceMask == true);
            results(fileIdx).numSplitAudioSegments = numAudioSegments;
    
            % Increment the file index for the next iteration
            fileIdx = fileIdx + 1;
        end
    
        % Save the output
        save(saveNamePathRaw, 'results', '-v7.3')
        fprintf('Saved %g detections to %s\n', length(results), inferenceOutputPath)
    
    else
        fprintf('Found file containing raw model outputs. Loading... \n')
    
        % Load the pre-saved raw results
        load(saveNamePathRaw)
    end
    
    %% Reload and post-process the raw predictions to get detections and confidence scores.
    
    fprintf('Postprocesing model outputs...\n')
    for i = 1:length(results)
    
        % Get audio for this file:
        [audioIn, fileFs] = audioread(fullfile(inferenceAudioPath, results(i).fileName));
    
        % Run postprocessing to determine decision boundaries. 
        [results(i).eventSampleBoundaries, ~, ...
            results(i).confidence] = gavdNetPostprocess(...
            audioIn, fileFs, results(i).probabilities, model.preprocParams, ...
            postProcOptions);
    
        % Get number of detections
        results(i).nDetections = size(results(i).eventSampleBoundaries, 1);
    
        % Get the datetime start and end times for each detected event using 
        if ~isempty(results(i).eventSampleBoundaries)
            for detIdx = 1:results(i).nDetections
    
                % Get event boundaries (as sample indices)
                eventStart = results(i).eventSampleBoundaries(detIdx, 1);
                eventEnd = results(i).eventSampleBoundaries(detIdx, 2);
    
                % Convert samples indices to datetime relative to file start.
                results(i).eventTimesDT(detIdx, 1) = results(i).sampleDomainTimeVector(eventStart);
                results(i).eventTimesDT(detIdx, 2) = results(i).sampleDomainTimeVector(eventEnd);
            end
        end
         fprintf('File %g: %g events detected\n', i, results(i).nDetections)
    end
    
    % Detections are one row per audio file, potentially multiple detections per row.
    % Flatten detections to one row per detection.
    results = flattenDetections(results, model.preprocParams);
    
    %% Save the output
    saveNamePath = fullfile(inferenceOutputPath, 'detector_results_postprocessed.mat');
    save(saveNamePath, 'results', 'featureFraming', 'frameStandardization', 'postProcOptions', '-v7.3')
    
    fprintf('Saved %g post processed detections to %s\n', length(results), inferenceOutputPath)
    diary off
end