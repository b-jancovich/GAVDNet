% This script is a quick test of the trained detector using a single short
% recording that has 5 high SNR calls in it. Use this as a basis for your
% inference script.

clear all
close all
clc

%% Settings

featureFraming = 'simple'; % 'simple' or 'smart' or 'none'; 
frameDuration = 600; % seconds
frameOverlapPercent = 50; % percent

%% File paths

% Model File Path
modelPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\chagos_DGS_2025\GAVDNet_trained_12-Jun-2025_15-39.mat";

% Config file Path
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";
% configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_SORP_BmAntZ.m";

% Test Audio File Path - File 1: target call
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-000000_EarthquakeDynamicRangeTest.wav"; 
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-160000_HighDynamicRangeCalls.wav";
audioPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_150605-120000_calls+extremelyHighPowerNoise.wav";
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-200000_TypicalLotsOfCalls_TrimmedTo2.08.39.wav";
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\Chagos_whale_song_DGS_071102.wav";
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\BmAnt_ZCall_Casey_2014-03-30_04-00-00.wav";

% Audio Path - File 2: non-target call
% audioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Animal Recordings\Whale Calls\NewZealand_BW_L7910NZ01_002K_Site04_multi_20160225_051500Z_15min.wav";

%% Load Config, Model & audio
addpath('C:\Users\z5439673\Git\GAVDNet\Functions');
load(modelPath)
run(configPath)
[audio, fs] = audioread(audioPath);
audio = gpuArray(audio);

% Use the "minimum call duration" parameter from the training data synthesis 
% information, stored in the model metadata as the length threshold for 
% post-processing. You can try using smaller values for this if parts of 
% your target call are frequently missing due to propagation effects.
postProcOptions.LT = model.dataSynthesisParams.minTargetCallDuration * postProcOptions.LT_scaler;

%% Run the model, either with or without feature framing

switch featureFraming
    case 'simple'
        % Normalize first
        audio = audio ./ max(abs(audio));

        % Run Preprocessing & Feature Extraction on audio
        fprintf('Preprocesing audio & extracting features...\n')
        features = gavdNetPreprocess(...
            audio, ...
            fs, ...
            model.preprocParams.fsTarget, ...
            model.preprocParams.bandwidth, ...
            model.preprocParams.windowLen,...
            model.preprocParams.hopLen,...
            model.preprocParams.saturationRange);

        % Framing params
        frameLength = round(frameDuration / model.preprocParams.hopDur);
        frameHopLength = round(frameLength * (1 - frameOverlapPercent/100));
        
        % Break featrures into frames
        featureFrames = featureBuffer(features, frameLength, frameOverlapPercent);
        
        % Run each frame through the model
        probabilitiesFrames = cell(size(featureFrames));
        fprintf('\tRunning frames through model.')
        for i = 1:length(featureFrames)
            % Run model & move output back to CPU from GPU
            probabilitiesFrames{i} = gather(predict(model.net, featureFrames{i}));
        
            % Do some dots so the user knows we haven't hung 
            if mod(i, 10) == 0
                fprintf('.')
            end
        end
        fprintf('\n')
        
        % Stitch together probability vectors for each frame and take the 
        % average of overlapping elements
        numSpectrogramTimeBins = size(features, 2);
        probabilities = concatenateOverlappingProbs(probabilitiesFrames, ...
            numSpectrogramTimeBins, frameHopLength);

    case 'smart'

        % Split audio into segments based on signal statistics
        smoothingWindowDuration = model.dataSynthesisParams.maxTargetCallDuration * 4;
        eventOverlapDuration = model.dataSynthesisParams.maxTargetCallDuration;
        [audioSegments, splitIndices, changePtsIndices] = eventSplitter(...
            audio, fs, smoothingWindowDuration, eventOverlapDuration);

        % Process single or multiple segments
        if isscalar(audioSegments)

            fprintf('\tPreprocesing audio & extracting features...\n')
            % Run Preprocessing & Feature Extraction on audio
            features = gavdNetPreprocess(...
                audio, ...
                fs, ...
                model.preprocParams.fsTarget, ...
                model.preprocParams.bandwidth, ... 
                model.preprocParams.windowLen,...
                model.preprocParams.hopLen, ...
                model.preprocParams.saturationRange);
    
            fprintf('\tRunning features through model...')
            tic
            if useGPU == true
                % % Run model & move output back to CPU from GPU
                 results(fileIdx).probabilities = gather(minibatchpredict(model.net, ...
                     features, 'SequenceLength', 'shortest'));
            else
                % Run model on CPU
                results(fileIdx).probabilities = minibatchpredict(model.net, ...
                     features, 'SequenceLength', 'shortest');
            end
            execTime = toc;
            fprintf('\n')

        else
            featuresSegments = cell(size(audioSegments));
            for i = 1:length(audioSegments)
                fprintf('\tPreprocesing audio & extracting features for segment %d of %d...\n',...
                    i, length(audioSegments))
                
                % Normalize segments
                audioSegments{i} = audioSegments{i} ./ max(abs(audioSegments{i}));

                % Run Preprocessing & Feature Extraction on audio
                featuresSegments{i} = gavdNetPreprocess(...
                    audioSegments{i}, ...
                    fs, ...
                    model.preprocParams.fsTarget, ...
                    model.preprocParams.bandwidth, ... 
                    model.preprocParams.windowLen,...
                    model.preprocParams.hopLen, ...
                    model.preprocParams.saturationRange);
            end

            % Run each frame through the model
            probs = cell(size(featuresSegments));
            fprintf('\tRunning frames through model.')
            tic
            for i = 1:length(featuresSegments)
                % % Run model & move output back to CPU from GPU
                probs{i} = gather(minibatchpredict(model.net, ...
                featuresSegments{i}, 'SequenceLength', 'shortest'));
            end
            fprintf('\n')
            execTime = toc;
            
            % Stitch together probability vectors for each segment
            [probabilities, features] = segmentStitcher(probs, splitIndices, ...
                model.preprocParams.hopLen, featuresSegments);
        end


    case 'none'

        % normalize first
        audio = audio ./ max(abs(audio));

        % Run Preprocessing & Feature Extraction on audio
        fprintf('Preprocesing audio & extracting features...\n')
        features = gavdNetPreprocess(...
            audio, ...
            fs, ...
            model.preprocParams.fsTarget, ...
            model.preprocParams.bandwidth, ...
            model.preprocParams.windowLen,...
            model.preprocParams.hopLen,...
            model.preprocParams.saturationRange);

        % Run Model in minibatch mode to save memory
        fprintf('Running features through model...\n')
        probabilities = minibatchpredict(model.net, gpuArray(features));
end

%% Run postprocessing to determine decision boundaries. 

fprintf('Postprocesing model outputs...\n')
gavdNetPostprocess(audio, fs, probabilities, model.preprocParams, postProcOptions, features);
 
% Get regions of interest
roi = gavdNetPostprocess(audio, fs, probabilities, model.preprocParams, postProcOptions);


