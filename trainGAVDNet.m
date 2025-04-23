%% GAVDNet Training With Synthetic Corpus (Transfer Learning)
%
% This script applies transfer learning to adapt a pretrained voice activity 
% detection network (vadnet) for detecting stereotyped animal vocalizations 
% of in passive acoustic monitoring recordings.
%
% The script constructs a synthetic training corpus by augmenting clean 
% vocalization samples and mixing them with ambient noise at various 
% signal-to-noise ratios. These synthetic training sequences are then used 
% to fine-tune the pretrained vadnet model.
%
% INPUT:
%   The script requires:
%     1. Noiseless samples of the target call (wav files) in 
%        'noiseless_sample_path', manually denoised using Izotope RX 
%        Spectral Editor.
%     2. Background noise recordings (wav files) in 'noise_library_path',
%        ideally containing extraneous noise sources similar to the signal of
%        interest, e.g, the calls of other 'non-target' animals.
%     3. A configuration file with detection parameters
%
% OUTPUT:
%   The script produces:
%     1. Augmented clean vocalization samples
%     2. Synthetic training and validation sequences + masks
%     3. A trained GAVDNet neural network model with embedded metadata
%
% WORKFLOW:
%   1. Load and preprocess clean vocalization samples
%   2. Create augmented versions with various distortions
%   3. Construct synthetic training sequences by mixing clean samples with noise
%   4. Extract spectrogram features using avdnetPreprocess
%   5. Apply transfer learning to the pretrained VAD network
%   6. Save the trained model with metadata
%
% REQUIREMENTS:
%   - MATLAB R2023a or later
%   - Audio Toolbox
%   - Deep Learning Toolbox
%   - Parallel Computing Toolbox (recommended for speed)
%   - customAudioAugmenter
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
[git_root_path, ~, ~] = fileparts(projectRoot);
addpath(fullfile(git_root_path, 'customAudioAugmenter'));

%% Start logging
% 
% % Begin logging
% ts = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
% logname = ['train_detector_network_log_', ts, '.txt'];
% diary(fullfile(outputPath, logname));

%% Look for pre-generated audio, sequences, inputs and targets to reload

fprintf('\nStarting dataset construction/loading...\n')
% If the clean signals folder doesn't exist, or contains no wav files, 
% make the folder, and set build clean signals to true: 
cleanSignalsPath = fullfile(gavdNetDataPath, 'clean_signals');
if ~exist(cleanSignalsPath, 'dir')
    mkdir(cleanSignalsPath);
end
if size(dir(fullfile(cleanSignalsPath, '*.wav')), 1) == 0
    buildCleanSignals = true;
    fprintf('No pre-saved clean signals found. Building new set.\n')
else
    buildCleanSignals = false;
    fprintf('Found %g pre-saved clean signals. Using existing signals.\n',  ...
        size(dir(fullfile(cleanSignalsPath, '*.wav')), 1))
end

% If the sequences folder doesn't exist, or contains no mat files, 
% make the folder, and set build sequences to true: 
sequencesPath = fullfile(gavdNetDataPath, 'sequences');
if ~exist(sequencesPath, 'dir')
    mkdir(sequencesPath);
end
if size(dir(fullfile(sequencesPath, '*.mat')), 1) == 0
    buildSequences = true;
    fprintf('No pre-saved sequences found. Building new set.\n')
else
    buildSequences = false;
    fprintf('Found pre-saved sequences. Using existing sequences.\n')
end

% If the 'X and T' folders don't exist, or contain no mat files, 
% make the folders, and set build data to true:
% X and T refer to the inputs and targets for training and validaiton
trainXandTpath = fullfile(gavdNetDataPath, 'trainXandT');
valXandTpath   = fullfile(gavdNetDataPath, 'valXandT');
if ~exist(trainXandTpath, 'dir')
    mkdir(trainXandTpath);
end
if ~exist(valXandTpath, 'dir')
    mkdir(valXandTpath);
end
if size(dir(fullfile(trainXandTpath, '*.mat')), 1) == 0 || ...
        size(dir(fullfile(valXandTpath, '*.mat')), 1) == 0 
    buildXandT = true;
    fprintf('No pre-saved inputs (x) and targets (T) found. Building new set.\n')
else
    buildXandT = false;
    fprintf('Found %g pre-saved training inputs (x) and targets (T)\n', ...
        size(dir(fullfile(trainXandTpath, '*.mat')), 1))
    fprintf('Found %g pre-saved validation inputs (x) and targets (T)\n', ...
        size(dir(fullfile(valXandTpath, '*.mat')), 1))
    fprintf('Using existing inputs and targets.\n')
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


%% Load the noiseless sample(s)

% Get list of noiseless samples
noiseless_samples = dir(fullfile(noiseless_sample_path, '*.wav'));
n_noiseless_samples = length(noiseless_samples);
if buildCleanSignals == true  
    for i = 1:n_noiseless_samples
        % Load the noiseless detection from which to build the test dataset.
        [noiseless_samples(i).audio, noiseless_samples(i).Fs] = audioread(...
            fullfile(noiseless_samples(i).folder, noiseless_samples(i).name));
    end
end

%% Pre-process the noiseless sample(s)

if buildCleanSignals == true
    fprintf('Preprocessing Noiseless Sample(s)...\n')
    
    for i = 1:n_noiseless_samples
        % Run pre-processor
        [noiseless_samples(i).preprocessedAudio, operatingFs] = preprocessSignals(...
            noiseless_samples(i).audio, noiseless_samples(i).Fs, ...
            preAugfadeIn, preAugfadeOut, [windowDur, windowDur]);
    end
    
    % Calculate Dynamic Range Compression Parameters based on signal analysis
    compressorParams = struct();
    for i = 1:n_noiseless_samples
        compressorParams(i) = calculateCompressionParameters(noiseless_samples(i).preprocessedAudio,...
            target_dynamic_range);
    end
    
    % Build compressor system object
    dRC = compressor(MakeUpGainMode='auto', SampleRate=operatingFs);
    
    % Compress all samples 
    for i = 1:n_noiseless_samples
        % Set compressor params for this sample
        dRC.Threshold = compressorParams(i).threshold;
        dRC.Ratio = compressorParams(i).ratio;
        dRC.KneeWidth = compressorParams(i).kneeWidth;
        dRC.AttackTime = compressorParams(i).attackTime;
        dRC.ReleaseTime = compressorParams(i).releaseTime;
        
        % Run sample through compressor
        noiseless_samples(i).compressedAudio = dRC(noiseless_samples(i).preprocessedAudio);
        
        % Normalize to avoid clipping
        noiseless_samples(i).compressedAudio = noiseless_samples(i).compressedAudio ./ ...
            max(abs(noiseless_samples(i).compressedAudio));
    end
end

%% Build augmentation object(s)

% Divide the total number of desired training samples by the number of
% clean samples we have:
n_augmented_copies = ceil(n_training_samples / n_noiseless_samples);

% Number of years of data in noise library
year_list = detect_year_range(1):1:detect_year_range(2);

for i = 1:n_noiseless_samples
    % Extract the year from a filename using regular expression
    noiselessSampleFileName = noiseless_samples(i).name;
    pattern = '.*-(\d{4})_.*';
    matches = regexp(noiselessSampleFileName, pattern, 'tokens');
    noiseless_samples(i).sample_year = str2double(matches{1}{1});
    noiseless_samples(i).CFreq = initial_freq + ((noiseless_samples(i).sample_year...
        - initial_freq_year) * pitch_shift_rate);

    % Calculate Max range of freq shift for each noiseless sample
    n_future_years = max(year_list) - noiseless_samples(i).sample_year;
    n_past_years = noiseless_samples(i).sample_year - min(year_list);
    max_up_shift_hz = ((n_future_years + 1) * pitch_shift_rate) + pitch_shift_tol;
    max_down_shift_hz = ((n_past_years + 1) * pitch_shift_rate) + pitch_shift_tol;

    % Augmenter's pitch shifter works in semitones, so convert from Hz relative
    % to the fundamental frequency of the clean sample:
    max_up_shift_semitones = 12 * log2((noiseless_samples(i).CFreq + max_up_shift_hz) ...
        / noiseless_samples(i).CFreq);
    max_down_shift_semitones = 12 * log2((noiseless_samples(i).CFreq - max_down_shift_hz) ...
        / noiseless_samples(i).CFreq);
    noiseless_samples(i).pitch_shift_range_semitones = [max_down_shift_semitones, max_up_shift_semitones];
        
    if buildCleanSignals == true
        % Build the augmenter, and set it to make "n_augmented_copies" per
        % noiseless sample:
        noiseless_samples(i).augmenter = setupAugmenter(operatingFs, n_augmented_copies, ...
            speedup_factor_range, noiseless_samples(i).pitch_shift_range_semitones,...
            distortionRange, source_velocity_range, lpf_cutoff_range, ...
            decayTimeRange, trans_loss_strength_range, trans_loss_density_range, ...
            c);
    end
end

%% Build augmented clean signals, post process & save to disk

if buildCleanSignals == true
    disp("Generating augmented clean signal dataset...")
    
    % Run Clean Signal Augmentation on each sample
    clean_signals = table();
    parfor i = 1:n_noiseless_samples
        augOut = augment(noiseless_samples(i).augmenter, ...
            noiseless_samples(i).compressedAudio, operatingFs);
    
        clean_signals = [clean_signals; augOut];
        fprintf('Finished Generating %d augmented signals for noiseless sample %d.\n', n_augmented_copies, i)
    end
    for i = 1:height(clean_signals)   
        % Post process the signals (top & tail, DC filt, trim silence from start/end, 
        % fade start/end, highpass filt @ fMin, top & tail, DC Filt again. 
        clean_signals.Audio{i} = postprocessSignals(clean_signals.Audio{i}, operatingFs, outputFs, ...
            trim_threshold_ratio, trim_window_size, postAugfades, bandwidth(1));

        % Set this signal's filename
        cleanSignalsFilePath = fullfile(cleanSignalsPath, ...
            sprintf('augmentedNoiselessSample_%g.wav', i));

        % If we have valid audio, write the audio to disk
        if isValidAudio(clean_signals.Audio{i}) == true
            audiowrite(cleanSignalsFilePath, clean_signals.Audio{i}, outputFs);
        end

    end
    fprintf('Finished building clean signals. %g signals saved to disk.\n', i)
end

clear clean_signals

%% Initialise audioDataStores for clean signals & noise
    
% Init the training/validation split
split = [trainPercentage/100, (100-trainPercentage)/100];

% If we haven't yet built the training & validation sequences and saved 
% them to disk, set up datastores for the raw noise & call samples:
if buildSequences == true

    % Run dataset validation checks on file length & fs
    disp('Checking all clean signals have uniform sample rates...')
    cleanSignals_fs = validateTrainingDataFs(cleanSignalsPath);
    
    disp('Setting up datastores...')
    % Build audioDataStore containing training samples of the target sound
    ads_cleanSignals = audioDatastore(cleanSignalsPath, "IncludeSubfolders", true,...
        "FileExtensions",".wav","LabelSource","foldernames", ...
        OutputDataType="single");
    
    % Build audioDataStore containing samples of background noise
    ads_noise = audioDatastore(noise_library_path, "IncludeSubfolders", false,...
        "FileExtensions",".wav","LabelSource","foldernames", ...
        OutputDataType="single");

    if useGPU == true
        % If we are using the GPU, set audioDatastore to output there:
        ads_noise.OutputEnvironment = 'gpu';
        ads_cleanSignals.OutputEnvironment = 'gpu';
    else
        ads_noise.OutputEnvironment = 'cpu';
        ads_cleanSignals.OutputEnvironment = 'cpu';
    end

    % Split the full datasets into training data and validation data.
    [ads_cleanSignals_train, ads_cleanSignals_validation] = splitEachLabel(...
        ads_cleanSignals, split(1), split(2));
    [ads_noise_train, ads_noise_validation] = splitEachLabel(...
        ads_noise, split(1), split(2));
    
    clear ads_noise ads_cleanSignals

else
    % Just get the sampling frequency:
    % Build audioDataStore containing training samples of the target sound
    ads_cleanSignals = audioDatastore(cleanSignalsPath, "IncludeSubfolders", true,...
        "FileExtensions",".wav","LabelSource","foldernames", ...
        OutputDataType="single");
    [~, ads_info] = read(ads_cleanSignals);
    cleanSignals_fs = ads_info.SampleRate;
    clear ads_cleanSignals
end

%% Build Training & Validation Sequences from Clean Samples & Noise

% If we haven't yet built the training & validation sequences and saved 
% them to disk, build them now:
if buildSequences == true

    % Set number of sequences for training and validation
    numSequencesTrain = round(numSequences * split(1));
    numSequencesVal = round(numSequences * split(2));

    % Build sequences & masks by concatenating clean signals and random 
    % durations of silence. Masks = 1 @ sample indices of clean signals:
    disp('Building sequences of training samples...')
    [audioSequencesTrain, masksTrainPerSample] = constructSequence(...
        ads_cleanSignals_train, cleanSignals_fs, sequenceDuration, ...
        numSequencesTrain, minSilenceSegment);
    disp('Building sequences of validation samples...')
    [audioSequencesVal, masksValPerSample] = constructSequence(...
        ads_cleanSignals_validation, cleanSignals_fs, sequenceDuration,...
        numSequencesVal, minSilenceSegment);
    
    % Add noise to training & validationsequences
    disp('Adding noise to training sequences...')
    [audioSeqsNoisyTrain, audioTrainSNRs] = addNoiseToSequences(audioSequencesTrain,...
        ads_noise_train, snrRange, sequenceDuration, cleanSignals_fs);
    disp('Adding noise to validation sequences...')
    [audioSeqsNoisyVal, audioValSNRs] = addNoiseToSequences(audioSequencesVal, ...
        ads_noise_validation, snrRange, sequenceDuration, cleanSignals_fs);
    
    % Display some sequences, masks, & noise-corrupted sequences 
    if plotting == true
        for i = 1:10
            maskTrain = signalMask(gather(masksTrainPerSample{i}), SampleRate=cleanSignals_fs);
            maskVal = signalMask(gather(masksValPerSample{i}), SampleRate=cleanSignals_fs);
        
            figure(1)
            tiledlayout(2,2)
            nexttile
            plotsigroi(maskTrain, gather(audioSequencesTrain{i}), true)
            title(["Train # ", num2str(i), " - Clean Sequence & Mask"])
        
            nexttile
            plotsigroi(maskVal, gather(audioSequencesVal{i}), true)
            title(["Validation # ", num2str(i), " - Clean Sequence & Mask"])
        
            nexttile
            plotsigroi(maskTrain, gather(audioSeqsNoisyTrain{i}), true)
            title(["Validation # ", num2str(i), " - Noise Corrrupted Sequence & Mask - SNR: ", num2str(audioTrainSNRs(i))])
        
            nexttile
            plotsigroi(maskVal, gather(audioSeqsNoisyVal{i}), true)
            title(["Validation # ", num2str(i), " - Noise Corrrupted Sequence & Mask - SNR: ", num2str(audioValSNRs(i))])
        
            waitforbuttonpress
        end
    end
    
    % Save both the sequences and the masks to mat files
    save(fullfile(sequencesPath, 'trainingSequencesAndMasks.mat'), "audioSeqsNoisyTrain", "masksTrainPerSample", '-v7.3')
    save(fullfile(sequencesPath, 'validationSequencesAndMasks.mat'), "audioSeqsNoisyVal", "masksValPerSample", '-v7.3')
    
else
    % Load the pre-built training and validation sequences and masks
    load(fullfile(sequencesPath, 'trainingSequencesAndMasks.mat'))
    load(fullfile(sequencesPath, 'validationSequencesAndMasks.mat'))
end

%% Extract features from training & validation data, & prep for network

% Convert STFT parameters from seconds to samples
windowLen = windowDur * fsTarget;
hopLen = hopDur * fsTarget;
overlapLen = windowLen - hopLen;
analysisHopLength = windowLen - overlapLen;
analysisTimeStepLength = round(analysisTimeStepDuration * cleanSignals_fs / analysisHopLength);
% If we haven't already saved the final inputs and targets for training and
% validation:
if buildXandT == true

    % Training data
    ntrainSequences = length(audioSeqsNoisyTrain);
    reportInterval = floor(ntrainSequences / 10);
    featuresTrain = cell(ntrainSequences, 1);
    for i = 1:ntrainSequences
        % Run preprocessor:
        featuresTrain{i} = gavdNetPreprocess(audioSeqsNoisyTrain{i}, cleanSignals_fs, fsTarget, bandwidth, windowDur, hopDur);
    
        if useGPU == true
            % Move back to the CPU to save memory
            featuresTrain{i} = gather(featuresTrain{i});
        end
    
        % Report progress
        if mod(i, reportInterval) == 0
            fprintf('Completed preprocessing and feature extraction on training sequence %g of %g.\n', i, ntrainSequences)
        end
    end
    
    % Validation data
    nValSequences = length(audioSeqsNoisyVal);
    reportInterval = floor(nValSequences / 10);
    featuresVal = cell(nValSequences, 1);
    for i = 1:nValSequences
        % Run preprocessor:
        featuresVal{i} = gavdNetPreprocess(audioSeqsNoisyVal{i}, cleanSignals_fs, fsTarget, bandwidth, windowDur, hopDur);
    
        if useGPU == true
            % Move back to the CPU to save memory
            featuresVal{i} = gather(featuresVal{i});
        end
    
        % Report progress
        if mod(i, reportInterval) == 0
            fprintf('Completed preprocessing and feature extraction on validation sequence %g of %g.\n', i, nValSequences)
        end
    end
       
    % The ground truth mask has elements that correspond to the original time 
    % domain audio samples. Buffer the mask vectors so they correspond to the 
    % spectrogram time bins instead:
    maskTrainPerSamplePadded = cell(length(masksTrainPerSample), 1);
    TTrain = cell(length(masksTrainPerSample), 1);
    maskValPerSamplePadded = cell(length(masksValPerSample), 1);
    TValidation = cell(length(masksValPerSample), 1);
    for i = 1:length(masksTrainPerSample)
        % Zero pad to match the features
        maskTrainPerSamplePadded{i} = [zeros(floor(windowLen/2), 1); masksTrainPerSample{i}; zeros(ceil(windowLen/2), 1)];
        
        % Buffer to match the features
        TTrain{i} = mode(buffer(maskTrainPerSamplePadded{i}, windowLen, overlapLen, "nodelay"), 1);
    end
    for i = 1:length(masksValPerSample)
        % Zero pad to match the features
        maskValPerSamplePadded{i} = [zeros(floor(windowLen/2), 1); masksValPerSample{i}; zeros(ceil(windowLen/2), 1)];
    
        % Buffer to match the features
        TValidation{i} = mode(buffer(maskValPerSamplePadded{i}, windowLen, overlapLen, "nodelay"), 1);
    end
    
    % Buffer the long sequences into shorter ones
    XTrainBuffered = cell(length(featuresTrain), 1);
    TTrainBuffered = cell(length(featuresTrain), 1);
    XValBuffered = cell(length(featuresVal), 1);
    TValBuffered = cell(length(TValidation), 1);
    for i = 1:length(featuresTrain)
    
        % Buffer training features
        XTrainBuffered{i} = featureBuffer(featuresTrain{i}, analysisTimeStepLength, ...
            analysisTimeStepOverlapPercent);
        
        % Buffer training masks
        TTrainBuffered{i} = featureBuffer(TTrain{i}, analysisTimeStepLength, ...
            analysisTimeStepOverlapPercent);
    end
    for i = 1:length(featuresVal)
        % Buffer validation features
        XValBuffered{i} = featureBuffer(featuresVal{i}, analysisTimeStepLength, ...
            analysisTimeStepOverlapPercent);
        
        % Buffer validation masks
        TValBuffered{i} = featureBuffer(TValidation{i}, analysisTimeStepLength, ...
            analysisTimeStepOverlapPercent);
    end
    
    % Concatenate/unnest cells of training & validation data
    [XTrainAll, TTrainAll] = concatenateTrainData(XTrainBuffered, TTrainBuffered);
    [XValAll, TValAll] = concatenateTrainData(XValBuffered, TValBuffered);
    
    % Save train chunks
    for i = 1:numel(XTrainAll)
        X = XTrainAll{i}; 
        T = TTrainAll{i}; 
        save(fullfile(trainXandTpath,sprintf('trainChunk_%05d.mat', i)),...
             'X','T');
    
        % Report progress
        if mod(i, 1000) == 0
            fprintf('Saved XTrain & TTrain sample %g of %g.\n', i, numel(XTrainAll))
        end
    end
    
    % Save validation chunks
    for i = 1:numel(XValAll)
        X = XValAll{i}; 
        T = TValAll{i}; 
        save(fullfile(valXandTpath,sprintf('valChunk_%05d.mat',i)),...
             'X','T');
        % Report progress
        if mod(i, 1000) == 0
            fprintf('Saved XValidation & TValidation sample %g of %g.\n', i, numel(XValAll))
        end
    end
    
    clear XTrainBuffered XValBuffered TTrainBuffered TValBuffered featuresVal featuresTrain TValidation
end

%% Set up datastores to allow access to on-disk training and validation data:

% Create a custom read function that returns the data in proper format
readMatFileFunction = @(filename) readMatFileWithCellOutput(filename);

% TRAINING datastore
trainDS = fileDatastore(trainXandTpath, ...
    'ReadFcn', readMatFileFunction, ...
    'FileExtensions', '.mat');

% VALIDATION datastore 
valDS = fileDatastore(valXandTpath, ...
    'ReadFcn', readMatFileFunction, ...
    'FileExtensions', '.mat');


%% Ramdomly plot some training and validation samples & masks

% debug figure - show some sequences, masks, & noise-corrupted sequences 

if plotting == true
    for i = 1:30
        idx = randi(length(XTrainAll));
    
        figure(2)
        tiledlayout(2,1)
        nexttile
        imagesc(XTrainAll{idx})
        set(gca, "YDir", "normal")
        title(["Train # ", num2str(idx), " - Training Subsequence"])
    
        nexttile
        plot(TTrainAll{idx})
        set(gca, "YDir", "normal")
        title(["Train # ", num2str(idx), " - Training Subsequence"])
        ylim([-1.1, 1.1])
    
        waitforbuttonpress
    end
end

%% Memory Management & Cleanup Before Training

% Clear intermediate variables that are no longer needed
clear audioSequencesTrain audioSequencesVal % Original clean sequences
clear audioSeqsNoisyTrain audioSeqsNoisyVal % Noisy sequences 
clear masksTrainPerSample masksValPerSample % Original time domain labels
clear maskTrainPerSamplePadded maskValPerSamplePadded % Padded labels
clear featuresTrain % Raw extracted features
clear XTrainBuffered TTrainBuffered % Buffered training data
clear TTrain XTrainAll TTrainAll TValAll XValAll

% Reset GPU memory if using GPU
if gpuDeviceCount > 0
    % Force garbage collection on GPU
    gpuDevice(maxMemoryDeviceID);
    disp('GPU memory has been reset');
    
    % Display available memory
    gpuInfo = gpuDevice(maxMemoryDeviceID);
    fprintf('Available GPU memory: %.2f GB\n', gpuInfo.AvailableMemory/1e9);
end

% Report memory usage
memStats = memory;
fprintf('CPU Memory Usage: %.2f GB used, %.2f GB available\n', ...
    memStats.MemUsedMATLAB/1e9, memStats.MemAvailableAllArrays/1e9);

% Close any existing parallel pools
if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
end

% Create a fresh parallel pool
parpool('local');

% Compact variables to reduce memory fragmentation
disp('Memory cleanup completed, proceeding to training...');

%% Training

% Validation Frequency (iterations)
numIterationsPerEpoch = ceil(numel(trainDS.Files) / miniBatchSize);
valFreq = round(numIterationsPerEpoch / 8);

% Set training options
options = trainingOptions("adam", ...
    InitialLearnRate = lrInitial, ...
    LearnRateSchedule = "piecewise", ...
    LearnRateDropPeriod = lrDropPeriod, ... 
    LearnRateDropFactor = lrDropFac, ...
    MiniBatchSize = miniBatchSize, ...
    ValidationFrequency = valFreq, ...
    ValidationData = valDS, ...
    ValidationPatience = valPatience, ...
    ExecutionEnvironment = 'parallel-gpu',...
    Verbose = 1, ...
    Plots="none", ...
    MaxEpochs = maxEpochs, ...
    OutputNetwork = "best-validation-loss");
%     Shuffle = "every-epoch", ...

% Load pretrained network
net = audioPretrainedNetwork("vadnet");

% Begin transfer learning
[model.net, model.trainInfo] = trainnet(trainDS, net, "mse", options);

disp('Training Complete.')

%% Prepare model metadata & save trained model

disp('Saving model...')

tsEnd = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm"));

% Preprocessor parameters:
model.preprocParams.fsSource = cleanSignals_fs; % The sample rate of the source audio for the call samples (Hz)
model.preprocParams.fsTarget = fsTarget; % The sample rate at which the features are computed (Hz)
model.preprocParams.bandwidth = bandwidth; % The frequency bandwidth of the features spectrograms [min, max] (Hz)
model.preprocParams.windowDur = windowDur; % The duration of the window function used to compute the STFT (seconds)
model.preprocParams.windowLen = windowLen; % The length of the window function used to compute the STFT (samples)
model.preprocParams.hopDur = hopDur; % The duration of the window hop used to compute the STFT (seconds)
model.preprocParams.hopLen = hopLen; % The length of the window hop used to compute the STFT (samples)
model.preprocParams.overlapDur = overlapLen / cleanSignals_fs; % The duration of the window overlap used to compute the STFT (seconds)
model.preprocParams.overlapLen = overlapLen; % The length of the window overlap used to compute the STFT (samples)
model.preprocParams.overlapPerc = (overlapLen / windowLen) * 100; % The length of the window overlap used to compute the STFT (percent of window length)

% Training Data Construction:
model.dataSynthesisParams.numSequences = numSequences; % The total number of training sequences built
model.dataSynthesisParams.trainPercentage = trainPercentage; % The percentage of "numSequences" used for training (as opposed to validation)
model.dataSynthesisParams.sequenceDuration = sequenceDuration; % The duration of the sequences (s)
model.dataSynthesisParams.snrRange = snrRange; % Range of random SNRs in the synthetic sequences (dB)
model.dataSynthesisParams.minSilenceSegment = minSilenceSegment; % Minimum random duration of silence between concatenated samples in a sequence (s)
model.dataSynthesisParams.nNoiselessSamples = n_noiseless_samples;
model.dataSynthesisParams.nAugmentedCleanSamples = n_training_samples;
model.dataSynthesisParams.nSamplesPerSequence = round(n_training_samples/numSequences);
model.dataSynthesisParams.speedupFactorRange = speedup_factor_range;
model.dataSynthesisParams.pitchShiftRangeHz = [-max_down_shift_hz, max_up_shift_hz];
model.dataSynthesisParams.distortionRange = distortionRange;
model.dataSynthesisParams.dopplerSrcVelocityRange = source_velocity_range;
model.dataSynthesisParams.lpfCutoffRange = lpf_cutoff_range;
model.dataSynthesisParams.reverbDecayTimeRange = decayTimeRange;
model.dataSynthesisParams.transmissionLossStrengthRange = trans_loss_strength_range;
model.dataSynthesisParams.transmissionLossDensityRange = trans_loss_density_range;

% Training Hyper-parameters:
model.trainingHyperparams.miniBatchSize = miniBatchSize; % Number of training samples to run per training iterationmodel.
model.trainingHyperparams.maxEpochs = maxEpochs; % Maximum number runs through the entire dataset
model.trainingHyperparams.valPatience = valPatience; % Number of validations we allow a failure to improve before stopping training
model.trainingHyperparams.valFreq = valFreq; % Frequency of validation tests (iterations)
model.trainingHyperparams.initialLearnRate = lrInitial; % The learning rate schedule name
model.trainingHyperparams.learnRateSchedule = "piecewise"; % The learning rate schedule name
model.trainingHyperparams.learnRateDropPeriod = lrDropPeriod; % The period over which the learning rate drops (Epochs)
model.trainingHyperparams.learnRateDropFactor = lrDropFac; % The factor by which the learning rate drops.

% Training Sequence Chunking:
model.sequenceChunking.analysisTimeStepDuration = analysisTimeStepDuration; % The duration of the overlapping spectrogram chunks (seconds)
model.sequenceChunking.analysisTimeStepLength = analysisTimeStepLength; % The length of the overlapping spectrogram chunks (spectrogram time bins)
model.sequenceChunking.analysisTimeStepOverlapPercent = analysisTimeStepOverlapPercent; % The overlap of the overlapping spectrogram chunks (percent of analysisTimeStepDuration)
model.sequenceChunking.analysisHopLength = analysisHopLength; % The length hop between overlapping spectrogram chunks (spectrogram time bins)

% Training end timestamp
model.trainingCompletionDateTime = tsEnd;

% Save the model with embedded metadata
modelName = ['AVDNet_trained_', tsEnd, '.mat'];
save(fullfile(gavdNetDataPath, modelName), "model");

disp('Done.')

%% Helper Functions

function out = readMatFileWithCellOutput(filename)
    % Load the file
    data = load(filename, 'X', 'T');
    
    % Return as a 1x2 cell array {X, T}
    out = {data.X, data.T};
end
