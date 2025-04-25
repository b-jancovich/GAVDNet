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
%   4. Extract spectrogram features using gavdnetPreprocess
%   5. Apply transfer learning to the pretrained VAD network
%   6. Save the trained model with metadata
%
% REQUIREMENTS:
%   - MATLAB R2024a or later
%   - Audio Toolbox
%   - Deep Learning Toolbox
%   - Parallel Computing Toolbox (recommended for speed)
%   - customAudioAugmenter github repository synced and in the same git
%   root folder as the GAVDNet repo.
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

plotting = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.
%% Set Paths and Load Input Variables 

% Add dependencies to path
run(configPath) % Load config file
projectRoot = pwd;
[gitRoot, ~, ~] = fileparts(projectRoot);
addpath(fullfile(projectRoot, "Functions"))
addpath(fullfile(gitRoot, 'customAudioAugmenter'));

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
    fprintf('Found %g pre-saved sequences. Using existing sequences.\n', ....
        size(dir(fullfile(sequencesPath, '*.mat')), 1))
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
fprintf('\n')

%% Set up for GPU or CPU processing

[useGPU, gpuDeviceID, ~] = gpuConfig();

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

%% Get the sample rate of the noise library

% Use this as the output sample rate of the clean samples
fprintf('Checking noise library files are all at the same sample rate...\n')
outputFs = validateTrainingDataFs(noise_library_path);
fprintf('Noise library sample rate is %g Hz. Synthetic audio data will be built at this rate.\n\n', outputFs)

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

% Estimate how many augmented samples we will need to generate the call sequences:
n_augmented_Samples_total = estimateCleanSamplesNeeded(numSequences, sequenceDuration, ICI, ICI_variation);

% calculate how many augmented copies per noiseless sample 
% (round up to nearest 100)
n_augmented_copies_per_noiseless_sample = 10^2 * ceil(ceil(n_augmented_Samples_total / n_noiseless_samples) / 10^2);

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
        noiseless_samples(i).augmenter = setupAugmenter(operatingFs, n_augmented_copies_per_noiseless_sample, ...
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
        fprintf('Finished Generating %d augmented signals for noiseless sample %d.\n', n_augmented_copies_per_noiseless_sample, i)
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

else
    % Just get the sampling frequency:
    % Build audioDataStore containing training samples of the target sound
    ads_cleanSignals = audioDatastore(cleanSignalsPath, "IncludeSubfolders", true,...
        "FileExtensions",".wav","LabelSource","foldernames", ...
        OutputDataType="single");
    [~, ads_info] = read(ads_cleanSignals);
    cleanSignals_fs = ads_info.SampleRate;
end

% Get clean signal lengths
nCleanSigals = length(ads_cleanSignals.Files);
cleanSignalDurations = zeros(1, nCleanSigals);
i = 1;
while hasdata(ads_cleanSignals)
    [audio, audioInfo] = read(ads_cleanSignals);
    cleanSignalDurations(i) = length(audio) / audioInfo.SampleRate;
    i = i+1;
end
reset(ads_cleanSignals)

% Get the minimum, mean & max lenght of clean signals
meanCallDuration = mean(cleanSignalDurations);
minCallDuration = min(cleanSignalDurations);
maxCallDuration = max(cleanSignalDurations);

%% Build Sequences from clean samples & noise, saving them direct to disk

if buildSequences == true
    % If we haven't already built the sequences of calls, build now
    constructNoisySequences(ads_cleanSignals, ads_noise, numSequences, ...
        sequenceDuration, snrRange, ICI, ICI_variation, ...
        sequencesPath);
end

%% Set up datastores for pre-saved sequences

% Init the training/validation split
split = [trainPercentage/100, (100-trainPercentage)/100];

% Build sequence datastores
[sequenceDS_train, sequenceDS_val] = createSequenceDatastore(sequencesPath, split);

% Display some sequences, masks, & noise-corrupted sequences 
if plotting == true
    for i = 1:10
        % Read a training sequence
        [sequenceTrain, ~] = read(sequenceDS_train);
        audioSeqTrain = sequenceTrain.audioSequence;
        maskTrain = sequenceTrain.mask;

        % Read a validation sequence
        [sequenceVal, ~] = read(sequenceDS_val);
        audioSeqVal = sequenceVal.audioSequence;
        maskVal = sequenceVal.mask;

        % Convert masks to sigroi mask format
        maskTrain = signalMask(gather(maskTrain), SampleRate=outputFs);
        maskVal = signalMask(gather(maskVal), SampleRate=outputFs);
        
        % Draw Figure
        figure(1)
        tiledlayout(1,2)
        nexttile
        plotsigroi(maskTrain, gather(audioSeqTrain), true)
        title(["Train sequence # ", num2str(i), " - Noisy Sequence & Mask"])
        nexttile
        plotsigroi(maskVal, gather(audioSeqVal), true)
        title(["Validation sequence # ", num2str(i), " - Noisy Sequence & Mask"])
        waitforbuttonpress
    end
end

% Reset datastores after plotting examples
reset(sequenceDS_train)
reset(sequenceDS_val)

%% Extract features from training & validation data

% Init STFT and framing parameters 
windowLen = windowDur * fsTarget;
hopLen = hopDur * fsTarget;
overlapLen = windowLen - hopLen;
frameHopLength = windowLen - overlapLen;
frameStepLength = round(frameDuration * fsTarget / frameHopLength);

% If we haven't already saved the final inputs and targets for training and
% validation:
if buildXandT == true

    % Training data
    ntrainSequences = length(sequenceDS_train.Files);
    reportInterval = floor(ntrainSequences / 10);
    i = 1;
    while hasdata(sequenceDS_train) 
        % Retrieve audio and mask from sequence
        [sequence, ~] = read(sequenceDS_train);
        audio = sequence.audioSequence;
        mask = sequence.mask;
        
        % Run preprocessor:
        [XTrain, TTrain] = gavdNetPreprocess(audio, outputFs, fsTarget, bandwidth, windowDur, hopDur, mask);
    
        % Buffer training features into frames
        XTrainBuffered = featureBuffer(XTrain, frameStepLength, ...
            frameOverlapPercent);
        
        % Buffer training masks into frames
        TTrainBuffered = featureBuffer(TTrain, frameStepLength, ...
            frameOverlapPercent);

        % Save train frame
        for j = 1:numel(XTrainBuffered)
            X = XTrainBuffered{j}; 
            T = TTrainBuffered{j}; 
            save(fullfile(trainXandTpath,sprintf('trainSeq_%05d_frame_%05d.mat', i, j)),...
                 'X','T');
        end

        % Report progress
        if mod(i, reportInterval) == 0
            fprintf('Completed preprocessing and feature extraction on training sequence %g of %g.\n', i, ntrainSequences)
        end
        i = i+1;
    end
    
    % Validation data
    nValSequences = length(sequenceDS_val.Files);
    reportInterval = floor(nValSequences / 10);
    i = 1;
    while hasdata(sequenceDS_val) 
        % Retrieve audio and mask from sequence
        [sequence, ~] = read(sequenceDS_val);
        audio = sequence.audioSequence;
        mask = sequence.mask;
        
        % Run preprocessor:
        [XVal, TVal] = gavdNetPreprocess(audio, outputFs, fsTarget, bandwidth, windowDur, hopDur, mask);

        % Buffer training features into frames
        XValBuffered = featureBuffer(XVal, frameStepLength, ...
            frameOverlapPercent);
        
        % Buffer training masks into frames
        TValBuffered = featureBuffer(TVal, frameStepLength, ...
            frameOverlapPercent);

        % Save train frame
        for j = 1:numel(XValBuffered)
            X = XValBuffered{j}; 
            T = TValBuffered{j}; 
            save(fullfile(valXandTpath,sprintf('valSeq_%05d_frame_%05d.mat', i, j)),...
                 'X','T');
        end

        % Report progress
        if mod(i, reportInterval) == 0
            fprintf('Completed preprocessing and feature extraction on validation sequence %g of %g.\n', i, nValSequences)
        end
        i = i+1;
    end
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

%% Prep output struct

% Validation Frequency (iterations)
numIterationsPerEpoch = ceil(numel(trainDS.Files) / miniBatchSize);
valFreq = round(numIterationsPerEpoch / 10);

% Preprocessor parameters:
model.preprocParams.fsSource = outputFs; % The final sample rate of the augmented call samples (Hz)
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
model.dataSynthesisParams.nAugmentedCleanSamples = n_augmented_Samples_total;
model.dataSynthesisParams.nSamplesPerSequence = round(n_augmented_Samples_total/numSequences);
model.dataSynthesisParams.speedupFactorRange = speedup_factor_range;
model.dataSynthesisParams.pitchShiftRangeHz = [-max_down_shift_hz, max_up_shift_hz];
model.dataSynthesisParams.distortionRange = distortionRange;
model.dataSynthesisParams.dopplerSrcVelocityRange = source_velocity_range;
model.dataSynthesisParams.lpfCutoffRange = lpf_cutoff_range;
model.dataSynthesisParams.reverbDecayTimeRange = decayTimeRange;
model.dataSynthesisParams.transmissionLossStrengthRange = trans_loss_strength_range;
model.dataSynthesisParams.transmissionLossDensityRange = trans_loss_density_range;
model.dataSynthesisParams.meanTargetCallDuration = meanCallDuration;
model.dataSynthesisParams.minTargetCallDuration = minCallDuration;
model.dataSynthesisParams.maxTargetCallDuration = maxCallDuration;

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
model.featureFraming.frameDuration = frameDuration; % The duration of the overlapping spectrogram frames (seconds)
model.featureFraming.frameStepLength = frameStepLength; % The length of the overlapping spectrogram frames (spectrogram time bins)
model.featureFraming.frameOverlapPercent = frameOverlapPercent; % The overlap of the overlapping spectrogram frames (percent of frameDuration)
model.featureFraming.frameHopLength = frameHopLength; % The length hop between overlapping spectrogram frames (spectrogram time bins)

%% Memory Management & Cleanup Before Training

% Clear intermediate variables that are no longer needed
clearvars -except gpuDeviceID trainDS valDS model gavdNetDataPath

% Reset GPU memory if using GPU
if gpuDeviceCount > 0
    % Force garbage collection on GPU
    gpuDevice(gpuDeviceID);
    disp('GPU memory has been reset');
    
    % Display available memory
    gpuInfo = gpuDevice(gpuDeviceID);
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
disp(newline)

%% Training

% Set training options
options = trainingOptions("adam", ...
    InitialLearnRate = model.trainingHyperparams.initialLearnRate, ...
    LearnRateSchedule = model.trainingHyperparams.learnRateSchedule, ...
    LearnRateDropPeriod = model.trainingHyperparams.learnRateDropPeriod, ... 
    LearnRateDropFactor = model.trainingHyperparams.learnRateDropFactor, ...
    MiniBatchSize = model.trainingHyperparams.miniBatchSize, ...
    ValidationFrequency = model.trainingHyperparams.valFreq, ...
    ValidationData = valDS, ...
    ValidationPatience = model.trainingHyperparams.valPatience, ...
    Shuffle = "every-epoch", ...
    ExecutionEnvironment = 'parallel-gpu',...
    Verbose = 1, ...
    Plots = "none", ...
    MaxEpochs = model.trainingHyperparams.maxEpochs, ...
    OutputNetwork = "best-validation-loss");

% Load pretrained network
net = audioPretrainedNetwork("vadnet");

% Begin transfer learning
[model.net, model.trainInfo] = trainnet(trainDS, net, "mse", options);

disp('Training Complete.')

%% Prepare model metadata & save trained model

disp('Saving model...')

% Training end timestamp
tsEnd = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm"));
model.trainingCompletionDateTime = tsEnd;

% Save the model with embedded metadata
modelName = ['GAVDNet_trained_', tsEnd, '.mat'];
save(fullfile(gavdNetDataPath, modelName), "model");

disp('Done.')

%% Helper Functions

function out = readMatFileWithCellOutput(filename)
    % Load the file
    data = load(filename, 'X', 'T');
    
    % Return as a 1x2 cell array {X, T}
    out = {data.X, data.T};
end
