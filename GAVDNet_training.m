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
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";
% configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_SORP_BmAntZ.m";

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

% Begin logging
ts = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
logname = ['train_detector_network_log_', ts, '.txt'];
diary(fullfile(gavdNetDataPath, logname));

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

%% Set Output Fs as the sample rate of the files in the noise library

% Use this as the output sample rate of the clean samples
if buildCleanSignals == true
    fprintf('Checking noise library files are all at the same sample rate...\n')
    outputFs = validateTrainingDataFs(noise_library_path);
    fprintf('Noise library sample rate is %g Hz. Synthetic audio data will be built at this rate.\n\n', outputFs)
else
    noiseList = dir(fullfile(noise_library_path, '*.wav'));
    noiseInfo = audioinfo(fullfile(noiseList(1).folder, noiseList(1).name));
    outputFs = noiseInfo.SampleRate;
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

%% Calculate Number of calls to synthesise

switch sequenceMode
    case 'multi-call'
        if buildCleanSignals == true
            % Estimate mean call duration from noiseless samples:
            meanCallDuration = mean(arrayfun(@(x) length(x.compressedAudio), noiseless_samples) / operatingFs);
        
        else
            % Estimate mean call duration from the clean signals already built:
            cleanSignalsList = dir(fullfile(cleanSignalsPath, '*.wav'));
            cleanSignalDurations = zeros(length(cleanSignalsList), 1);
            for i = 1:length(cleanSignalsList)
                sigInfo = audioinfo(fullfile(cleanSignalsList(i).folder, cleanSignalsList(i).name));
                cleanSignalDurations(i) = sigInfo.Duration;
            end
            meanCallDuration = mean(cleanSignalDurations);
        end
        
        % For a balanced dataset, ~50% of the sequence should be calls, and ~50% non-calls
        sequenceDurationWithCalls = sequenceDuration * 0.45;
        
        % Calculate number of calls per sequence considering separation
        % For N calls, we need: N * meanCallDuration + (N-1) * minCallSeparation <= sequenceDurationWithCalls
        % Solving for N: N = (sequenceDurationWithCalls + minCallSeparation) / (meanCallDuration + minCallSeparation)
        numCallsPerSequence = floor((sequenceDurationWithCalls + minCallSeparation) / ...
            (meanCallDuration + minCallSeparation));
        
        % Verify the calculation and warn if sequence might be too short
        totalTimeNeeded = numCallsPerSequence * meanCallDuration + (numCallsPerSequence - 1) * minCallSeparation;
        if totalTimeNeeded > sequenceDurationWithCalls
            warning('Calculated %d calls per sequence requires %.1fs but target is %.1fs. Consider reducing calls or increasing sequence duration.', ...
                numCallsPerSequence, totalTimeNeeded, sequenceDurationWithCalls);
        end
        
        fprintf('Mean call duration: %.2fs\n', meanCallDuration);
        fprintf('Minimum call separation: %.2fs\n', minCallSeparation);
        fprintf('Target sequence duration with calls: %.1fs (%.1f%% of total)\n', ...
            sequenceDurationWithCalls, (sequenceDurationWithCalls/sequenceDuration)*100);
        fprintf('Calculated calls per sequence: %d\n', numCallsPerSequence);
        fprintf('Total time with calls and separations: %.1fs\n', totalTimeNeeded);
        
        % Calculate total number of calls to synthesise
        totalCleanSignalsNeeded = numCallsPerSequence * numSequences;

    case 'single-call'

    % For single call sequences:
    totalCleanSignalsNeeded = numSequences;
    numCallsPerSequence = 1;
end

%% Build augmentation object(s)

% calculate how many augmented copies per noiseless sample 
% (round up to nearest 100)
n_augmented_copies_per_noiseless_sample = 10^2 * ceil(ceil(totalCleanSignalsNeeded / ...
    n_noiseless_samples) / 10^2);

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
            hpf_cutoff_range, decayTimeRange, trans_loss_strength_range, ...
            trans_loss_density_range, ...
            c);
    end
end

%% Build augmented clean signals, post process & save to disk

if buildCleanSignals == true
    disp("Generating augmented clean signal dataset...")
    
    % Run Clean Signal Augmentation on each sample
    clean_signals = table();
    for i = 1:n_noiseless_samples
        augOut = augment(noiseless_samples(i).augmenter, ...
            noiseless_samples(i).compressedAudio, operatingFs);
    
        clean_signals = [clean_signals; augOut];
        fprintf('Finished Generating %d augmented signals for noiseless sample %d.\n', n_augmented_copies_per_noiseless_sample, i)
    end

    cleanSignalDurations = zeros(height(clean_signals), 1);
    for i = 1:height(clean_signals)   
        % Post process the signals (top & tail, DC filt, trim silence from start/end, 
        % fade start/end, highpass filt @ fMin, top & tail, DC Filt again. 
        clean_signals.Audio{i} = postprocessSignals(clean_signals.Audio{i}, operatingFs, outputFs, ...
            trim_threshold_ratio, trim_window_size, postAugfades, bandwidth(1));

        % Save the clean signal durations for later.
        cleanSignalDurations(i) = length(clean_signals.Audio{i}) / outputFs;

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

if ~exist("cleanSignalDurations", "var")
    % If we are using pre-built clean signals already saved to disk, we
    % need to remeasure their durations:
    nCleanSigals = length(ads_cleanSignals.Files);
    cleanSignalDurations = zeros(1, nCleanSigals);
    i = 1;
    while hasdata(ads_cleanSignals)
        [audio, audioInfo] = read(ads_cleanSignals);
        if isValidAudio(audio) == true
            cleanSignalDurations(i) = length(audio) / audioInfo.SampleRate;
            i = i + 1;
        end
    end
    reset(ads_cleanSignals)
end

% Get the clean signals duration statistics
meanCallDuration = mean(cleanSignalDurations);
minCallDuration = min(cleanSignalDurations(cleanSignalDurations > 0));
maxCallDuration = max(cleanSignalDurations);
if strcmp(sequenceMode, 'single-call') == true
    sequenceDuration = maxCallDuration * 3;
end

%% Build Sequences from clean samples & noise, saving them direct to disk

if buildSequences == true
    switch sequenceMode
        case 'single-call'
            % Single call sequences
            constructSingleCallNoisySequences(ads_cleanSignals, ads_noise, ...
                numSequences, snrRange, bandwidth, sequencesPath)
            sequenceDuration = maxCallDuration * 3;
    
        case 'multi-call'
            % Multiple calls per sequence
            constructMultiCallNoisySequences(ads_cleanSignals, ads_noise, ...
                numSequences, numCallsPerSequence, sequenceDuration, ...
                minCallSeparation, snrRange, bandwidth, sequencesPath)
    end
end

%% Set up datastores for pre-saved sequences

% Init the training/validation split
split = [trainPercentage/100, (100-trainPercentage)/100];

% Build sequence datastores
[sequenceDS_train, sequenceDS_val] = createSequenceDatastore(sequencesPath, split);

%% Extract features from training & validation data

% Init STFT parameters (ensuring even values)
windowLen = 2 * round((windowDur * fsTarget) / 2);
hopLen = 2 * round((hopDur * fsTarget) / 2);

% Duration of each STFT frame hop
stftTimeBinDuration = hopLen / fsTarget;

% Number of STFT frames needed to span frameDuration
frameLength = round(frameDuration / stftTimeBinDuration);

% Define frame hop lengths
frameHopLength = round(frameLength * (1 - frameOverlapPercent/100));

% If we haven't already saved the final inputs and targets for training and
% validation:
if buildXandT == true

    % Preprocess training data
    ntrainSequences = length(sequenceDS_train.Files);
    reportInterval = floor(ntrainSequences / 10);
    i = 1;
    while hasdata(sequenceDS_train) 

        % Retrieve audio and mask from sequence
        [sequence, ~] = read(sequenceDS_train);
        audio = sequence.audioSequence;
        mask = sequence.mask;

        switch sequenceMode
            case 'single-call'
                % Run the preprocessor:
                [X, T] = gavdNetPreprocess(audio, outputFs, fsTarget, ...
                    bandwidth, windowLen, hopLen, saturationRange, mask);
        
                % Save the features & mask in one chunk
                save(fullfile(trainXandTpath,sprintf('trainSeq_%05d.mat', i)),...
                         'X','T');

            case 'multi-call'
                        
                % Run the preprocessor:
                [XTrain, TTrain] = gavdNetPreprocess(audio, outputFs, fsTarget, ...
                    bandwidth, windowLen, hopLen, saturationRange, mask);

                % Buffer training features into shorter frames, and 
                % standardize each frame to its local statistics:
                XTrainBuffered = featureBuffer(XTrain, frameLength, ...
                    frameOverlapPercent);

                % Buffer the masks in an identical manner:
                TTrainBuffered = maskBuffer(TTrain, frameLength, ...
                    frameOverlapPercent);

                % Save each frame and mask to disk as a separate chunk
                for j = 1:numel(XTrainBuffered)
                    X = XTrainBuffered{j}; 
                    T = TTrainBuffered{j}; 
                    save(fullfile(trainXandTpath,sprintf('trainSeq_%05d_frame_%05d.mat', i, j)),...
                         'X','T');
                end
        end

        % Report progress
        if mod(i, reportInterval) == 0
            fprintf('Completed preprocessing and feature extraction on training sequence %g of %g.\n', i, ntrainSequences)
        end
        i = i+1;
    end
    
    % Preprocess validation data
    nValSequences = length(sequenceDS_val.Files);
    reportInterval = floor(nValSequences / 10);
    i = 1;
    while hasdata(sequenceDS_val) 
        
        % Retrieve audio and mask from sequence
        [sequence, ~] = read(sequenceDS_val);
        audio = sequence.audioSequence;
        mask = sequence.mask;       
        
        switch sequenceMode
            case 'single-call'
                % Run the preprocessor:
                [X, T] = gavdNetPreprocess(audio, outputFs, fsTarget, ...
                    bandwidth, windowLen, hopLen, saturationRange, mask);
        
                % Save the features & mask in one chunk
                save(fullfile(valXandTpath,sprintf('valSeq_%05d.mat', i)),...
                         'X','T');

            case 'multi-call'
                        
                % Run the preprocessor:
                [XVal, TVal] = gavdNetPreprocess(audio, outputFs, fsTarget, ...
                    bandwidth, windowLen, hopLen, saturationRange, mask);

                % Buffer training features into shorter frames, and 
                % standardize each frame to its local statistics:
                XValBuffered = featureBuffer(XVal, frameLength, ...
                    frameOverlapPercent);

                % Buffer the masks in an identical manner:
                TValBuffered = maskBuffer(TVal, frameLength, ...
                    frameOverlapPercent);

                % Save each frame and mask to disk as a separate chunk
                for j = 1:numel(XValBuffered)
                    X = XValBuffered{j}; 
                    T = TValBuffered{j}; 
                    save(fullfile(valXandTpath,sprintf('valSeq_%05d_frame_%05d.mat', i, j)),...
                         'X','T');
                end
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

%% Populate Metadata to output struct

% Training Data Construction:
model.dataSynthesisParams.sequenceMode = sequenceMode;
model.dataSynthesisParams.numSequences = numSequences;
model.dataSynthesisParams.numCallsPerSequence = numCallsPerSequence; 
model.dataSynthesisParams.sequenceDuration = sequenceDuration; 
model.dataSynthesisParams.nNoiselessSamples = n_noiseless_samples;
model.dataSynthesisParams.trainPercentage = trainPercentage; % The percentage of "numSequences" used for training (as opposed to validation)
model.dataSynthesisParams.snrRange = snrRange; % Range of random SNRs in the synthetic sequences (dB)
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

% Preprocessor parameters:
model.preprocParams.fsSource = outputFs; % The final sample rate of the augmented call samples (Hz)
model.preprocParams.fsTarget = fsTarget; % The sample rate at which the features are computed (Hz)
model.preprocParams.bandwidth = bandwidth; % The frequency bandwidth of the features spectrograms [min, max] (Hz)
model.preprocParams.windowDur = windowDur; % The duration of the window function used to compute the STFT (seconds)
model.preprocParams.windowLen = windowLen; % The length of the window function used to compute the STFT (samples)
model.preprocParams.hopDur = hopDur; % The duration of the window hop used to compute the STFT (seconds)
model.preprocParams.hopLen = hopLen; % The length of the window hop used to compute the STFT (samples)
model.preprocParams.saturationRange = saturationRange;
model.preprocParams.targetCallDuration = meanCallDuration;

% % Training Sequence Framing:
model.featureFraming.frameDuration = frameDuration; % The duration of the overlapping spectrogram frames (seconds)
model.featureFraming.frameLength = frameLength; % The length of the overlapping spectrogram frames (spectrogram time bins)
model.featureFraming.frameOverlapPercent = frameOverlapPercent; % The overlap of the overlapping spectrogram frames (percent of frameDuration)
model.featureFraming.frameHopLength = frameHopLength; % The length hop between overlapping spectrogram frames (spectrogram time bins)

% Training Hyper-parameters:
model.trainingHyperparams.miniBatchSize = miniBatchSize; % Number of training samples to run per training iterationmodel.
model.trainingHyperparams.maxEpochs = maxEpochs; % Maximum number runs through the entire dataset
model.trainingHyperparams.numIterationsPerEpoch = ceil(numel(trainDS.Files) / miniBatchSize);
model.trainingHyperparams.valPatience = valPatience; % Number of validations we allow a failure to improve before stopping training
model.trainingHyperparams.valFreq = round(model.trainingHyperparams.numIterationsPerEpoch / valPerEpoch);
model.trainingHyperparams.initialLearnRate = lrInitial; % The learning rate schedule name
model.trainingHyperparams.learnRateSchedule = "piecewise"; % The learning rate schedule name
model.trainingHyperparams.learnRateDropPeriod = lrDropPeriod; % The period over which the learning rate drops (Epochs)
model.trainingHyperparams.learnRateDropFactor = lrDropFac; % The factor by which the learning rate drops.
model.trainingHyperparams.l2RegFac = l2RegFac; % L2 Regularization Factor

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
fprintf('Memory cleanup completed, proceeding to training...\n\n');

%% Training

% Load pretrained network
net = audioPretrainedNetwork("vadnet", Weights="pretrained");

% Report training hyperparams
fprintf('Total training samples: %d\n', numel(trainDS.Files));
fprintf('Mini-batch size: %d\n', model.trainingHyperparams.miniBatchSize);
fprintf('Iterations per epoch: %d\n', model.trainingHyperparams.numIterationsPerEpoch);
fprintf('Max Epochs: %d\n', model.trainingHyperparams.maxEpochs);
fprintf('Running Validation every %d iterations.\n', model.trainingHyperparams.valFreq);

% Set training options 
% Note: Phase 1 validation patience is 50% of the value specified in config:
options = trainingOptions("adam", ...
    InitialLearnRate = model.trainingHyperparams.initialLearnRate, ...
    LearnRateSchedule = model.trainingHyperparams.learnRateSchedule, ...
    LearnRateDropPeriod = model.trainingHyperparams.learnRateDropPeriod, ... 
    LearnRateDropFactor = model.trainingHyperparams.learnRateDropFactor, ...
    L2Regularization = model.trainingHyperparams.l2RegFac, ...
    MiniBatchSize = model.trainingHyperparams.miniBatchSize, ...
    ValidationData = valDS, ...
    ValidationFrequency = model.trainingHyperparams.valFreq, ...
    ValidationPatience = round(model.trainingHyperparams.valPatience/2), ...
    Shuffle = "every-epoch", ...
    Verbose = 1, ...
    Plots = "none", ...
    MaxEpochs = model.trainingHyperparams.maxEpochs, ...
    SequenceLength = 'shortest');

% Progress update:
fprintf('\nPhase 1: Training with GRU layers frozen...\n');

% Explicitly freeze GRU layers
net = setGRULearnRate(net, 0);

% Transfer learning on CNN/DNN layers only
[net, model.trainInfo_phase1] = trainnet(trainDS, net, "binary-crossentropy", options);

% Progress update:
fprintf('Phase 1 Training Complete.\n\n');
fprintf('Phase 2: Unfreezing GRU layers and continuing training...\n');

% Unfreeze the GRU layers
net = setGRULearnRate(net, 1);

% Set Phase 2 validation patience to 100% of the value specified in config:
options.ValidationPatience = model.trainingHyperparams.valPatience;

% Transfer learning on all layers (incl. GRU's)
[model.net, model.trainInfo_phase2] = trainnet(trainDS, net, "binary-crossentropy", options);

disp('Training Complete.')

%% Evaluate training

figure(1)
tiledlayout(2,1)
nexttile
plot(model.trainInfo_phase1.TrainingHistory.Iteration, model.trainInfo_phase1.TrainingHistory.Loss)
grid on
hold on
plot(model.trainInfo_phase1.ValidationHistory.Iteration, model.trainInfo_phase1.ValidationHistory.Loss)
xlabel('Iteration')
ylabel('Loss (Binary-Crossentropy)')
legend({'Training', 'Validation'})
title('Phase 1 - CNN & DNN Layers')
nexttile
plot(model.trainInfo_phase2.TrainingHistory.Iteration, model.trainInfo_phase2.TrainingHistory.Loss)
grid on
hold on
plot(model.trainInfo_phase2.ValidationHistory.Iteration, model.trainInfo_phase2.ValidationHistory.Loss)
xlabel('Iteration')
ylabel('Loss (Binary-Crossentropy)')
legend({'Training', 'Validation'})
title('Phase 2 - All Layers')
sgtitle('Training & Validation Loss')

% Evaluate training info:
fprintf('\nAnalysing Phase 1 Training...\n')
model.trainInfoEval_phase1 = analyzeTrainingInfo(model.trainInfo_phase1);
fprintf('\nAnalysing Phase 2 Training...\n')
model.trainInfoEval_phase2 = analyzeTrainingInfo(model.trainInfo_phase2);

%% Save trained model

fprintf('\nSaving model to disk...\n')

% Training end timestamp
tsEnd = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm"));
model.trainingCompletionDateTime = tsEnd;

% Save the model with embedded metadata
modelName = ['GAVDNet_trained_', tsEnd, '.mat'];
save(fullfile(gavdNetDataPath, modelName), "model");

disp('Done.')
diary off

%% Helper Functions0

function out = readMatFileWithCellOutput(filename)
    % Load the file
    data = load(filename, 'X', 'T');

    % If the target is a logical, convert to double.
    if islogical(data.T)
        data.T = single(data.T);
    end

    % Return as a 1x2 cell array {X, T}
    out = {data.X, data.T};
end

function net = setGRULearnRate(net, lrScaleFac)
% Set the learning rate scaling factor for the network's GRU layers.
% lrScaleFac = 0 freezes the layers
% lrScaleFac = 1 un-freezes the layers
    gruLayerNames = {'gru1.forward', 'gru1.reverse', 'gru2.forward', 'gru2.reverse'};
    for i = 1:length(gruLayerNames)
        net = setLearnRateFactor(net, gruLayerNames{i}, 'InputWeights', lrScaleFac);
        net = setLearnRateFactor(net, gruLayerNames{i}, 'RecurrentWeights', lrScaleFac);
        net = setLearnRateFactor(net, gruLayerNames{i}, 'Bias', lrScaleFac);
    end
end