% GAVDNet Training Configuration File
% B.musculus ssp. brevicauda "Chagos Call" (aka. "Diego Garcia Downsweep")
%
% This is a configuration file for training the GAVDNet model on the Chagos 
% pygmy blue whale call. It provides all parameters needed for the GAVDNet 
% training script.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%% Data Paths and Locations

% Path to high-quality, low-noise exemplar recording(s) of target call to
% be used as the source data to construct the synthetic training sequences.
% Each file name must contain a string indicating the year it was 
% recorded, in the format "-2016_":
noiseless_sample_path = "D:\DGS_Chagos_Exemplars\U1 & U2\Denoised";

% Path to folder containing background noise samples (target call absent):
% NOTE: noise library audio must have sample rate => sample rate of 
% noiseless sample(s) 
noise_library_path = "D:\DGS_noise_library";

% Output path for trained model and intermediate training files:
% gavdNetDataPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\chagos_DGS_2025";
gavdNetDataPath =  "D:\GAVDNet\Chagos_DGS\Training & Models";

% Folder containing audio files to run the detector on:
% inferenceAudioPath = "D:\GAVDNet\Chagos_DGS\Test Data\2007subset";
inferenceAudioPath = "D:\GAVDNet\Chagos_DGS\Test Data\2007subset_small";

% Results path for inference
inferenceOutputPath = "D:\GAVDNet\Chagos_DGS\Test Results";

%% Target Call Characteristics

% Frequency parameters for the target call
initial_freq = 35.74;        % Mean frequency of the fundamental component (Hz)
initial_freq_year = 2017;    % The year of the initial_freq measurement
pitch_shift_rate = 0.33;     % Annual frequency shift rate (Hz/year)
pitch_shift_tol = 0.01;       % Additional random variance in pitch shift (Hz)
detect_year_range = [2006, 2008]; %[2000, 2030]; % Time period represented by the synthetic dataset

%% Noiseless Sample & Augmented Signal Cleanup Parameters

% Pre-augmentation "noiseless_samples" processing
preAugfadeIn = 0.2;          % Duration of fade-in (seconds)
preAugfadeOut = 0.2;         % Duration of fade-out (seconds)
target_dynamic_range = 3;    % Target dynamic range (dB)

% Post-augmentation "cleanSignals" processing 
trim_threshold_ratio = 0.025; % Ratio threshold for silence detection
trim_window_size = 10;        % Sliding window size for silence trimming
postAugfades = 0.2;           % Fade duration after augmentation (seconds)

%% Training Sequence Construction Parameters

sequenceMode = 'multi-call'; % Sets training sequence construction mode. 
%                   Options:
%                       'single-call' - Each sequence has a single call
%                       plus noise. Sequence length is automatically 
%                       set at 3x call length. 
%
%                       'multi-call' - Each sequences has many randomly
%                       placed calls all at different random SNRs.
%                       Sequence duration is set below. System will 
%                       automatically calculate the number calls 
%                       required to build sequences such that 50% of 
%                       their durations are 'call' and 50% 'non-call'.

% Sequence parameters
numSequences = 1200;    % Number of sequences to generate

% these parameters used for 'multi-call' mode only:
sequenceDuration = 1800;% Duration of training sequences to build (seconds)
minCallSeparation = 1;  % Minimum separation between consecutive calls in a sequence (seconds)

%% Data Augmentation Parameters

% Parameters for augmenting clean samples
snrRange = [-3, 15];                    % Range of call SNRs across synthetic training data (dB)
c = 1500;                               % Typical sound propagation velocity (m/s)
speedup_factor_range = [0.96, 1.03];    % Time stretching factor range
lpf_cutoff_range = [42.5, 50];          % Low-pass filter cutoff range (Hz)
hpf_cutoff_range = [10, 30];            % Low-pass filter cutoff range (Hz)   
source_velocity_range = [1, 30];        % Source velocity range for Doppler (m/s)
distortionRange = [0.1, 0.5];           % Nonlinear distortion magnitude range
decayTimeRange = [0.1, 5];              % Reverberation decay time range (s)
trans_loss_strength_range = [0.1, 0.5]; % Transmission loss magnitude range
trans_loss_density_range = [0.1, 0.5];  % Transmission loss event density range

%% Neural Network Parameters

% Feature extraction parameters for gavdNetPreprocess()
fsTarget = 250;           % Target sample rate for feature extraction (Hz)
bandwidth = [15, 50];     % Frequency bandwidth for spectrograms (Hz)
windowDur = 1;            % STFT window duration (seconds)
hopDur = 0.04;            % STFT hop duration (seconds)
saturationRange = 70;     % Spectrograms are saturated to (maxPowVal minus saturationRange)

% % Feature Framing settings for featureBuffer()
frameDuration = 30;       % Duration of each frame passed to the network (seconds)
frameOverlapPercent = 50;  % Overlap of each frame (percent of frameDuration)

% Note: Very long frame durations are not recommended. Long durations 
% increase memory consumption, and in training, can cause instability due 
% to exploding or vanishing gradients. For a target call that is ~30 
% seconds long, recommend training frames of 60-240 seconds with 75% overlap. 

% Training hyperparameters
trainPercentage = 85;        % Percentage of data used for training vs. validation
miniBatchSize = 12;          % Number of training samples per iteration
maxEpochs = 10;              % Maximum number of training epochs
valPerEpoch = 10;            % Number of validation tests per epoch
valPatience = 6;             % Validation patience (n validation tests)
lrInitial = 0.005;           % Initial learning rate
lrDropPeriod = 2;            % Period for learning rate drop (epochs)
lrDropFac = 0.5;             % Learning rate drop factor
l2RegFac = 1e-4;             % L2 Regularization Factor

%% Inference Post-Pprocessing Parameters

postProcOptions.AT = 0.5; % Activation Threshold. Sets the probability 
%                           threshold for starting a vocalisation segment. 
%                           Specify as a scalar in the range [0,1].

postProcOptions.DT = 0.25;  % Deactivation Threshold. Sets the probability 
%                           threshold for ending a vocalisation segment. 
%                           Specify as a scalar in the range [0,1].

postProcOptions.AEAVD = 0; % Apply Energy Animal Vocalisation Detection
%                           Specifies whether to apply an energy-based 
%                           vocalization activity detector to refine the 
%                           regions detected by the neural network.

postProcOptions.MT = 1;   % Merge Threshold. Merges vocalization regions
%                           that are separated by MT seconds or less. 
%                           Specify as a nonnegative scalar.

postProcOptions.LT_scaler = 0.75; % the Length threshold is set based on 
%                           the length of the shortest song in the training
%                           set, scaled by this number

%% Ground Truth Comparison Parameters

detectionTolerance = 30;