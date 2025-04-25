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

% Path to high-quality, low-noise exemplar recording(s) of target call
% Each file name must contain a string indicating the year it was 
% recorded, in the format "-2016_":
noiseless_sample_path = "D:\DGS_Chagos_Exemplars\U1 & U2\Denoised";

% Path to folder containing background noise samples (target call absent):
% NOTE: noise library audio must have sample rate => sample rate of noiseless sample(s) 
noise_library_path = "D:\DGS_noise_library";

% Output path for trained model and intermediate files:
gavdNetDataPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Training\chagos_DGS_2025";

% Folder containing audio files to test on:
testAudioPath = "D:\PGDetector_Test\DGS_2015_subset";

% Path to "groundtruth" file containing date and time stamps of the true 
% detections of the target call in the test audio files:
groundtruthPath = "D:\PGDetector_Test\DGS_2015_subset\detections_H08S1_DiegoGarciaS_2015_cleaned_subset.mat";

%% Call Characteristics

% Frequency parameters for the target call
initial_freq = 35.74;        % Mean frequency of the fundamental component (Hz)
initial_freq_year = 2017;    % The year of the initial_freq measurement
pitch_shift_rate = 0.33;     % Annual frequency shift rate (Hz/year)
pitch_shift_tol = 0.1;       % Additional tolerance for pitch shifting (Hz)
detect_year_range = [2000, 2030]; % Time period represented by the synthetic dataset

%% Audio Cleanup Parameters

% Pre-augmentation "noiseless_samples" processing
preAugfadeIn = 0.2;          % Duration of fade-in (seconds)
preAugfadeOut = 0.2;         % Duration of fade-out (seconds)
target_dynamic_range = 2;  % Target dynamic range (dB)

% Post-augmentation "cleanSignals" processing 
trim_threshold_ratio = 0.025; % Ratio threshold for silence detection
trim_window_size = 10;        % Sliding window size for silence trimming
postAugfades = 0.2;           % Fade duration after augmentation (seconds)

%% Sequence Construction Parameters

% Parameters for building synthetic training sequences
% callsPerSequence = 10;
numSequences = 400;
sequenceDuration = 3600;     % Duration of synthetic sequences (seconds)
minSilenceSegment = 1;       % Minimum silence between samples (seconds)
snrRange = [-15, 10];        % Range of SNRs in training data (dB)
ICI = 190.79;                % Inter-Call-Interval (seconds) 
ICI_variation = 1.49;        % Inter-Call-Interval +/- variation (seconds)

%% Data Augmentation Parameters

% Parameters for augmenting clean samples
c = 1500;                   % Typical sound propagation velocity (m/s)
speedup_factor_range = [0.97, 1.02]; % Time stretching factor range
lpf_cutoff_range = [38, 50];         % Low-pass filter cutoff range (Hz)
source_velocity_range = [1, 30];      % Source velocity range for Doppler (m/s)
distortionRange = [0.1, 0.4];        % Nonlinear distortion magnitude range
decayTimeRange = [0.1, 3];           % Reverberation decay time range (s)
trans_loss_strength_range = [0.1, 0.3]; % Transmission loss magnitude range
trans_loss_density_range = [0.1, 0.2];  % Transmission loss event density range

%% Neural Network Training Parameters

% Feature extraction parameters for gavdNetPreprocess
fsTarget = 200;              % Target sample rate for feature extraction (Hz)
bandwidth = [10, 60];        % Frequency bandwidth for spectrograms (Hz)
windowDur = 0.85;            % STFT window duration (seconds)
hopDur = 0.05;               % STFT hop duration (seconds)

% Training hyperparameters
trainPercentage = 85;        % Percentage of data used for training vs. validation
miniBatchSize = 12;          % Number of training samples per iteration
maxEpochs = 9;               % Maximum number of training epochs
valPatience = 5;             % Validation patience (n validation tests)
lrInitial = 0.01;            % Initial learning rate
lrDropPeriod = 2;            % Period for learning rate drop (epochs)
lrDropFac = 1;               % Learning rate drop factor

% Feature Framing settings
frameDuration = ICI / 2;    % Duration of each frame passed to the network (seconds)
frameOverlapPercent = 0.5;  % Overlap of each frame (percent of frameDuration)

%% Post processing parameters

postProcOptions.AT = 0.5; % Activation Threshold. Sets the probability 
%                           threshold for starting a vocalisation segment. 
%                           Specify as a scalar in the range [0,1]. If 
%                           unspecified, defaults to 0.5.
%
postProcOptions.DT = 0.25;  % Deactivation Threshold. Sets the probability 
%                           threshold for ending a vocalisation segment. 
%                           Specify as a scalar in the range [0,1]. If 
%                           unspecified, defaults to 0.25.
%
postProcOptions.AEAVD = true; % Apply Energy Animal Vocalisation Detection
%                           Specifies whether to apply an energy-based 
%                           vocalization activity detector to refine the 
%                           regions detected by the neural network. If 
%                           unspecified, ApplyEnergyGAVD defaults to false.
%
postProcOptions.MT = 1;     % Merge Threshold. Merges vocalization regions
%                           that are separated by MT seconds or less. 
%                           Specify as a nonnegative scalar. If unspecified, 
%                           defaults to 0.25 (seconds).
