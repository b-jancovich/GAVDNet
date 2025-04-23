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
% recorded in the format "-2016_"
noiseless_sample_path = "D:\DGS_Chagos_Exemplars\U1 & U2\Denoised";

% Path to folder containing background noise samples (without target calls)
noise_library_path = "D:\DGS_noise_library";

% Output path for trained model and intermediate files
gavdNetDataPath = "D:\GAVDNet_Training\Chagos_DGS";

%% Call Characteristics

% Frequency parameters for the target call
initial_freq = 35.74;        % Mean frequency of the fundamental component (Hz)
initial_freq_year = 2017;    % The year of the initial_freq measurement
pitch_shift_rate = 0.33;     % Annual frequency shift rate (Hz/year)
pitch_shift_tol = 0.1;       % Additional tolerance for pitch shifting (Hz)
detect_year_range = [2000, 2030]; % Time period represented by the synthetic dataset

%% Preprocessing Parameters

% Pre-augmentation processing
preAugfadeIn = 0.2;          % Duration of fade-in (seconds)
preAugfadeOut = 0.2;         % Duration of fade-out (seconds)
target_dynamic_range = 2;  % Target dynamic range (dB)

% Post-augmentation processing 
trim_threshold_ratio = 0.025; % Ratio threshold for silence detection
trim_window_size = 10;        % Sliding window size for silence trimming
postAugfades = 0.2;           % Fade duration after augmentation (seconds)

%% Sequence Construction Parameters

% Parameters for building synthetic training sequences
sequenceDuration = 1000;     % Duration of synthetic sequences (seconds)
minSilenceSegment = 5;       % Minimum silence between samples (seconds)
snrRange = [-20, 10];        % Range of SNRs in training data (dB)

%% Data Augmentation Parameters

% Parameters for augmenting clean samples
outputFs = 200;             % Output sample rate of synthetic samples (Hz)
c = 1500;                   % Typical sound propagation velocity (m/s)
n_training_samples = 12000;  % Number of synthetic examples to generate
speedup_factor_range = [0.95, 1.05]; % Time stretching factor range
lpf_cutoff_range = [38, 50];         % Low-pass filter cutoff range (Hz)
source_velocity_range = [1, 30];      % Source velocity range for Doppler (m/s)
distortionRange = [0.1, 0.4];        % Nonlinear distortion magnitude range
decayTimeRange = [0.1, 3];           % Reverberation decay time range (s)
trans_loss_strength_range = [0.1, 0.15]; % Transmission loss magnitude range
trans_loss_density_range = [0.1, 0.15];  % Transmission loss event density range

%% Neural Network Training Parameters

% Feature extraction parameters for vadnetLFPreprocess
fsTarget = 200;              % Target sample rate for feature extraction (Hz)
bandwidth = [10, 60];        % Frequency bandwidth for spectrograms (Hz)
windowDur = 0.85;            % STFT window duration (seconds)
hopDur = 0.05;               % STFT hop duration (seconds)

% Training hyperparameters
trainPercentage = 85;        % Percentage of data used for training vs. validation
miniBatchSize = 32;          % Number of training samples per iteration
maxEpochs = 9;               % Maximum number of training epochs
valPatience = 5;             % Validation patience (n validation tests)
lrInitial = 0.01;            % Initial learning rate
lrDropPeriod = 2;            % Period for learning rate drop (epochs)
lrDropFac = 1;               % Learning rate drop factor

% Analysis chunking settings
analysisTimeStepDuration = 8;          % Duration of each chunk passed to the network (seconds)
analysisTimeStepOverlapPercent = 0.75; % Overlap of each chunk (percent of analysisTimeStepDuration)