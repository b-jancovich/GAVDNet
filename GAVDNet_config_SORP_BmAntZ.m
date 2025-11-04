% GAVDNet Training Configuration File
% B.musculus intermedia "Z Call"
%
% This is a configuration file for training the GAVDNet model on the
% Antactic blue whale Z-call. It provides all parameters needed for the 
% GAVDNet training script.
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
noiseless_sample_path = "D:\SORP_BmAntZ_exemplars\Denoised";

% Path to folder containing background noise samples (target call absent):
% NOTE: noise library audio must have sample rate => sample rate of 
% noiseless sample(s) 
noise_library_path = "D:\SORP_BmAntZ_noise_library";

% Output path for trained model and intermediate training files:
gavdNetDataPath = "D:\GAVDNet\BmAntZ_SORP\Training & Models\-10 to 10 Single Exemplar";

% Results path for inference
inferenceOutputPath = "D:\GAVDNet\BmAntZ_SORP\Test Results\Final Test - Casey2014\-10 to 10 Single Exemplar";

% Folder containing audio files to run the detector on:
inferenceAudioPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Detector Test Datasets\AAD_AcousticTrends_BlueFinLibrary\DATA\casey2014\wav"; 


%% Target Call Characteristics

% The synthetic data system models the downward frequency shift phenomenon
% seen in the song of most (possibly all) Blue whale populations. If your
% target call does not frequency shift, set pitch shift rate to zero and
% use "freq_shift_tol" to set the maximum range of frequency variation.

% Frequency parameters for the target call
initial_freq = 25.8;        % Mean frequency of the fundamental component (Hz)
initial_freq_year = 2015;    % The year of the initial_freq measurement
freq_shift_rate = 0.135;     % Annual frequency decline rate (Hz/year)
freq_shift_tol = 0.5;       % Additional range of variation in pitch shifting to account for intra-seasonal shift (Hz)
detect_year_range = [2013, 2015]; % Time period represented by the synthetic dataset

%% Input Audio Cleanup Parameters

% Pre-augmentation "noiseless_samples" processing
preAugfadeIn = 0.2;          % Duration of fade-in (seconds)
preAugfadeOut = 0.2;         % Duration of fade-out (seconds)
target_dynamic_range = 2;  % Target dynamic range (dB)

% Post-augmentation "cleanSignals" processing 
trim_threshold_ratio = 0.2; % Ratio threshold for silence detection
trim_window_size = 25;        % Sliding window size for silence trimming
postAugfades = 0.2;           % Fade duration after augmentation (seconds)

%% Data Augmentation Parameters

% Parameters for augmenting clean samples
c = 1500;                               % Typical sound propagation velocity (m/s)
speedup_factor_range = [0.97, 1.03];    % Time stretching factor range
lpf_cutoff_range = [];                  % Low-pass filter cutoff range (Hz)
hpf_cutoff_range = [];                  % Low-pass filter cutoff range (Hz)
source_velocity_range = [1, 8.3];        % Source velocity range for Doppler (m/s)
distortionRange = [0.1, 0.5];           % Nonlinear distortion magnitude range
decayTimeRange = [0.1, 10];             % Reverberation decay time range (s)
trans_loss_strength_range = [0.1, 0.75];% Transmission loss magnitude range
trans_loss_density_range = [0.1, 0.5];  % Transmission loss event density range
end_trim_duration_range = [0.1, 1];     % Maximum duration of signal to 
%                                       randomly remove from the end of
%                                       clean signals (s)

%% Training Sequence Construction Parameters

% Parameters for building synthetic training sequences
snrRange = [-10, 10];       % Range of randomly set Signal to Noise ratios for calls in training sequences (dB)
numSequences = 1200;        % Number of sequences to generate
sequenceDuration = 1800;    % Duration of training sequences to build (seconds)
minCallSeparation = 1;      % Minimum separation between consecutive calls in a sequence (seconds)

% NOTE: The number of calls per sequence is calculated automatically to 
% ensure that approximately 50% of every sequence's duration contains the 
% call, and 50% does not.

%% Neural Network Training Parameters

% Feature extraction parameters for gavdNetPreprocess
fsTarget = 250;              % Target sample rate for feature extraction (Hz)
bandwidth = [10, 50];        % Frequency bandwidth for spectrograms (Hz)
windowDur = 0.85;            % STFT window duration (seconds)
hopDur = 0.05;               % STFT hop duration (seconds)
saturationRange = 70;        % The dynamic range to saturate spectrograms to (dB)

% Feature Framing settings
frameDuration = 60;         % Duration of each frame passed to the network (seconds)
frameOverlapPercent = 50;  % Overlap of each frame (percent of frameDuration)

% Training hyperparameters
trainPercentage = 85;        % Percentage of data used for training vs. validation
miniBatchSize = 12;          % Number of training samples per iteration
maxEpochs = 7;               % Maximum number of training epochs
valPatience = 7;             % Validation patience (n validation tests)
lrInitial = 0.005;           % Initial learning rate
lrDropPeriod = 2;            % Period for learning rate drop (epochs)
lrDropFac = 0.5;             % Learning rate drop factor
l2RegFac = 1e-4;             % L2 Regularization Factor

%% Audio Pre-Processing Parameters

    featureFraming = 'event-split'; % Different modes for splitting long inputs. 
% Options: 
% 'none'          - Computes the spectrogram for the whole audio 
%                   file, and runs this through the network in one pass.
% 'simple'        - Computes the spectrogram for the whole audio file,
%                   and breaks it into frames of same size and overlap 
%                   as the training data frames.
% 'event-splt'    - Uses signal statistics to find local regions of 
%                   the audio file that have very high energy peaks, 
%                   and splits the file based on changes in the mean 
%                   of the signal envelope.
%                   (Inference only)

minSilenceDuration = 1; % Silence causes the detector to return garbage. 
%                       There is a silence detector that returns sample 
%                       indices of silent or near-silent regions of the
%                       file before preprocessing so that any detections 
%                       from within these times can be ignored. This 
%                       variable sets the largest duration of audio that 
%                       may be 'silent' without being flagged as a silent
%                       region. Suggested value = 1 (seconds)
%                       (Inference only)

frameStandardization = 'true'; % Sets whether the frequency bins of the 
%                               spectrogram frames are re-standardized to
%                               to their local, frame-level statistics.
%                               This setting always applies to training 
%                               and applies to inference, IF running in 
%                               "event-split" and "simple" feature framing
%                               modes, but does not apply if feature 
%                               framing is set to 'none'.
%                               (Training AND Inference)

%% Inference Post-Processing Parameters

postProcOptions.AT = 0.5; % Activation Threshold. Sets the probability 
%                           threshold for starting a vocalisation segment. 
%                           Specify as a scalar in the range [0,1].

postProcOptions.DT = 0.001;  % Deactivation Threshold. Sets the probability 
%                           threshold for ending a vocalisation segment. 
%                           Specify as a scalar in the range [0,1].

postProcOptions.AEAVD = 0; % Apply Energy Animal Vocalisation Detection
%                           Specifies whether to apply an energy-based 
%                           vocalization activity detector to refine the 
%                           regions detected by the neural network.

postProcOptions.MT = 0.1;   % Merge Threshold. Merges vocalization regions
%                           that are separated by MT seconds or less. 
%                           Specify as a nonnegative scalar.

postProcOptions.LT_scaler = 0.5; % the Length threshold is set based on 
%                           the length of the shortest song in the training
%                           set, scaled by this number

%% Ground Truth Comparison Parameters

detectionTolerance = 30;