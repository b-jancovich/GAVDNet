% GAVDNet Training Configuration File (chorus DISABLED variant)
% B.musculus ssp. brevicauda "Chagos Song" (aka. "Diego Garcia Downsweep")
%
% Identical to GAVDNet_config_DGS_chagos.m except that synthetic chorus
% injection is turned off (enableChorus = false). Use this config to
% train a baseline detector without chorus injection, for A/B comparison
% against a chorus-aware model trained with GAVDNet_config_DGS_chagos.m.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%% Data Paths and Locations

% Path to high-quality, low-noise exemplar recording(s) of target call to
% be used as the source data to construct the synthetic training sequences.
% May be a file path or a folder path. If folder, all wav files in folder
% will be used. Each file name must contain a string indicating the year it
% was recorded, in the format "-2016_". This is used to calculate correct
% annual pitch shift based on known pitch decline rate for the target call.
% noiseless_sample_path = "E:\DGS_Chagos_Exemplars\U1 & U2\Denoised";
noiseless_sample_path = "E:\DGS_Chagos_Exemplars\U1 & U2\Denoised\detectionAudio_21560_19-Aug-2005_19_35_36_24.295708_RXDENOISED.wav";

% Path to folder containing background noise samples (target call absent):
% NOTE: noise library audio must have sample rate => sample rate of
% noiseless sample(s)
noise_library_path = "E:\DGS_noise_library";

% Output path for trained model and intermediate training files:
gavdNetDataPath = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar Exclude Chorus";

% Folder containing audio files to run the detector on:
inferenceAudioPath = "E:\GAVDNet\Chagos_DGS\Test Data\2007subset";

% Results path for inference
inferenceOutputPath = "E:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-10 to 10 Single Exemplar Exclude Chorus";

%% Target Call Characteristics

% The synthetic data system models the downward frequency shift phenomenon
% seen in the song of most (possibly all) Blue whale populations. If your
% target call does not frequency shift, set pitch shift rate to zero and
% use "freq_shift_tol" to set the maximum range of frequency variation.

% Frequency parameters for the target call
initial_freq = 32.97;        % Mean frequency of the fundamental component (Hz)
initial_freq_year = 2017;    % The year of the initial_freq measurement
freq_shift_rate = 0.33;     % Annual frequency decline rate (if none, set to 0) (Hz/year)
freq_shift_tol = 0.5;       % Additional tolerance for pitch shifting (Hz)
detect_year_range = [2006, 2008]; % Time period represented by the synthetic dataset

%% Input Audio Cleanup Parameters

% Pre-augmentation "noiseless_samples" processing
preAugfadeIn = 0.2;          % Duration of fade-in (seconds)
preAugfadeOut = 0.2;         % Duration of fade-out (seconds)
target_dynamic_range = 2;    % Target dynamic range (dB)

% Post-augmentation "cleanSignals" processing
trim_threshold_ratio = 0.2;   % Ratio threshold for silence detection
trim_window_size = 25;        % Sliding window size for silence trimming (samples)
postAugfades = 0.2;           % Fade duration after augmentation (seconds)

%% Data Augmentation Parameters

% Parameters for augmenting clean samples
c = 1500;                               % Typical sound propagation velocity (m/s)
speedup_factor_range = [0.97, 1.03];    % Time stretching factor range
lpf_cutoff_range = [37, 50];            % Low-pass filter cutoff range (Hz)
hpf_cutoff_range = [10, 33];            % Low-pass filter cutoff range (Hz)
source_velocity_range = [1, 8.3];        % Source velocity range for Doppler (m/s)
distortionRange = [0.1, 0.5];           % Nonlinear distortion magnitude range
decayTimeRange = [0.1, 10];              % Reverberation decay time range (s)
trans_loss_strength_range = [0.1, 0.75]; % Transmission loss magnitude range
trans_loss_density_range = [0.1, 0.5];  % Transmission loss event density range
end_trim_duration_range = [0.1, 10];    % Maximum duration of signal to
%                                       randomly remove from the end of
%                                       clean signals (s)

%% Training Sequence Construction Parameters

% Parameters for building synthetic training sequences
snrRange = [-10, 10];       % Range of randomly set Signal to Noise ratios for calls in training sequences (dB)
numSequences = 1200;        % Number of sequences to generate
sequenceDuration = 1800;    % Duration of training sequences to build (seconds)
minCallSeparation = 0.5;    % Minimum separation between consecutive calls in a sequence (seconds)

% NOTE: The number of calls per sequence is calculated automatically to
% ensure that approximately 50% of every sequence's duration contains the
% call, and 50% does not.

%% Chorus Injection Parameters

% Synthetic chorus (overlapping distant conspecifics forming a continuous
% background) can be mixed into a fraction of training sequences and is
% automatically labelled negative (the per-sample mask is not touched),
% teaching the network to distinguish discrete calls from chorus.

enableChorus                       = true;        % Master switch (DISABLED in this baseline variant).
chorus_probability                 = 0.35;        % Per-sequence inclusion probability.
num_calls_in_chorus                = 200;         % Calls overlap-summed into the chorus base.
chorus_calls_level_range           = [2, 2]; % Per-call amplitude jitter (dB).
chorus_call_overlap_range          = [0.85, 0.999];% Fractional temporal overlap of consecutive calls (perecent of total call duration)
chorus_sequence_level_range        = 10;          % Slow amp. modulation depth (dB, positive).
chorus_modulation_period_s         = 30;          % Approx fundamental period of the slow envelope (s).
chorus_to_calls_snr_offset_range   = [-15, -6];   % chorusSNR_vs_noise = max(sequenceSNRs) + offset (dB).

% Chorus loudness relative to the loudest discrete call. Both are SNRs
% vs the SAME reference (original noise BEFORE chorus is mixed in).
% Chorus is NOT included in the noise term used to scale the calls.
%   chorus_SNR_vs_noise = max(sequenceSNRs) + offset_dB
% where offset_dB is drawn uniformly from this range (dB, typically
% negative so chorus sits below the loudest call). Note: the effective
% SNR seen by the network is lower because the actual background is
% (noise + chorus);

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

%% Inference Pre-Processing Parameters

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

minSilenceDuration = 1; % Silence causes the detector to return garbage.
%                       There is a silence detector that returns sample
%                       indices of silent or near-silent regions of the
%                       file before preprocessing so that any detections
%                       from within these times can be ignored. This
%                       variable sets the largest duration of audio that
%                       may be 'silent' without being flagged as a silent
%                       region. Suggested value = 1 (seconds)

frameStandardization = 'true'; % Sets whether the frequency bins of the
%                               frames of features are re-standardized to
%                               to their local, frame-level statistics.
%                               This setting applies to inference only, and
%                               takes effect in "event-split" and "simple"
%                               feature framing modes, but not in 'none'.

%% Inference Post-Processing Parameters

postProcOptions.AT = 0.50;   % Activation Threshold. Sets the probability
%                           threshold for starting a vocalisation segment.
%                           Specify as a scalar in the range [0,1].

postProcOptions.DT = 0.49;  % Deactivation Threshold. Sets the probability
%                           threshold for ending a vocalisation segment.
%                           Specify as a scalar in the range [0,1].

postProcOptions.AEAVD = 0;  % Apply Energy Animal Vocalisation Detection
%                           Specifies whether to apply an energy-based
%                           vocalization activity detector to refine the
%                           regions detected by the neural network.

postProcOptions.MT = 0.1;  % Merge Threshold. Merges vocalization regions
%                           that are separated by MT seconds or less.
%                           Specify as a nonnegative scalar.

postProcOptions.LT_scaler = 0.5; % The length threshold is set based on
%                           the mean duration of the calls in the training
%                           set, multiplied by this number. Any detection
%                           peak shorter than the length threshold is
%                           excluded.

%% Ground Truth Comparison Parameters

detectionTolerance = 30;
