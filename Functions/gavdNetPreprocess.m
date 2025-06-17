function varargout = gavdNetPreprocess(x, fsIn, fsTarget, ...
    bandwidth, windowLen, hopLen, saturationRange, mask, pcenParams)
%GAVDNETPREPROCESS Preprocess audio for animal vocalization detection 
%   This function generates a mel spectrogram from the audio input, audioIn,
%   that can be fed to the pretrained network either during training or 
%   inference. Spectrogram output has 40 Mel Frequency bins and T time
%   bins. Dynamic range of the spectrogram is saturated to the range set by 
%   the 'saturationRange' argument. By default, when no PCEN parameters are 
%   supplied, each spectrogram frequency bin is standardized by subtracting 
%   its mean, and dividing by the standard deviation. If PCEN parameters 
%   are supplied, PCEN replaces this standardization procedure.If a signal 
%   presence mask is provided, the function also transforms this mask from 
%   the audio sample domain to spectrogram time-bin domain for training 
%   purposes.
%
%   Inputs:
%   x               = the signal to preprocess
%   fsIn            = the original sampling rate of 'x' (Hz)
%   fsTarget        = the target sample rate for resampling 'x' (Hz)
%   bandwidth       = the [min, max] frequency range of interest (Hz)
%   windowLen       = the length of the window function in the STFT (samples)
%   hopLen          = the length of the window hop in the STFT (samples)
%   saturationRange = the dynamic range to saturate the spectrogram to, in dB (maxPow - saturationRange)
%   mask            = (optional) binary mask vector in audio sample domain
%   pcenParams      = Structure containing fields "T, alpha, r, epsilon'.
%                       See spectPCEN() function for more info.
%
%   Outputs:
%   features      = The spectrogram, returned as a 40-by-T array, where T is the number of
%                   time bins, dependent on windowLen, hopLen and the length of x.
%   transformedMask = (optional) Mask transformed to spectrogram time-bin domain
%
% References:
%   This function is based on the MATLAB function "vadnetPreprocess" [1, 2]
%   That function is a port of code from the open source toolkit
%   "SpeechBrain" [3].
%
%   [1] The MathWorks Inc. (2022-2024). Audio Toolbox version: 24.2 (R2024b),
%   Natick, Massachusetts: The MathWorks Inc. https://www.mathworks.com
%
%   [2] The MathWorks Inc. (2022-2024). Deep Learning Toolbox version: 24.2 (R2024b),
%   Natick, Massachusetts: The MathWorks Inc. https://www.mathworks.com
%
%   [3] Ravanelli, Mirco, et al. SpeechBrain: A General-Purpose Speech Toolkit.
%   arXiv, 8 June 2021. arXiv.org, http://arxiv.org/abs/2106.04624
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

arguments
    x {validateattributes(x,{'single','double'},{'nonempty','vector','real','finite'},'gavdNetPreprocess','audioIn')}
    fsIn {validateattributes(fsIn,{'single','double'},{'nonempty','scalar','real','finite','positive'},'gavdNetPreprocess','fsIn')}
    fsTarget {validateattributes(fsTarget,{'single','double'},{'nonempty','scalar','real','finite','positive'},'gavdNetPreprocess','fsTarget')}
    bandwidth {validateattributes(bandwidth,{'single','double'},{'nonempty','vector','real','finite','positive'},'gavdNetPreprocess','bandwidth')}
    windowLen {validateattributes(windowLen,{'single','double'},{'nonempty','scalar','real','finite','positive'},'gavdNetPreprocess','windowLen')}
    hopLen {validateattributes(hopLen,{'single','double'},{'nonempty','scalar','real','finite','positive'},'gavdNetPreprocess','hopLen')}
    saturationRange {validateattributes(saturationRange,{'single','double'},{'nonempty','scalar','real','finite','positive'},'gavdNetPreprocess','saturationRange')}
    mask = [] % Optional mask vector
    pcenParams = [] % Optional PCEN parameter struct
end

% If PCEN parameters are passed in, set PCEN to on and unpack.
pcen = false;
if ~isempty(pcenParams)
    pcen = true;
    TIn = pcenParams.T;
    alphaIn = pcenParams.alpha;
    rIn = pcenParams.r;
    epsilonIn = pcenParams.epsilon;
end

% Validate mask dimensions if provided
if ~isempty(mask)
    if numel(mask) ~= numel(x)
        error('Mask must have the same dimensions as the audio input.');
    end
end

% Get Fs of input from GPU
fsIn = gather(fsIn);

% If fs of input differs from target Fs, resample audio
if fsIn~=fsTarget
    [p, q] = rat(fsTarget/fsIn, 1e-9);
    xx = cast(resample(double(x(:)), p, q), like=x);
else
    xx = x(:);
end

% Set the FFT length at 8 * the next power of 2 larger than window length
FFTLen = 4 * 2^(ceil(log2(windowLen)));

% Error if signal is less than one window + one hop.
% With padding, this means the minimum number of frames output is 4.
assert(numel(xx) >= (windowLen+hopLen),'Signal is Too Short. Must be >= windowLen + hopLen')

% Normalize audio to either [-2147483648, 2147483647] for PCEN mode
% or [-1, 1] for normal mode:
switch pcen
    case true
        xx = scale32bit(xx);
    case false
        xx = xx ./ max(abs(xx));
end

% Compute the spectrogram
spect = computeSpectrogram(xx, windowLen, hopLen, FFTLen);

% Process spectrgram with either conventional frequency bin standardization
% or Per-Channel energy normalization.
switch pcen
    case false
        % Process Spectrogram with Mel Filterbank, convert to dB, then
        % saturate (original, deactivated)
        spectMeldB = spectrogramToMeldB(spect, fsTarget, FFTLen, bandwidth, saturationRange);
        
        % Standardize to zero mean and unity standard deviation (original, deactivated)
        varargout{1} = standardizeMelSpect(spectMeldB);
    case true
        % Process Spectrogram with Mel Filterbank (PCEN Mode)
        spectMel = spectrogramToMel(spect, fsTarget, FFTLen, bandwidth);

        % Calculate delta parameter
        deltaCalc = calculatePCENDelta(bandwidth);

        % Compute PCEN - tuned parameters
        spectPCENized = spectPCEN(spectMel, hopLen, fsTarget, T=TIn, alpha=alphaIn, r=rIn, epsilon=epsilonIn, delta=deltaCalc);
    
        % Mean normalization
        varargout{1} = spectPCENized - mean(spectPCENized, 'all');
end


% Process mask if provided
if nargout == 2 && ~isempty(mask)
    varargout{2} = maskToSpectDomain(mask, fsIn, fsTarget, windowLen, hopLen);
end
end

%% Helper Functions

function z = computeSpectrogram(x, windowLen, hopLen, FFTLen)
%centeredPowerSTFT Centered power spectrogram

N = FFTLen/2+1; % Number of bins in half-sided spectrum
padLen = ceil(windowLen/2);

% Design Window
hammingwindow = hamming(windowLen, "periodic");

% Zero-pad for half a frame, so the first prediction window is centered at 0 s.
x = [zeros(padLen, 1, like=x); x(:); zeros(padLen, 1, like=x)];

% Buffer the signal into overlapping windows
overlapLen = windowLen-hopLen;
xb = buffer(x, windowLen, overlapLen, "nodelay");

% Compute Single-sided STFT & convert to power
Xtwosided = abs(fft(xb .* hammingwindow, FFTLen)).^2;

% Take positive-frequency side
if isa(x,'gpuArray')
    z = head(Xtwosided,N);
else
    z = Xtwosided(1:N,:);
end
end

function spectMeldB = spectrogramToMeldB(spect, fsTarget, FFTLen, bandwidth, saturationRange)
%spectrogramToMeldB Compute mel spectrogram in the style of vadnet

% Design filter bank
persistent filterBank
if isempty(filterBank)
    filterBank = designAuditoryFilterBank(fsTarget, ...
        FFTLength = FFTLen, ...
        Normalization = "none", ...
        OneSided = true, ...
        FrequencyRange = bandwidth, ...
        FilterBankDesignDomain = "warped", ...
        FrequencyScale = "mel", ...
        NumBands = 40);
end

% Apply filter bank
spectrogramMel = filterBank * spect;

% Convert to dB (add small number to avoid log of 0)
spectMeldB = 10*log10(max(spectrogramMel, 1e-10));

% % Saturate at the max dB minus 'saturationRange'.
spectMeldB = max(spectMeldB,max(spectMeldB(:)) - saturationRange);
end

function spectMel = spectrogramToMel(spect, fsTarget, FFTLen, bandwidth)
%spectrogramToMeldB Compute mel spectrogram in the style of vadnet

% Design filter bank
persistent filterBank
if isempty(filterBank)
    filterBank = designAuditoryFilterBank(fsTarget, ...
        FFTLength = FFTLen, ...
        Normalization = "none", ...
        OneSided = true, ...
        FrequencyRange = bandwidth, ...
        FilterBankDesignDomain = "warped", ...
        FrequencyScale = "mel", ...
        NumBands = 40);
end

% Apply filter bank
spectMel = filterBank * spect;
end

function y = standardizeMelSpect(x)
%standardizeMelSpect Standardize audio to zero mean and unity standard deviation

amean = mean(x,2);
astd = max(std(x,[],2), 1e-10);
y = (x-amean)./astd;
end

function transformedMask = maskToSpectDomain(mask, fsIn, fsTarget, windowLen, hopLen)
%   This function transforms a binary mask from the audio sample domain to
%   the spectrogram time-bin domain, matching the processing steps used in
%   STFT computation.

% Resample mask if sample rate changes
if fsIn ~= fsTarget
    [p, q] = rat(fsTarget/fsIn, 1e-9);
    maskResamp = cast(resample(double(mask(:)), p, q), like=mask);
else
    maskResamp = mask(:);
end

% Zero pad to match the features (same padding as in STFT)
padLen = ceil(windowLen/2);
maskPadded = [zeros(padLen, 1, like=maskResamp); maskResamp; zeros(padLen, 1, like=maskResamp)];

% Buffer the mask to match the spectrogram time bins
overlapLen = windowLen - hopLen;
maskBuffered = buffer(maskPadded, windowLen, overlapLen, "nodelay");

% Take the mode of each frame to get the dominant mask value per time bin
transformedMask = mode(maskBuffered, 1);
end

function out = scale32bit(in)
% Define 32-bit signed integer range
int32_max = 2^31 - 1;

% Find maximum absolute value to preserve dynamic range
max_abs_value = max(abs(in(:)));

% Calculate scaling factor
if max_abs_value > 0
    scale_factor = int32_max / max_abs_value;
    out = in * scale_factor;
else
    % Handle edge case of all-zero signal
    out = in;
end
end
