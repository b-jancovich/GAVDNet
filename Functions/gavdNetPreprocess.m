function [features, transformedMask] = gavdNetPreprocess(x, fsIn, fsTarget, bandwidth, windowLen, hopLen, mask)
%GAVDNETPREPROCESS Preprocess audio for general animal vocalization detection network
%   This function generates a mel spectrogram from the audio input, audioIn, 
%   that can be fed to the pretrained network. If a signal presence mask 
%   is provided, it also transforms the mask from audio sample domain to 
%   spectrogram time-bin domain for training purposes.
%
%   Inputs:
%   x           = the signal to preprocess
%   fsIn        = the original sampling rate of 'x' (Hz)
%   fsTarget    = the target sample rate for resampling 'x' (Hz)
%   bandwidth   = the [min, max] frequency range of interest (Hz)
%   windowLen   = the length of the window function in the STFT (samples)  
%   hopLen      = the length of the window hop in the STFT (samples)
%   mask        = (optional) binary mask in audio sample domain  
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
    mask = [] % Optional mask parameter
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
    xx = cast(resample(double(x(:)),fsTarget,double(fsIn)),like=x);
else
    xx = x(:);
end

% Calculate STFT params at target Fs
overlapLen = windowLen - hopLen;

% Set the FFT length at 2 * the next power of 2 larger than window length 
FFTLen = 8 * 2^(ceil(log2(windowLen)));

% Error if signal is less than one window + one hop.
% With padding, this means the minimum number of frames output is 4.
assert(numel(xx) >= (windowLen+hopLen),'Signal is Too Short. Must be >= windowLen + hopLen')

% Compute the spectrogram
spectrogram = icenteredPowerSTFT(xx, windowLen, hopLen, FFTLen);

% Process Spectrogram with Mel Filterbank
spectroramMeldB = imelSpectrogram(spectrogram, fsTarget, FFTLen, bandwidth);

% Standardize to zero mean and unity standard deviation
features = istandardize(spectroramMeldB);

% Process mask if provided
transformedMask = [];
if ~isempty(mask)
    % Resample mask if sample rate changes
    if fsIn~=fsTarget
        maskResamp = cast(resample(double(mask(:)),fsTarget,double(fsIn)),like=mask);
    else
        maskResamp = mask(:);
    end
    
    % Zero pad to match the features (same padding as in STFT)
    padLen = round(windowLen/2);
    maskPadded = [zeros(padLen, 1, like=maskResamp); maskResamp; zeros(padLen, 1, like=maskResamp)];
    
    % Buffer the mask to match the spectrogram time bins
    transformedMask = mode(buffer(maskPadded, windowLen, overlapLen, "nodelay"), 1);
end

end

function spectroramMeldB = imelSpectrogram(S, fsTarget, FFTLen, bandwidth)
%imelSpectrogram Compute mel spectrogram in the style of vadnet

% Design filter bank
persistent filterBank
if isempty(filterBank)
    filterBank = designAuditoryFilterBank(fsTarget, ...
        FFTLength=FFTLen, ...
        Normalization="none", ...
        OneSided=true, ...
        FrequencyRange=bandwidth, ...
        FilterBankDesignDomain="warped", ...
        FrequencyScale="mel", ...
        NumBands=40);
end

% Apply filter bank
spectrogramMel = filterBank*S;

% Convert to dB
spectroramMeldB = 10*log10(max(spectrogramMel,1e-10));

% Saturate at the max dB minus 80.
topdB = 80;
spectroramMeldB = max(spectroramMeldB,max(spectroramMeldB(:)) - topdB);

end

function z = icenteredPowerSTFT(x, windowLen, hopLen, FFTLen)
%centeredPowerSTFT Centered power spectrogram

N = FFTLen/2+1; % Number of bins in half-sided spectrum
padLen = round(windowLen/2);

% Design Window
persistent hammingwindow
if isempty(hammingwindow)
    hammingwindow = hamming(windowLen,"periodic");
end

% Zero-pad for half a frame, so the first prediction window is centered at 0 s.
x = [zeros(padLen,1,like=x); x(:); zeros(padLen,1,like=x)];

% Half-sided STFT
xb = audio.internal.buffer(x, windowLen, hopLen);
Xtwosided = abs(fft(xb .* hammingwindow, FFTLen)).^2;
if isa(x,'gpuArray')
    z = head(Xtwosided,N);
else
    z = Xtwosided(1:N,:);
end

end

function y = istandardize(x)
%istandardize Standardize audio to zero mean and unity standard deviation

amean = mean(x,2);
astd = max(std(x,[],2),1e-10);

y = (x-amean)./astd;
end