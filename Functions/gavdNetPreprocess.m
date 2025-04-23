function features = gavdNetPreprocess(x, fsIn, fsTarget, bandwidth, windowDur, hopDur)
%vadnetTunablePreprocess Preprocess audio for voice activity detection network
%   This function generates a mel spectrogram from the audio input, audioIn, 
%   that can be fed to the pretrained network returned by the vadnet function. 
%   
%   Inputs:
%   x           = the signal to preprocess
%   fsIn        = the original sampling rate of 'x' (Hz)
%   fsTarget    = the target sample rate for resampling 'x' (Hz)
%   bandwidth   = the [min, max] frequency range of interest (Hz)
%   windowDur   = the duration of the window function in the STFT (s)  
%   hopDur      = the duration of the window hop in the STFT (s)   
% 
%   Outputs:
%   features    = The spectrogram, returned as a 40-by-T array, where T is the number of
%               time bins, dependent on windowDur, hopDur and the length of x.
%
%

arguments
    x {validateattributes(x,{'single','double'},{'nonempty','vector','real','finite'},'vadnetLFPreprocess','audioIn')}
    fsIn {validateattributes(fsIn,{'single','double'},{'nonempty','scalar','real','finite','positive'},'vadnetLFPreprocess','fsIn')}
    fsTarget {validateattributes(fsTarget,{'single','double'},{'nonempty','scalar','real','finite','positive'},'vadnetLFPreprocess','fsTarget')}
    bandwidth {validateattributes(bandwidth,{'single','double'},{'nonempty','vector','real','finite','positive'},'vadnetLFPreprocess','bandwidth')}
    windowDur {validateattributes(windowDur,{'single','double'},{'nonempty','scalar','real','finite','positive'},'vadnetLFPreprocess','windowDur')}
    hopDur {validateattributes(hopDur,{'single','double'},{'nonempty','scalar','real','finite','positive'},'vadnetLFPreprocess','hopDur')}
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
windowLen = windowDur * fsTarget;
hopLen = hopDur * fsTarget;

% Set the FFT length at 2 * the next power of 2 larger than window length 
FFTLen = 4 * 2^(ceil(log2(windowLen)));

% Error if signal is less than one window + one hop.
% With padding, this means the minimum number of frames output is 4.
coder.internal.errorIf(numel(xx)<(windowLen+hopLen),'audio:vadnet:SignalTooShort')

% Compute the spectrogram
spec = icenteredPowerSTFT(xx, windowLen, hopLen, FFTLen);

% Process Spectrogram with Mel Filterbank
SmeldB = imelSpectrogram(spec, fsTarget, FFTLen, bandwidth);

% Standardize to zero mean and unity standard deviation
features = istandardize(SmeldB);

end

function SmeldB = imelSpectrogram(S, fsTarget, FFTLen, bandwidth)
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
Smel = filterBank*S;

% Convert to dB
SmeldB = 10*log10(max(Smel,1e-10));

% Saturate at the max dB minus 80.
topdB = 80;
SmeldB = max(SmeldB,max(SmeldB(:)) - topdB);

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
if isempty(coder.target) && isa(x,'gpuArray')
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