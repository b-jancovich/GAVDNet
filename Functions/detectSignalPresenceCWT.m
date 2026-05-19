function [signalPresenceMask, correlationStrength] = detectSignalPresenceCWT(...
    signal, signalOfInterest, bandOfInterest, targetCallType, ...
    corrThresh, bufferSec, fs)
%
% The function is designed to find instances of a target signal within a
% longer signal. It computes the CWT of both signals, then computes their
% 2D correlation matrix, and looks for peaks using adaptive thresholding. 
% Implementation uses a custom 2D correlation function for computational 
% efficiency.
%
%   Inputs:
%       signal          - Audio signal to analyze [Nx1 real]
%       signalOfInterest- Template signal to detect [Mx1 real]
%       bandOfInterest  - Frequency band of interest [fMin fMax] (Hz)
%       targetCallType  - String specifying 'tonal' or non-tonal target
%       corrThresh      - Base correlation threshold [scalar, 0:1]
%       bufferSec       - Time buffer around detections (s)
%       fs             - Sampling frequency (Hz)
%
%   Outputs:
%       signalPresenceMask - Binary detection mask [Nx1 logical]
%       correlationStrength - Magnitude of correlation to signal of interest
%                             per sample [Nx1, real, 0:1]
%
%   Algorithm Details:
%   1. Preprocessing:
%      - Signal normalization
%      - Elliptical bandpass filtering
%      - Mean Centering
%      - Secondary normalization
%
%   2. Wavelet Analysis:
%      - Computes CWT using Morse wavelets (48 voices/octave)
%      - Time-Bandwidth Product:
%        * 70 for tonal signals
%        * 50 for non-tonal signals
%      - Template processing:
%        * Half-Hann windowing
%        * Energy normalization
%        * Soft thresholding (25%)
%        * 2D windowing
%
%   3. Detection:
%      - FFT-based 2D correlation of CWT coefficients
%      - Peak detection with adaptive thresholding
%      - Uses correlation strength as a kind of confidence scoring
%
%   Performance Optimizations:
%   - Persistent caching of template CWT coefficients
%   - Persistent caching of filter coefficients
%   - Custom 2D cross correlation function (faster than MATLAB's)
%   - Matrix operations over loops
%
%   Dependencies:
%   - Signal Processing Toolbox
%   - Wavelet Toolbox
%
%   Notes:
%   - Requires pre-filtered, clean template signal
%   - Memory usage scales with signal length
%   - Performance depends on frequency band width
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%% Init:

% Create persistent variables
persistent targetCoeffs a b prevSignalOfInterest prevBandOfInterest prevFs

% Set debug mode
debug = false;

% Input validation
validateattributes(signal, {'numeric'}, {'vector'});
validateattributes(signalOfInterest, {'numeric'}, {'vector'});
validateattributes(corrThresh, {'numeric'}, {'scalar', '>=', 0, '<=', 1});
validateattributes(bandOfInterest, {'numeric'}, {'vector', 'increasing', 'positive', '>', 0,'2d', '<', fs})

% Set highest frequency resolution possible: 48 Voices/octave
VPO = 48;

% Set time bandwidth product based on call type.
if strcmp(targetCallType, 'tonal') == 1
    TBP = 70;
else
    TBP = 50;
end

% Check if we need to create/recreate targetCoeffs
if isempty(targetCoeffs) || isempty(prevSignalOfInterest) || ...
        ~isequal(fs, prevFs) || ...
        ~isequal(signalOfInterest, prevSignalOfInterest)

    % Build targetCoeffs
    targetCoeffs = createtargetCoeffs(signalOfInterest, bandOfInterest, ...
        VPO, TBP, fs);
    prevSignalOfInterest = signalOfInterest;
    prevFs = fs;
end

% Check if we need to create/recreate the filter coefficients
if isempty(b) || ~exist('b', 'var') || isempty(a) || ~exist('a', 'var') || ...
        ~isequal(fs, prevFs) || ...
        ~isequal(bandOfInterest, prevBandOfInterest)
    % Design elliptical Band Pass Filter
    Rp = 0.5; % Passband ripple in dB
    Rs = 60; % Stopband attenuation in dB
    n = 6; % Filter order
    Wp = bandOfInterest / (fs/2);
    [b, a] = ellip(n, Rp, Rs, Wp, "bandpass", "ctf");
    prevBandOfInterest = bandOfInterest;
    prevFs = fs;
end

%% Compute 2D Correlation Of Wavelet Transforms

% Normalize the signal
signal = signal ./ max(abs(signal));

% Band pass filter the signal
signalBandPassed = ctffilt(b, a, signal);

% Mean Centering
signalBandPassed = signalBandPassed - mean(signalBandPassed);

% Normalize the signal
signalBandPassed = signalBandPassed ./ max(abs(signalBandPassed));

% Compute CWT of signal (limited to band of interest)
[wt, ~] = cwt(signalBandPassed, 'morse', fs, ...
    'FrequencyLimits', bandOfInterest, VoicesPerOctave=VPO, ...
    TimeBandwidth=TBP);

% Normalize wavelets of the two versions of the signal
wt = wt ./ max(abs(wt(:)));

% Compute scalogram correlation with mother wavelet
corrMatrix = abs(xcorr2F(abs(wt), abs(targetCoeffs)));

% Initialize outputs to match signal length
signalPresenceMask = false(size(signalBandPassed));
correlationStrength = zeros(size(signalBandPassed));

% Get the mean correlation as a function of time
corrSignal = mean(corrMatrix);

% Trim the extra samples (produced by padding and convolution in xcorr2f)
corrSignal = corrSignal(1:length(signalBandPassed));

% Normalize the corrSignal
corrSignal = corrSignal ./ max(corrSignal);

%% Find & filter peaks in the correlation signal

% Find peaks in correlation that exceed threshold
[peaksCorr, locsCorr] = findpeaks(corrSignal);

% Calculate adaptive correlation threshold
corrThreshAdapt = calculateAdaptiveThreshold(peaksCorr, corrThresh);

% Filter In-Band peaks by correlation threshold
peaksAboveThresh = (peaksCorr >= corrThreshAdapt);
peaksFiltered = peaksCorr(peaksAboveThresh);
locsFiltered = locsCorr(peaksAboveThresh);

% Extract detection windows and assign confidence
minSigDur = length(signalOfInterest)/fs;  % Duration in seconds

for i = 1:length(peaksFiltered)
    % Convert peak location to samples
    peakLoc = locsFiltered(i);

    % Calculate window boundaries in samples
    startIdx = max(1, round(peakLoc - (minSigDur/2 + bufferSec)*fs));
    endIdx = min(length(signal), round(peakLoc + (minSigDur/2 + bufferSec)*fs));

    % Mark these samples as containing the signal
    signalPresenceMask(startIdx:endIdx) = true;

    % Assign confidence scores to these samples
    correlationStrength(startIdx:endIdx) = peaksFiltered(i);
end

% Debug visualization if requested
if debug == true
    % Create spectrogram
    window = round(fs * 1); % 1s window
    noverlap = round(window * 0.75); % 75% overlap
    nfft = 2^nextpow2(window * 2); % Pad FFT
    figure('Name', 'Signal Detection Debug', 'Position', [100 100 1200 600]);

    % Compute spectrogram
    [s, f, t] = spectrogram(signal, window, noverlap, nfft, fs, 'yaxis');

    % Calculate time vector that matches the signal length
    tFull = (0:length(signal)-1)/fs;

    yyaxis left
    imagesc(t, f, 10*log10(abs(s)));
    set(gca, "YDir", "normal")
    ylim([0 fs/4]); % Limit to quarter Nyquist for better visibility

    % Add patches for detected regions
    hold on;

    % Find contiguous regions in signalPresenceMask
    regions = bwconncomp(signalPresenceMask);
    for i = 1:regions.NumObjects
        % Get start and end indices for this region
        regionInds = regions.PixelIdxList{i};
        tStart = tFull(regionInds(1));
        tEnd = tFull(regionInds(end));

        % Create region-start lines
        xline(tStart, '--black')

        % Create semi-transparent patch
        patch([tStart tEnd tEnd tStart], ...
            [bandOfInterest(1) bandOfInterest(1) bandOfInterest(2) bandOfInterest(2)], ...
            'white', 'FaceAlpha', 0.2, 'EdgeColor', 'black', 'LineStyle', '--');

        % Add confidence score text
        meanConf = mean(correlationStrength(regionInds));
        text(tStart, bandOfInterest(2)*1.2, sprintf('%.2f', meanConf), ...
            'Color', 'red', 'FontSize', 8);
    end

    % Formatting for left axis
    cb = colorbar;
    colormap(parula);
    clim([-40, 20])
    cb.Label.String = 'Power (dB)';
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Spectrogram with Detected Regions');

    % Add correlation peaks using the same time base
    yyaxis right
    plot(tFull, corrSignal);
    hold on
    scatter(tFull(locsFiltered), peaksFiltered, "Marker", '+');
    yline(corrThreshAdapt, 'black--', 'Label', 'Correlation Peak Height Thresh');
    ylabel('Correlation');
    hold off

    % Ensure axes align
    xlim([t(1) t(end)]);
end

end

%% Helper functions

function coeffs = createtargetCoeffs(exemplar, bandOfInterest, ...
    VPO, TBP, fs)
% Ensure column vector
exemplar = exemplar(:);

% Create custom window using half-Hann window
N = length(exemplar);
windowLen = floor(fs/4);
window = hann(2*windowLen);
windowIn = window(1:windowLen);
flattopLen = N - 2*windowLen;
windowFull = [windowIn; ones(flattopLen, 1); flip(windowIn)];

% Apply the window
x = exemplar .* windowFull;

% Ensure zero mean
x = x - mean(x);

% Normalize energy
x = x ./ sqrt(sum(abs(x).^2));

% Compute wavelet transform
[coeffs, ~] = cwt(x, 'morse', fs, 'FrequencyLimits', bandOfInterest, ...
    VoicesPerOctave=VPO, TimeBandwidth=TBP);

% Normalize the wavelet coeffs
coeffs = coeffs ./ max(abs(coeffs));

% Make the window funtion 2D
windowFull2D = repmat(windowFull', size(coeffs, 1), 1);

% Window the wavelet coeffs
coeffs = coeffs .* windowFull2D;

% Soft threshold the wavelet coeffs at 25%
coeffs = wthresh(coeffs, "s", 0.25);
end
%%

function c = xcorr2F(a, b)
% Frequency Domain 2D Cross Correlation
if nargin == 1
    b = a;
end

% Matrix dimensions
adim = size(a);
bdim = size(b);

% Cross-correlation dimension
cdim = adim+bdim-1;
bpad = zeros(cdim);
apad = zeros(cdim);
apad(1:adim(1),1:adim(2)) = a;
bpad(1:bdim(1),1:bdim(2)) = b(end:-1:1,end:-1:1);
ffta = fft2(apad);
fftb = fft2(bpad);
c = real(ifft2(ffta.*fftb));
end
%%

function threshAdapt = calculateAdaptiveThreshold(data, userThresh)
% Calculates adaptive threshold by cleaning the data for outliers then
% calculates data range, and multiples by the user threshold factor.
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
% Calculate mean
meanVal = mean(data);

% Clean data by removing values < 1 std below the mean
cleanData = data(data > meanVal - std(data));

% Calculate the range of the clean data
minClean = min(cleanData);
maxClean = max(cleanData);

% Map userThresh to cleaned data range
threshAdapt = minClean + userThresh * (maxClean - minClean);
end
