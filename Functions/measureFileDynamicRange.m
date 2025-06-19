function fileStats = measureFileDynamicRange(audio, fs, fsTarget, bandwidth, windowLen, hopLen, plotResults)
%MEASUREFILEDYNAMICRANGE Analyzes dynamic range for a single audio signal.
%   This function analyzes dynamic range statistics for a given audio signal
%   to understand its characteristics. It computes a Mel spectrogram and
%   derives various metrics from it.
%
%   REQUIRED TOOLBOXES:
%   - Signal Processing Toolbox (for resample, buffer, hamming, fft)
%   - Audio Toolbox (for designAuditoryFilterBank)
%
%   Inputs:
%   audio           = input audio data (vector or matrix, first channel used)
%   fs              = sampling rate of the input audio (Hz)
%   fsTarget        = target sampling rate for analysis (Hz)
%   bandwidth       = [min, max] frequency range of interest (Hz)
%   windowLen       = STFT window length (samples)
%   hopLen          = STFT hop length (samples)
%   plotResults     = (optional) true/false to plot results (default: false)
%
%   Outputs:
%   fileStats    = struct with statistics for the input audio signal
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
arguments
    audio {validateattributes(audio,{'single','double'},{'nonempty','2d'},'measureFileDynamicRange','audio')}
    fs {validateattributes(fs,{'single','double'},{'nonempty','scalar','real','finite','positive'},'measureFileDynamicRange','fs')}
    fsTarget {validateattributes(fsTarget,{'single','double'},{'nonempty','scalar','real','finite','positive'},'measureFileDynamicRange','fsTarget')}
    bandwidth {validateattributes(bandwidth,{'single','double'},{'nonempty','vector','real','finite','positive','numel',2},'measureFileDynamicRange','bandwidth')}
    windowLen {validateattributes(windowLen,{'single','double'},{'nonempty','scalar','real','finite','positive','integer'},'measureFileDynamicRange','windowLen')}
    hopLen {validateattributes(hopLen,{'single','double'},{'nonempty','scalar','real','finite','positive','integer'},'measureFileDynamicRange','hopLen')}
    plotResults {validateattributes(plotResults, {'logical', 'numeric'}, {'scalar', 'binary'}, 'measureFileDynamicRange', 'plotResults')} = false
end

% Use first channel and convert to double for numerical stability
audioIn = double(audio(:,1));
fsIn = fs;

% Resample if necessary
if fsIn ~= fsTarget
    % High precision for rational approximation in resample
    [p, q] = rat(fsTarget/fsIn, 1e-9); 
    audioProcessed = resample(audioIn, p, q);
else
    audioProcessed = audioIn;
end

% Compute spectrogram
% Ensure FFTLen is a power of 2 and sufficiently large for the window
FFTLen = 8 * 2^(ceil(log2(windowLen))); 
spect = local_computeSTFT(audioProcessed, windowLen, hopLen, FFTLen);
spectMeldB = local_spectrogramToMeldB(spect, fsTarget, FFTLen, bandwidth);

% Initialize storage for statistics
fileStats = struct();

% Calculate file-level statistics
fileStats.durationAnalyzed = size(spectMeldB,2) * hopLen / fsTarget; % Duration in seconds
fileStats.globalMax = max(spectMeldB(:));
fileStats.globalMin = min(spectMeldB(:));
fileStats.globalDynamicRange = fileStats.globalMax - fileStats.globalMin;

% Percentiles
prctiles_levels = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9];
fileStats.percentiles.levels = prctiles_levels; 
fileStats.percentiles.values = prctile(spectMeldB(:), prctiles_levels);
% Robust dynamic range: 99.9th percentile - 0.1th percentile
fileStats.robustDynamicRange = fileStats.percentiles.values(end) - fileStats.percentiles.values(1); 

% Temporal dynamic range (dynamic range within each time frame)
temporalDR = max(spectMeldB, [], 1) - min(spectMeldB, [], 1);
fileStats.temporal.max = max(temporalDR);
fileStats.temporal.mean = mean(temporalDR);
fileStats.temporal.std = std(temporalDR);
fileStats.temporal.median = median(temporalDR);

% Frequency dynamic range (dynamic range within each frequency band)
frequencyDR = max(spectMeldB, [], 2) - min(spectMeldB, [], 2);
fileStats.frequency.max = max(frequencyDR);
fileStats.frequency.mean = mean(frequencyDR);
fileStats.frequency.std = std(frequencyDR);
fileStats.frequency.median = median(frequencyDR);

% Amplitude statistics (overall distribution of Mel spectrogram values)
fileStats.amplitude.mean = mean(spectMeldB(:));
fileStats.amplitude.std = std(spectMeldB(:));
fileStats.amplitude.median = median(spectMeldB(:));
% Median Absolute Deviation (MAD), scaled to be comparable to std for normal dist.
fileStats.amplitude.mad = mad(spectMeldB(:), 1); % MAD calculated w.r.t median (flag=1)
fileStats.amplitude.robustStd = fileStats.amplitude.mad * 1.4826;


% Outlier analysis (Robust method based on MAD)
outlierThresholdFactor = 5; % Number of MADs from median to define an outlier
outlierThresholdValue = outlierThresholdFactor * fileStats.amplitude.mad;
fileStats.outliers.thresholdFactor = outlierThresholdFactor;
fileStats.outliers.thresholdValueAbs = outlierThresholdValue;
fileStats.outliers.robustHighCount = sum(spectMeldB(:) > (fileStats.amplitude.median + outlierThresholdValue));
fileStats.outliers.robustLowCount  = sum(spectMeldB(:) < (fileStats.amplitude.median - outlierThresholdValue));
fileStats.outliers.percentRobust = (fileStats.outliers.robustHighCount + fileStats.outliers.robustLowCount) / numel(spectMeldB) * 100;

% Plot results if requested
if plotResults
    local_plotFileAnalysis(fileStats, spectMeldB, fsTarget, hopLen);
end

end

% Local Helper Functions 

function spect = local_computeSTFT(x, windowLen, hopLen, FFTLen)
%local_computeSTFT Compute STFT (Short-Time Fourier Transform) power spectrogram
% This implementation matches the one in measureDatasetDynamicRange.

% Ensure x is a column vector
x_col = x(:); 

% Window function
win = hamming(windowLen, "periodic");

% Padding strategy from original measureDatasetDynamicRange function
% This pads the signal at both ends by half a window length.
padLen = ceil(windowLen/2);
x_padded = [zeros(padLen, 1, 'like', x_col); x_col; zeros(padLen, 1, 'like', x_col)];

overlapLen = windowLen - hopLen;
if overlapLen < 0
    error('local_computeSTFT:InvalidHopLength', 'Hop length cannot be greater than window length.');
end
if windowLen > length(x_padded)
     warning('local_computeSTFT:ShortSignal', 'Window length is greater than padded signal length. Spectrogram might be empty or incorrect.');
    spect = zeros(FFTLen/2+1, 0); 
    return;
end

% Buffer the padded signal. buffer creates columns for each frame.
xb = buffer(x_padded, windowLen, overlapLen, 'nodelay');

% Apply window (element-wise to each column) and compute FFT along columns (dim 1)
Xtwosided = abs(fft(xb .* win, FFTLen, 1)).^2; 
spect = Xtwosided(1:(FFTLen/2 + 1), :); % Take the one-sided power spectrum
end


function spectMeldB = local_spectrogramToMeldB(spect, fsTarget, FFTLen, bandwidth)
%local_spectrogramToMeldB Convert power spectrogram to Mel scale in dB.
% Matches the conversion in measureDatasetDynamicRange.

numMelBands = 40;
filterBank = designAuditoryFilterBank(fsTarget, ...
    'FFTLength', FFTLen, ...
    'Normalization', 'none', ...
    'OneSided', true, ...
    'FrequencyRange', bandwidth, ...
    'FilterBankDesignDomain', 'warped', ...
    'FrequencyScale', 'mel', ...
    'NumBands', numMelBands);

spectrogramMel = filterBank * spect; % Apply Mel filter bank
% Convert to dB, adding a small constant to prevent log(0)
spectMeldB = 10*log10(max(spectrogramMel, 1e-10)); 
end



function local_plotFileAnalysis(fileStats, spectMeldB, fsTarget, hopLen)
%local_plotFileAnalysis Create visualization for single file dynamic range analysis.

figTitle = ['Dynamic Range Analysis: ' strrep(fileStats.source, '_', ' ')];
figure('Name', figTitle, 'NumberTitle', 'off', 'Position', [100 100 900 700]);
sgtitle(figTitle, 'FontSize', 14, 'FontWeight', 'bold');

% Subplot 1: Mel Spectrogram
subplot(2,2,1);
time_axis = (0:size(spectMeldB,2)-1) * hopLen / fsTarget;
numMelBands = size(spectMeldB,1);
imagesc(time_axis, 1:numMelBands, spectMeldB);
axis xy;
colorbarHandle = colorbar;
ylabel(colorbarHandle, 'Amplitude (dB)');
title('Mel Spectrogram');
xlabel('Time (s)');
ylabel('Mel Bands');
climVals = [min(spectMeldB(:)), max(spectMeldB(:))];
if all(isfinite(climVals)) && diff(climVals) > 0
    caxis(climVals);
else
    caxis auto; % Fallback if climVals are problematic
end

% Subplot 2: Histogram of Spectrogram Values
subplot(2,2,2);
histogram(spectMeldB(:), 50, 'Normalization', 'pdf', 'FaceColor', [0.3 0.7 0.9]);
title('Distribution of Mel Spectrogram Values');
xlabel('Amplitude (dB)');
ylabel('Probability Density');
grid on;
hold on;
current_ylim = ylim;
plot([fileStats.amplitude.median, fileStats.amplitude.median], current_ylim, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Median');
plot([fileStats.amplitude.mean, fileStats.amplitude.mean], current_ylim, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Mean');
% Indicate robust dynamic range bounds (0.1th and 99.9th percentiles)
p01 = fileStats.percentiles.values(1); 
p999 = fileStats.percentiles.values(end); 
plot([p01, p01], current_ylim, 'k:', 'LineWidth', 1.2, 'DisplayName', '0.1th Pctl');
plot([p999, p999], current_ylim, 'k:', 'LineWidth', 1.2, 'DisplayName', '99.9th Pctl');
legend('show', 'Location', 'northeast');
hold off;

% Subplot 3: Amplitude Percentile Plot
subplot(2,2,3);
semilogx(fileStats.percentiles.levels, fileStats.percentiles.values, 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
title('Amplitude Percentiles');
xlabel('Percentile Level (%)');
ylabel('Amplitude (dB)');
grid on;
xticks([0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]);
if exist('xtickformat','builtin') % For newer MATLAB versions
    xtickformat('%.1f');
end
xlim([min(fileStats.percentiles.levels), max(fileStats.percentiles.levels)]);

% Subplot 4: Summary Statistics Text
subplot(2,2,4);
axis off;
summaryText = {
    sprintf('Duration: %.2f s', fileStats.durationAnalyzed), ...
    '', ...
    sprintf('Global Max: %.1f dB', fileStats.globalMax), ...
    sprintf('Global Min: %.1f dB', fileStats.globalMin), ...
    sprintf('Global DR: %.1f dB', fileStats.globalDynamicRange), ...
    sprintf('Robust DR (0.1-99.9th pctl): %.1f dB', fileStats.robustDynamicRange), ...
    '', ...
    sprintf('Mean Amplitude: %.1f dB', fileStats.amplitude.mean), ...
    sprintf('Median Amplitude: %.1f dB', fileStats.amplitude.median), ...
    sprintf('Std Dev: %.1f dB', fileStats.amplitude.std), ...
    sprintf('MAD: %.1f dB (Robust Std: %.1f dB)', fileStats.amplitude.mad, fileStats.amplitude.robustStd), ...
    sprintf('Robust Outliers (%% > %d MAD): %.2f %%', fileStats.outliers.thresholdFactor, fileStats.outliers.percentRobust)
};
text(0.05, 0.95, summaryText, 'FontSize', 9, 'VerticalAlignment', 'top', 'FontName', 'Consolas');
title('Key Statistics Summary', 'FontSize', 11);

drawnow;
end
