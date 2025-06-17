function [dynamicRangeStats, spectMeldB] = measureDynamicRange(audioFile, ...
    fsIn, fsTarget, bandwidth, windowLen, hopLen, plotResults)
%MEASURESPECTROGRAMDYNAMICRANGE Measure dynamic range of audio recordings
%   This function computes the mel spectrogram dynamic range statistics for
%   audio recordings to inform saturation parameter tuning.
%
%   Inputs:
%   audioFile   = path to audio file OR audio vector
%   fsIn        = original sampling rate of audio (Hz)
%   fsTarget    = target sampling rate for analysis (Hz)  
%   bandwidth   = [min, max] frequency range of interest (Hz)
%   windowLen   = STFT window length (samples)
%   hopLen      = STFT hop length (samples)
%   plotResults = (optional) true/false to plot results (default: true)
%
%   Outputs:
%   dynamicRangeStats = struct with dynamic range statistics
%   spectMeldB        = the mel spectrogram in dB (for further analysis)
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

arguments
    audioFile 
    fsIn {validateattributes(fsIn,{'single','double'},{'nonempty','scalar','real','finite','positive'},'measureSpectrogramDynamicRange','fsIn')}
    fsTarget {validateattributes(fsTarget,{'single','double'},{'nonempty','scalar','real','finite','positive'},'measureSpectrogramDynamicRange','fsTarget')}
    bandwidth {validateattributes(bandwidth,{'single','double'},{'nonempty','vector','real','finite','positive'},'measureSpectrogramDynamicRange','bandwidth')}
    windowLen {validateattributes(windowLen,{'single','double'},{'nonempty','scalar','real','finite','positive'},'measureSpectrogramDynamicRange','windowLen')}
    hopLen {validateattributes(hopLen,{'single','double'},{'nonempty','scalar','real','finite','positive'},'measureSpectrogramDynamicRange','hopLen')}
    plotResults = true
end

% Load audio if file path provided
if ischar(audioFile) || isstring(audioFile)
    [audio, ~] = audioread(audioFile);
    audio = audio(:,1); % Take first channel if stereo
else
    audio = audioFile(:); % Assume it's already audio data
end

% Resample if necessary
if fsIn ~= fsTarget
    [p, q] = rat(fsTarget/fsIn, 1e-9);
    audio = cast(resample(double(audio), p, q), like=audio);
end

% Compute spectrogram using same approach as gavdNetPreprocess
FFTLen = 8 * 2^(ceil(log2(windowLen)));
spect = computeSTFT(audio, windowLen, hopLen, FFTLen);
spectMeldB = spectrogramToMeldB(spect, fsTarget, FFTLen, bandwidth);

% Calculate dynamic range statistics
globalMax = max(spectMeldB(:));
globalMin = min(spectMeldB(:));
globalDynamicRange = globalMax - globalMin;

% Percentile-based statistics (more robust to outliers)
prctiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9];
percentileValues = prctile(spectMeldB(:), prctiles);

% Temporal dynamic range (max range in any time frame)
temporalDynamicRanges = max(spectMeldB, [], 1) - min(spectMeldB, [], 1);
maxTemporalDynamicRange = max(temporalDynamicRanges);
meanTemporalDynamicRange = mean(temporalDynamicRanges);

% Frequency band dynamic range (max range in any frequency band)
frequencyDynamicRanges = max(spectMeldB, [], 2) - min(spectMeldB, [], 2);
maxFrequencyDynamicRange = max(frequencyDynamicRanges);
meanFrequencyDynamicRange = mean(frequencyDynamicRanges);

% Robust dynamic range (99.9th percentile - 0.1st percentile)
robustDynamicRange = percentileValues(end) - percentileValues(1);

% Pack results into struct
dynamicRangeStats.globalDynamicRange = globalDynamicRange;
dynamicRangeStats.robustDynamicRange = robustDynamicRange;
dynamicRangeStats.globalMax = globalMax;
dynamicRangeStats.globalMin = globalMin;
dynamicRangeStats.percentiles.values = percentileValues;
dynamicRangeStats.percentiles.levels = prctiles;
dynamicRangeStats.temporal.maxDynamicRange = maxTemporalDynamicRange;
dynamicRangeStats.temporal.meanDynamicRange = meanTemporalDynamicRange;
dynamicRangeStats.frequency.maxDynamicRange = maxFrequencyDynamicRange;
dynamicRangeStats.frequency.meanDynamicRange = meanFrequencyDynamicRange;
dynamicRangeStats.recordingDuration = size(spectMeldB,2) * hopLen / fsTarget;

% Display results
fprintf('\n=== HYDROPHONE RECORDING DYNAMIC RANGE ANALYSIS ===\n');
fprintf('Recording duration: %.1f seconds\n', dynamicRangeStats.recordingDuration);
fprintf('Global dynamic range: %.1f dB\n', globalDynamicRange);
fprintf('Robust dynamic range (99.9%% - 0.1%%): %.1f dB\n', robustDynamicRange);
fprintf('Max temporal frame dynamic range: %.1f dB\n', maxTemporalDynamicRange);
fprintf('Mean temporal frame dynamic range: %.1f dB\n', meanTemporalDynamicRange);
fprintf('\nKey percentiles:\n');
fprintf('  99.9th percentile: %.1f dB\n', percentileValues(end));
fprintf('  99th percentile: %.1f dB\n', percentileValues(end-1));
fprintf('  95th percentile: %.1f dB\n', percentileValues(end-2));
fprintf('  5th percentile: %.1f dB\n', percentileValues(3));
fprintf('  1st percentile: %.1f dB\n', percentileValues(2));
fprintf('  0.1st percentile: %.1f dB\n', percentileValues(1));

% Saturation recommendations
fprintf('\n=== SATURATION RECOMMENDATIONS ===\n');
fprintf('For 99%% coverage: %.0f dB saturation\n', percentileValues(end-1) - percentileValues(2));
fprintf('For 95%% coverage: %.0f dB saturation\n', percentileValues(end-2) - percentileValues(3));
fprintf('Conservative (robust): %.0f dB saturation\n', ceil(robustDynamicRange));

% Plot results if requested
if plotResults
    plotDynamicRangeAnalysis(spectMeldB, dynamicRangeStats, fsTarget, hopLen);
end

end

function spect = computeSTFT(x, windowLen, hopLen, FFTLen)
%computeSTFT Compute STFT - same as in gavdNetPreprocess

N = FFTLen/2+1;
padLen = ceil(windowLen/2);

% Design Window
hammingwindow = hamming(windowLen, "periodic");

% Zero-pad for half a frame
x = [zeros(padLen, 1, like=x); x(:); zeros(padLen, 1, like=x)];

% Buffer the signal into overlapping windows
overlapLen = windowLen-hopLen;
xb = buffer(x, windowLen, overlapLen, "nodelay");

% Compute Single-sided STFT & convert to power
Xtwosided = abs(fft(xb .* hammingwindow, FFTLen)).^2;

% Take positive-frequency side
if isa(x,'gpuArray')
    spect = head(Xtwosided,N);
else
    spect = Xtwosided(1:N,:);
end
end

function spectMeldB = spectrogramToMeldB(spect, fsTarget, FFTLen, bandwidth)
%spectrogramToMeldB Convert to mel scale dB - same as in gavdNetPreprocess

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

% Convert to dB (without saturation)
spectMeldB = 10*log10(max(spectrogramMel, 1e-10));

end

function plotDynamicRangeAnalysis(spectMeldB, stats, fsTarget, hopLen)
%plotDynamicRangeAnalysis Create visualization of dynamic range analysis

figure('Position', [100, 100, 1200, 800]);

% Subplot 1: Spectrogram
subplot(2,3,1);
timeAxis = (0:size(spectMeldB,2)-1) * hopLen / fsTarget;
imagesc(timeAxis, 1:40, spectMeldB);
axis xy;
colorbar;
title('Mel Spectrogram (dB)');
xlabel('Time (s)');
ylabel('Mel Band');

% Subplot 2: Amplitude histogram
subplot(2,3,2);
histogram(spectMeldB(:), 50, 'Normalization', 'probability');
title('Amplitude Distribution');
xlabel('Amplitude (dB)');
ylabel('Probability');
grid on;

% Subplot 3: Temporal dynamic range
subplot(2,3,3);
temporalDR = max(spectMeldB, [], 1) - min(spectMeldB, [], 1);
plot(timeAxis, temporalDR, 'LineWidth', 1);
title('Temporal Dynamic Range');
xlabel('Time (s)');
ylabel('Dynamic Range (dB)');
grid on;

% Subplot 4: Percentile plot
subplot(2,3,4);
semilogy(stats.percentiles.values, stats.percentiles.levels, 'o-', 'LineWidth', 2);
title('Cumulative Distribution');
xlabel('Amplitude (dB)');
ylabel('Percentile');
grid on;

% Subplot 5: Frequency band dynamic range
subplot(2,3,5);
frequencyDR = max(spectMeldB, [], 2) - min(spectMeldB, [], 2);
plot(frequencyDR, 1:40, 'LineWidth', 2);
title('Frequency Band Dynamic Range');
xlabel('Dynamic Range (dB)');
ylabel('Mel Band');
grid on;

% Subplot 6: Statistics summary
subplot(2,3,6);
axis off;
text(0.1, 0.9, sprintf('Global DR: %.1f dB', stats.globalDynamicRange), 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.8, sprintf('Robust DR: %.1f dB', stats.robustDynamicRange), 'FontSize', 12);
text(0.1, 0.7, sprintf('Max Temporal DR: %.1f dB', stats.temporal.maxDynamicRange), 'FontSize', 12);
text(0.1, 0.6, sprintf('Mean Temporal DR: %.1f dB', stats.temporal.meanDynamicRange), 'FontSize', 12);
text(0.1, 0.4, 'Recommended Saturation:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.3, sprintf('Conservative: %.0f dB', ceil(stats.robustDynamicRange)), 'FontSize', 11);
text(0.1, 0.2, sprintf('95%% coverage: %.0f dB', stats.percentiles.values(end-2) - stats.percentiles.values(3)), 'FontSize', 11);
text(0.1, 0.1, sprintf('99%% coverage: %.0f dB', stats.percentiles.values(end-1) - stats.percentiles.values(2)), 'FontSize', 11);
title('Summary Statistics');

sgtitle('Hydrophone Recording Dynamic Range Analysis', 'FontSize', 14, 'FontWeight', 'bold');

end