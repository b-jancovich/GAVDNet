function [datasetStats, fileStats] = measureDatasetDynamicRange(audioFolderPath, fileSampleSize, ...
    fsTarget, bandwidth, windowLen, hopLen, plotResults)
%MEASUREDATASETDYNAMICRANGE Comprehensive dynamic range analysis across hydrophone dataset
%   This function analyzes dynamic range statistics across multiple hydrophone
%   recordings to inform optimal saturation and standardization parameters for
%   training data preprocessing.
%
%   REQUIRED TOOLBOXES:
%   - Signal Processing Toolbox (for resample, buffer, hamming)
%   - Audio Toolbox (for designAuditoryFilterBank)
%
%   Inputs:
%   audioFolderPath = path to folder containing audio files
%   fileSampleSize  = number of files to randomly sample and analyze
%   fsTarget        = target sampling rate for analysis (Hz)  
%   bandwidth       = [min, max] frequency range of interest (Hz)
%   windowLen       = STFT window length (samples)
%   hopLen          = STFT hop length (samples)
%   plotResults     = (optional) true/false to plot results (default: true)
%
%   Outputs:
%   datasetStats = struct with aggregate dataset statistics and recommendations
%   fileStats    = struct array with individual file statistics
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
% Code reviewed and revised based on external feedback (2024).

arguments
    audioFolderPath {validateattributes(audioFolderPath,{'char','string'},{'nonempty'},'measureDatasetDynamicRange','audioFolderPath')}
    fileSampleSize {validateattributes(fileSampleSize,{'single','double'},{'nonempty','scalar','real','finite','positive','integer'},'measureDatasetDynamicRange','fileSampleSize')}
    fsTarget {validateattributes(fsTarget,{'single','double'},{'nonempty','scalar','real','finite','positive'},'measureDatasetDynamicRange','fsTarget')}
    bandwidth {validateattributes(bandwidth,{'single','double'},{'nonempty','vector','real','finite','positive'},'measureDatasetDynamicRange','bandwidth')}
    windowLen {validateattributes(windowLen,{'single','double'},{'nonempty','scalar','real','finite','positive'},'measureDatasetDynamicRange','windowLen')}
    hopLen {validateattributes(hopLen,{'single','double'},{'nonempty','scalar','real','finite','positive'},'measureDatasetDynamicRange','hopLen')}
    plotResults = true
end

% Get list of audio files in folder (efficient method)
fprintf('Scanning audio folder: %s\n', audioFolderPath);
audioFiles = dir(fullfile(audioFolderPath, '*.wav'));

if isempty(audioFiles)
    error('No audio files found in the specified folder.');
end
fprintf('Found %d audio files in folder.\n', length(audioFiles));

% Sample files randomly if more files than requested sample size
if length(audioFiles) > fileSampleSize
    fprintf('Randomly sampling %d files from %d available.\n', fileSampleSize, length(audioFiles));
    randIdx = randperm(length(audioFiles), fileSampleSize);
    sampledFiles = audioFiles(randIdx);
else
    fprintf('Analyzing all %d available files.\n', length(audioFiles));
    sampledFiles = audioFiles;
    fileSampleSize = length(audioFiles);
end

% Initialize storage for statistics and data
fileStats = struct();
allSpectrogramData = cell(fileSampleSize, 1); % Store flattened data for aggregate stats
exampleSpectrogram = []; % Store one spectrogram for plotting

% Process each file
fprintf('\nProcessing files:\n');
for i = 1:fileSampleSize
    filePath = fullfile(sampledFiles(i).folder, sampledFiles(i).name);
    fprintf('  [%d/%d] %s... ', i, fileSampleSize, sampledFiles(i).name);
    
    try
        % Load audio and get its native sampling rate
        [audio, fsIn] = audioread(filePath);
    catch ME
        fprintf('✗ (Error loading file: %s)\n', ME.message);
        allSpectrogramData{i} = []; % Mark this entry as invalid
        continue;
    end
    
    % Use first channel and convert to double for numerical stability
    audio = double(audio(:,1));
    
    % Resample if necessary
    if fsIn ~= fsTarget
        [p, q] = rat(fsTarget/fsIn, 1e-9);
        audio = resample(audio, p, q);
    end
    
    % Compute spectrogram
    FFTLen = 8 * 2^(ceil(log2(windowLen)));
    spect = computeSTFT(audio, windowLen, hopLen, FFTLen);
    spectMeldB = spectrogramToMeldB(spect, fsTarget, FFTLen, bandwidth);
    
    % Store flattened data for aggregate analysis
    allSpectrogramData{i} = spectMeldB(:);
    
    % Store the first valid spectrogram for plotting
    if isempty(exampleSpectrogram)
        exampleSpectrogram = spectMeldB;
    end
    
    % Calculate file-level statistics
    fileStats(i).filename = sampledFiles(i).name;
    fileStats(i).duration = size(spectMeldB,2) * hopLen / fsTarget;
    fileStats(i).globalMax = max(spectMeldB(:));
    fileStats(i).globalMin = min(spectMeldB(:));
    fileStats(i).globalDynamicRange = fileStats(i).globalMax - fileStats(i).globalMin;
    
    % Percentiles
    prctiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9];
    fileStats(i).percentiles = prctile(spectMeldB(:), prctiles);
    fileStats(i).robustDynamicRange = fileStats(i).percentiles(end) - fileStats(i).percentiles(1);
    
    % Temporal and frequency statistics
    temporalDR = max(spectMeldB, [], 1) - min(spectMeldB, [], 1);
    frequencyDR = max(spectMeldB, [], 2) - min(spectMeldB, [], 2);
    
    fileStats(i).temporal.max = max(temporalDR);
    fileStats(i).temporal.mean = mean(temporalDR);
    fileStats(i).temporal.std = std(temporalDR);
    fileStats(i).temporal.median = median(temporalDR);
    
    fileStats(i).frequency.max = max(frequencyDR);
    fileStats(i).frequency.mean = mean(frequencyDR);
    fileStats(i).frequency.std = std(frequencyDR);
    fileStats(i).frequency.median = median(frequencyDR);
    
    % Statistics for standardization algorithm design
    fileStats(i).amplitude.mean = mean(spectMeldB(:));
    fileStats(i).amplitude.std = std(spectMeldB(:));
    fileStats(i).amplitude.median = median(spectMeldB(:));
    fileStats(i).amplitude.mad = mad(spectMeldB(:), 1);
    
    % Outlier analysis (Robust method based on MAD)
    outlierThreshold = 5 * fileStats(i).amplitude.mad;
    fileStats(i).outliers.robustHigh = sum(spectMeldB(:) > fileStats(i).amplitude.median + outlierThreshold);
    fileStats(i).outliers.robustLow  = sum(spectMeldB(:) < fileStats(i).amplitude.median - outlierThreshold);
    fileStats(i).outliers.percentRobust = (fileStats(i).outliers.robustHigh + fileStats(i).outliers.robustLow) / numel(spectMeldB) * 100;

    fprintf('✓\n');
end

% Remove any failed files
validIdx = ~cellfun(@isempty, allSpectrogramData);
if ~any(validIdx)
    error('Processing failed for all sampled files. Check file formats and parameters.');
end
fileStats = fileStats(validIdx);
allSpectrogramData = allSpectrogramData(validIdx);
actualFileCount = sum(validIdx);

fprintf('\nSuccessfully processed %d files.\n', actualFileCount);

% Compute dataset-level aggregate statistics
fprintf('Computing dataset-level statistics...\n');

% Extract top-level fields using direct indexing
globalDRs   = [fileStats.globalDynamicRange];
robustDRs   = [fileStats.robustDynamicRange];
globalMaxes = [fileStats.globalMax];
globalMins  = [fileStats.globalMin];

% Extract nested fields using arrayfun
amplitudeMeans   = arrayfun(@(s) s.amplitude.mean, fileStats);
amplitudeStds    = arrayfun(@(s) s.amplitude.std, fileStats);
amplitudeMedians = arrayfun(@(s) s.amplitude.median, fileStats);
amplitudeMads    = arrayfun(@(s) s.amplitude.mad, fileStats);

% STATISTICALLY CORRECT DATASET-WIDE PERCENTILE CALCULATION
fprintf('Concatenating data for dataset-wide analysis (this may take a moment)...\n');
fullDatasetVector = vertcat(allSpectrogramData{:});
clear allSpectrogramData; % Free up memory

fprintf('Calculating true dataset-wide percentiles...\n');
datasetStats.percentiles.levels = prctiles;
datasetStats.percentiles.values = prctile(fullDatasetVector, prctiles);

% Dataset statistics
datasetStats.numFilesAnalyzed = actualFileCount;
datasetStats.totalDuration = sum([fileStats.duration]);

% Global dynamic range statistics
datasetStats.globalDynamicRange.mean = mean(globalDRs);
datasetStats.globalDynamicRange.std = std(globalDRs);
datasetStats.globalDynamicRange.median = median(globalDRs);
datasetStats.globalDynamicRange.min = min(globalDRs);
datasetStats.globalDynamicRange.max = max(globalDRs);
datasetStats.globalDynamicRange.percentiles = prctile(globalDRs, [5, 25, 50, 75, 95]);

% Robust dynamic range statistics
datasetStats.robustDynamicRange.mean = mean(robustDRs);
datasetStats.robustDynamicRange.std = std(robustDRs);
datasetStats.robustDynamicRange.median = median(robustDRs);
datasetStats.robustDynamicRange.min = min(robustDRs);
datasetStats.robustDynamicRange.max = max(robustDRs);
datasetStats.robustDynamicRange.percentiles = prctile(robustDRs, [5, 25, 50, 75, 95]);

% Amplitude level statistics (for standardization design)
datasetStats.amplitudeLevels.globalMax = max(globalMaxes);
datasetStats.amplitudeLevels.globalMin = min(globalMins);
datasetStats.amplitudeLevels.meanAcrossFiles = mean(amplitudeMeans);
datasetStats.amplitudeLevels.medianAcrossFiles = median(amplitudeMedians); % median of medians
datasetStats.amplitudeLevels.stdConsistency = std(amplitudeStds);
datasetStats.amplitudeLevels.madConsistency = std(amplitudeMads);

% Variability analysis
datasetStats.variability.globalDRCoeff = std(globalDRs) / mean(globalDRs);
datasetStats.variability.robustDRCoeff = std(robustDRs) / mean(robustDRs);
datasetStats.variability.amplitudeLevelCoeff = std(amplitudeMeans) / abs(mean(amplitudeMeans));

% Advanced saturation recommendations
fprintf('Computing saturation recommendations...\n');
datasetStats.recommendations.saturation.conservative = ceil(mean(robustDRs) + 2*std(robustDRs));
p_vals = datasetStats.percentiles.values;
datasetStats.recommendations.saturation.adaptive95 = ceil(p_vals(end-2) - p_vals(3)); % 95th - 5th percentile
datasetStats.recommendations.saturation.adaptive99 = ceil(p_vals(end-1) - p_vals(2)); % 99th - 1st percentile
datasetStats.recommendations.saturation.dataDriven = ceil(prctile(globalDRs, 75)); % 75th percentile of observed ranges

% Standardization recommendations
fprintf('Computing standardization recommendations...\n');
meanStdRatio = mean(amplitudeStds ./ (amplitudeMads * 1.4826)); 
datasetStats.recommendations.standardization.useRobust = meanStdRatio > 1.5;

if datasetStats.recommendations.standardization.useRobust
    datasetStats.recommendations.standardization.method = "robust";
else
    datasetStats.recommendations.standardization.method = "standard";
end
datasetStats.recommendations.standardization.stdInflationFactor = meanStdRatio;

% Optimal processing strategy
if std(globalDRs) / mean(globalDRs) > 0.3
    datasetStats.recommendations.processingStrategy = "file-level-normalization";
else
    datasetStats.recommendations.processingStrategy = "global-normalization";
end

% Display results
displayDatasetResults(datasetStats);

% Plot results if requested
if plotResults
    plotDatasetAnalysis(datasetStats, fileStats, exampleSpectrogram);
end

end

function displayDatasetResults(datasetStats)
%displayDatasetResults Display comprehensive dataset analysis results

fprintf('\n');
fprintf('==========================================\n');
fprintf('DATASET DYNAMIC RANGE ANALYSIS RESULTS\n');
fprintf('==========================================\n');
fprintf('Files analyzed: %d\n', datasetStats.numFilesAnalyzed);
fprintf('Total duration: %.1f hours\n', datasetStats.totalDuration/3600);
fprintf('\n');

fprintf('GLOBAL DYNAMIC RANGE STATISTICS:\n');
fprintf('  Mean: %.1f ± %.1f dB\n', datasetStats.globalDynamicRange.mean, datasetStats.globalDynamicRange.std);
fprintf('  Median: %.1f dB\n', datasetStats.globalDynamicRange.median);
fprintf('  Range: %.1f - %.1f dB\n', datasetStats.globalDynamicRange.min, datasetStats.globalDynamicRange.max);
fprintf('  Variability (CoV): %.2f\n', datasetStats.variability.globalDRCoeff);
fprintf('\n');

fprintf('ROBUST DYNAMIC RANGE STATISTICS (0.1 to 99.9 percentile):\n');
fprintf('  Mean: %.1f ± %.1f dB\n', datasetStats.robustDynamicRange.mean, datasetStats.robustDynamicRange.std);
fprintf('  Median: %.1f dB\n', datasetStats.robustDynamicRange.median);
fprintf('  Range: %.1f - %.1f dB\n', datasetStats.robustDynamicRange.min, datasetStats.robustDynamicRange.max);
fprintf('  Variability (CoV): %.2f\n', datasetStats.variability.robustDRCoeff);
fprintf('\n');

fprintf('AMPLITUDE LEVEL CONSISTENCY:\n');
fprintf('  Mean amplitude across files: %.1f dB\n', datasetStats.amplitudeLevels.meanAcrossFiles);
fprintf('  Amplitude variability (CoV): %.2f\n', datasetStats.variability.amplitudeLevelCoeff);
fprintf('  Std inflation factor (vs. robust MAD): %.2f\n', datasetStats.recommendations.standardization.stdInflationFactor);
fprintf('\n');

fprintf('==========================================\n');
fprintf('OPTIMIZATION RECOMMENDATIONS\n');
fprintf('==========================================\n');

fprintf('SATURATION PARAMETERS (DYNAMIC RANGE CLIPPING):\n');
fprintf('  Conservative (robust mean + 2σ): %d dB\n', datasetStats.recommendations.saturation.conservative);
fprintf('  Data-driven (75th percentile of file DRs): %d dB\n', datasetStats.recommendations.saturation.dataDriven);
fprintf('  Adaptive 99%% coverage (1st-99th percentile): %d dB\n', datasetStats.recommendations.saturation.adaptive99);
fprintf('  Adaptive 90%% coverage (5th-95th percentile): %d dB\n', datasetStats.recommendations.saturation.adaptive95);
fprintf('\n');

fprintf('STANDARDIZATION METHOD:\n');
fprintf('  Recommended method: %s\n', datasetStats.recommendations.standardization.method);
if datasetStats.recommendations.standardization.useRobust
    fprintf('  Justification: Standard deviation inflated by %.1fx due to outliers\n', ...
        datasetStats.recommendations.standardization.stdInflationFactor);
    fprintf('  Use: (x - median) / (MAD * 1.4826)\n');
else
    fprintf('  Justification: Data well-behaved, standard methods appropriate\n');
    fprintf('  Use: (x - mean) / std\n');
end
fprintf('\n');

fprintf('PROCESSING STRATEGY:\n');
fprintf('  Recommended: %s\n', datasetStats.recommendations.processingStrategy);
if strcmp(datasetStats.recommendations.processingStrategy, "file-level-normalization")
    fprintf('  Justification: High variability between files (DR CoV = %.2f)\n', ...
        datasetStats.variability.globalDRCoeff);
    fprintf('  Apply normalization per file before model training\n');
else
    fprintf('  Justification: Consistent amplitude levels across files\n');
    fprintf('  Global normalization across entire dataset is likely appropriate\n');
end
fprintf('\n');

end

function spect = computeSTFT(x, windowLen, hopLen, FFTLen)
%computeSTFT Compute STFT

N = FFTLen/2+1;
padLen = ceil(windowLen/2);
hammingwindow = hamming(windowLen, "periodic");
x = [zeros(padLen, 1, 'like', x); x(:); zeros(padLen, 1, 'like', x)];
overlapLen = windowLen-hopLen;
xb = buffer(x, windowLen, overlapLen, "nodelay");
Xtwosided = abs(fft(xb .* hammingwindow, FFTLen)).^2;
spect = Xtwosided(1:N,:);
end

function spectMeldB = spectrogramToMeldB(spect, fsTarget, FFTLen, bandwidth)
%spectrogramToMeldB Convert to mel scale dB

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

spectrogramMel = filterBank * spect;
spectMeldB = 10*log10(max(spectrogramMel, 1e-10));
end

function plotDatasetAnalysis(datasetStats, fileStats, exampleSpectrogram)
%plotDatasetAnalysis Create comprehensive visualization of dataset analysis

figure();

% Extract data using efficient indexing
globalDRs = [fileStats.globalDynamicRange];
robustDRs = [fileStats.robustDynamicRange];
durations = [fileStats.duration] / 3600; % hours

% FIX: Extract nested fields using arrayfun for robustness 
amplitudeMeans  = arrayfun(@(s) s.amplitude.mean, fileStats);
temporalMeans   = arrayfun(@(s) s.temporal.mean, fileStats);
temporalMaxes   = arrayfun(@(s) s.temporal.max, fileStats);
stds            = arrayfun(@(s) s.amplitude.std, fileStats);
mads            = arrayfun(@(s) s.amplitude.mad, fileStats);
outlierPercents = arrayfun(@(s) s.outliers.percentRobust, fileStats);

% Subplot 1: Global dynamic range distribution
subplot(3,4,1);
histogram(globalDRs, 15, 'Normalization', 'probability', 'FaceColor', [0.2 0.4 0.8]);
title('Global Dynamic Range Distribution');
xlabel('Dynamic Range (dB)');
ylabel('Probability');
grid on;
xline(datasetStats.globalDynamicRange.mean, 'r--', 'LineWidth', 2, 'DisplayName', 'Mean');
legend('Location', 'best');

% Subplot 2: Robust dynamic range distribution
subplot(3,4,2);
histogram(robustDRs, 15, 'Normalization', 'probability', 'FaceColor', [0.8 0.4 0.2]);
title('Robust Dynamic Range Distribution');
xlabel('Dynamic Range (0.1-99.9 percentile, dB)');
ylabel('Probability');
grid on;
xline(datasetStats.robustDynamicRange.mean, 'r--', 'LineWidth', 2, 'DisplayName', 'Mean');
legend('Location', 'best');

% Subplot 3: Global vs Robust comparison
subplot(3,4,3);
scatter(globalDRs, robustDRs, 50, 'filled', 'MarkerFaceAlpha', 0.6)
xlabel('Global Dynamic Range (dB)');
ylabel('Robust Dynamic Range (dB)');
title('Global vs Robust Dynamic Range');
hold on;
lims = [min([globalDRs, robustDRs]), max([globalDRs, robustDRs])];
if all(isfinite(lims))
    plot(lims, lims, 'k--', 'LineWidth', 1); 
    % axis(lims); 
end
grid on; axis equal;

% Subplot 4: Amplitude level consistency
subplot(3,4,4);
histogram(amplitudeMeans, 15, 'Normalization', 'probability', 'FaceColor', [0.4 0.8 0.2]);
title('Distribution of File Mean Amplitudes');
xlabel('Mean Amplitude (dB)');
ylabel('Probability');
grid on;

% Subplot 5: Temporal dynamic range statistics
subplot(3,4,5);
scatter(temporalMeans, temporalMaxes, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Mean Temporal DR (dB)');
ylabel('Max Temporal DR (dB)');
title('Temporal Dynamic Range Patterns');
grid on;

% Subplot 6: Standardization analysis
subplot(3,4,6);
scatter(mads * 1.4826, stds, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Robust Std. (MAD × 1.4826)');
ylabel('Standard Deviation');
title('Standardization: Std vs. Robust Std');
hold on;
lims = [0, max([mads*1.4826, stds])];
if all(isfinite(lims))
    plot(lims, lims, 'k--', 'LineWidth', 1.5, 'DisplayName', 'y=x'); 
end
legend('Location', 'best'); grid on; axis equal;

% Subplot 7: Outlier analysis
subplot(3,4,7);
histogram(outlierPercents, 15, 'Normalization', 'probability', 'FaceColor', [0.8 0.2 0.8]);
title('Robust Outlier Percentage');
xlabel('Points > 5 MAD from Median (%)');
ylabel('Probability');
grid on;

% Subplot 8: Dataset percentiles
subplot(3,4,8);
semilogx(datasetStats.percentiles.levels, datasetStats.percentiles.values, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'b');
title('True Dataset Amplitude Percentiles');
xlabel('Percentile Level');
ylabel('Amplitude (dB)');
grid on;
xticks([0.1, 1, 10, 50, 90, 99, 99.9]);

% Subplot 9: Saturation recommendations visualization
subplot(3,4,9);
satRecs = [datasetStats.recommendations.saturation.conservative, ...
           datasetStats.recommendations.saturation.dataDriven, ...
           datasetStats.recommendations.saturation.adaptive99];
recNames = {'Conservative', 'Data-Driven', '99% Coverage'};
bar(categorical(recNames), satRecs, 'FaceColor', [0.6 0.6 0.6]);
title('Saturation (Dynamic Range) Recommendations');
ylabel('Saturation Value (dB)');
grid on;

% Subplot 10: Example spectrogram
subplot(3,4,10);
if ~isempty(exampleSpectrogram)
    imagesc(exampleSpectrogram);
    axis xy;
    colorbar;
    title('Example Mel Spectrogram');
    xlabel('Time Bins');
    ylabel('Mel Bands');
end

% Subplot 11: Dynamic range vs file duration
subplot(3,4,11);
scatter(durations, globalDRs, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('File Duration (hours)');
ylabel('Global Dynamic Range (dB)');
title('Duration vs Dynamic Range');
grid on;

% Subplot 12: Summary statistics table
subplot(3,4,12);
axis off;
summaryText = {
    sprintf('Files Analyzed: %d', datasetStats.numFilesAnalyzed), ...
    sprintf('Total Duration: %.1f hrs', datasetStats.totalDuration/3600), ...
    '', ...
    sprintf('Mean Global DR: %.1f dB', datasetStats.globalDynamicRange.mean), ...
    sprintf('Mean Robust DR: %.1f dB', datasetStats.robustDynamicRange.mean), ...
    '', ...
    sprintf('Recommended Saturation: %d dB', datasetStats.recommendations.saturation.dataDriven), ...
    sprintf('Recommended Method: %s', datasetStats.recommendations.standardization.method), ...
    sprintf('Processing Strategy: %s', datasetStats.recommendations.processingStrategy)
};
text(0, 0.5, summaryText, 'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
title('Summary & Recommendations', 'FontSize', 12, 'FontWeight', 'bold');

sgtitle('Hydrophone Dataset Dynamic Range Analysis', 'FontSize', 16, 'FontWeight', 'bold');

end