%% Analyse Augmentation Independence from Original Exemplar
%
% This script evaluates the independence of augmented training signals from
% their source exemplars using the Structural Similarity Index (SSIM) of
% mel spectrograms.
%
% Four similarity analyses are performed:
%   1. Exemplar vs Exemplar: pairwise SSIM between all denoised exemplar
%      recordings. Measures natural inter-call similarity.
%   2. Augmented vs Augmented: pairwise SSIM between randomly sampled
%      augmented training signals. Measures training data diversity.
%   3. Augmented vs Casey-2014: SSIM of augmented signals compared to the
%      casey-2014 exemplar (the exemplar sourced from the test set).
%   4. Augmented vs All Exemplars: SSIM of augmented signals compared to
%      every exemplar. Measures overall exemplar-training similarity.
%
% Cohen's d effect sizes are reported for every pairwise combination of
% the four analyses.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

clear
close all
clc

%% Paths

exemplarDir = "E:\SORP_BmAntZ_exemplars\Denoised";
caseyExemplarName = "Bm_Ant_Z__casey-2014_2014_4_21  1_20_27.695_rank1_19.3dB_RXDENOISE.wav";
augmentedSignalsDir = "E:\GAVDNet\BmAntZ_SORP\Training & Models\-10 to 10\clean_signals";

nSamples = 26600;  % Number of augmented signals to sample

%% Mel-spectrogram parameters (matching GAVDNet config: GAVDNet_config_SORP_BmAntZ.m)

fsTarget        = 250;       % Target sample rate for feature extraction (Hz)
nMelBands       = 40;
bandwidth       = [10, 50];  % Hz
windowDur       = 0.85;      % s
hopDur          = 0.05;      % s
saturationRange = 70;        % dB
windowLen = 2 * round((windowDur * fsTarget) / 2);  % 212 samples at 250 Hz
hopLen    = 2 * round((hopDur * fsTarget) / 2);      % 12 samples at 250 Hz
FFTLen    = 4 * 2^(ceil(log2(windowLen)));            % 1024

%% Load all exemplars and compute mel spectrograms

exemplarFiles = dir(fullfile(exemplarDir, '*.wav'));
nExemplars = length(exemplarFiles);
fprintf('Found %d exemplars in %s\n', nExemplars, exemplarDir)

exemplarMelSpecs = cell(nExemplars, 1);
caseyIdx = 0;

for e = 1:nExemplars
    [eAudio, eFs] = audioread(fullfile(exemplarFiles(e).folder, exemplarFiles(e).name));
    eAudio = eAudio(:, 1);
    fprintf('  %s (Fs = %d Hz)\n', exemplarFiles(e).name, eFs)

    % Identify casey-2014 exemplar
    if exemplarFiles(e).name == caseyExemplarName
        caseyIdx = e;
    end

    % Resample to fsTarget and compute mel spectrogram
    if eFs ~= fsTarget
        eResampled = resample(double(eAudio), fsTarget, eFs);
    else
        eResampled = double(eAudio);
    end
    exemplarMelSpecs{e} = computeMelSpec(eResampled, fsTarget, nMelBands, bandwidth, ...
        windowLen, hopLen, FFTLen, saturationRange);
end

assert(caseyIdx > 0, 'Casey-2014 exemplar not found in exemplar directory.')
fprintf('Casey-2014 exemplar identified: index %d\n', caseyIdx)

%% Analysis 1: Exemplar vs Exemplar

nExemplarPairs = nchoosek(nExemplars, 2);
ssim_exVsEx = zeros(nExemplarPairs, 1);

fprintf('\nAnalysis 1: Exemplar vs Exemplar (%d pairs)...\n', nExemplarPairs)
pairIdx = 0;
for i = 1:nExemplars
    for j = (i+1):nExemplars
        pairIdx = pairIdx + 1;
        ssim_exVsEx(pairIdx) = spectrogramSSIM(exemplarMelSpecs{i}, exemplarMelSpecs{j});
    end
end

%% Prepare augmented signal sampling

augFiles = dir(fullfile(augmentedSignalsDir, '*.wav'));
fprintf('\nTotal augmented clean signals available: %d\n', length(augFiles))

[~, augFs] = audioread(fullfile(augFiles(1).folder, augFiles(1).name));
fprintf('Augmented signals sample rate: %d Hz\n', augFs)

rng(42, 'twister');  % Reproducible
reportInterval = max(1, round(nSamples / 20));

%% Analysis 2: Augmented vs Augmented

nAugPairs = nSamples;
ssim_augVsAug = zeros(nAugPairs, 1);

fprintf('\nAnalysis 2: Augmented vs Augmented (%d pairs)...\n', nAugPairs)

pairIdxA = randperm(length(augFiles), nAugPairs);
pairIdxB = randperm(length(augFiles), nAugPairs);
% Ensure no self-comparisons
while any(pairIdxA == pairIdxB)
    collision = pairIdxA == pairIdxB;
    pairIdxB(collision) = randperm(length(augFiles), sum(collision));
end

for i = 1:nAugPairs
    [audioA, ~] = audioread(fullfile(augFiles(pairIdxA(i)).folder, augFiles(pairIdxA(i)).name));
    [audioB, ~] = audioread(fullfile(augFiles(pairIdxB(i)).folder, augFiles(pairIdxB(i)).name));
    audioA = audioA(:, 1);
    audioB = audioB(:, 1);

    if augFs ~= fsTarget
        resampA = resample(double(audioA), fsTarget, augFs);
        resampB = resample(double(audioB), fsTarget, augFs);
    else
        resampA = double(audioA);
        resampB = double(audioB);
    end
    melA = computeMelSpec(resampA, fsTarget, nMelBands, bandwidth, ...
        min(windowLen, length(resampA)), hopLen, FFTLen, saturationRange);
    melB = computeMelSpec(resampB, fsTarget, nMelBands, bandwidth, ...
        min(windowLen, length(resampB)), hopLen, FFTLen, saturationRange);

    ssim_augVsAug(i) = spectrogramSSIM(melA, melB);

    if mod(i, reportInterval) == 0
        fprintf('  Processed %d/%d\n', i, nAugPairs)
    end
end

%% Analyses 3 & 4: Augmented vs Casey-2014 / Augmented vs All Exemplars

sampleIdx = randperm(length(augFiles), nSamples);
ssim_augVsCasey = zeros(nSamples, 1);
ssim_augVsAll   = zeros(nSamples * nExemplars, 1);

fprintf('\nAnalyses 3 & 4: Augmented vs Exemplars (%d signals x %d exemplars)...\n', ...
    nSamples, nExemplars)

for i = 1:nSamples
    filepath = fullfile(augFiles(sampleIdx(i)).folder, augFiles(sampleIdx(i)).name);
    [augAudio, ~] = audioread(filepath);
    augAudio = augAudio(:, 1);

    if augFs ~= fsTarget
        augResampled = resample(double(augAudio), fsTarget, augFs);
    else
        augResampled = double(augAudio);
    end
    augMelSpec = computeMelSpec(augResampled, fsTarget, nMelBands, bandwidth, ...
        min(windowLen, length(augResampled)), hopLen, FFTLen, saturationRange);

    % Analysis 3: vs Casey-2014
    ssim_augVsCasey(i) = spectrogramSSIM(exemplarMelSpecs{caseyIdx}, augMelSpec);

    % Analysis 4: vs all exemplars
    for e = 1:nExemplars
        ssim_augVsAll((i-1)*nExemplars + e) = spectrogramSSIM(exemplarMelSpecs{e}, augMelSpec);
    end

    if mod(i, reportInterval) == 0
        fprintf('  Processed %d/%d\n', i, nSamples)
    end
end

%% Print summary statistics

fprintf('\n%s\n', repmat('=', 1, 78))
fprintf('AUGMENTATION INDEPENDENCE ANALYSIS (Spectrogram SSIM)\n')
fprintf('%s\n\n', repmat('=', 1, 78))

analysisNames = {
    '1. Exemplar vs Exemplar'
    '2. Augmented vs Augmented'
    '3. Augmented vs Casey-2014'
    '4. Augmented vs All Exemplars'
};
analysisDescs = {
    'Natural similarity between independent recordings of the same call type'
    'Diversity among augmented training signals'
    'Similarity of augmented training data to the test-set exemplar'
    'Similarity of augmented training data to all source exemplars'
};
analysisData = {ssim_exVsEx, ssim_augVsAug, ssim_augVsCasey, ssim_augVsAll};
analysisN = cellfun(@numel, analysisData);

fprintf('SSIM: 1.0 = identical spectrograms, 0.0 = no structural similarity\n\n')

fprintf('%-35s %6s %8s %8s %8s %8s %8s\n', ...
    'Analysis', 'N', 'Mean', 'Median', 'Std', 'Min', 'Max')
fprintf('%s\n', repmat('-', 1, 78))
for a = 1:length(analysisNames)
    vals = analysisData{a};
    fprintf('%-35s %6d %8.4f %8.4f %8.4f %8.4f %8.4f\n', ...
        analysisNames{a}, analysisN(a), mean(vals), median(vals), ...
        std(vals), min(vals), max(vals))
end

fprintf('\nAnalysis descriptions:\n')
for a = 1:length(analysisNames)
    fprintf('  %s: %s\n', analysisNames{a}, analysisDescs{a})
end

%% Effect sizes (Cohen's d) for all pairwise combinations

fprintf('\n\n%s\n', repmat('=', 1, 78))
fprintf('EFFECT SIZES (Cohen''s d)\n')
fprintf('%s\n', repmat('=', 1, 78))
fprintf('Cohen''s d: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large\n')
fprintf('Positive d = first analysis has higher SSIM (more similar)\n')
fprintf('Negative d = second analysis has higher SSIM (more similar)\n')

nAnalyses = length(analysisNames);
pairCombs = nchoosek(1:nAnalyses, 2);

% What each pairwise comparison indicates
compDescs = { ...
    '1-2', 'Whether real calls are more similar to each other than augmented signals are to each other'; ...
    '1-3', 'Whether natural inter-call similarity exceeds augmented-to-casey similarity'; ...
    '1-4', 'Whether natural inter-call similarity exceeds augmented-to-exemplar similarity'; ...
    '2-3', 'Whether augmented signals are more similar to casey-2014 than to each other'; ...
    '2-4', 'Whether augmented signals are more similar to source exemplars than to each other'; ...
    '3-4', 'Whether casey-2014 is more recoverable from augmented data than other exemplars' ...
};
compMap = containers.Map(compDescs(:,1), compDescs(:,2));

fprintf('\n')
for p = 1:size(pairCombs, 1)
    a = pairCombs(p, 1);
    b = pairCombs(p, 2);
    valsA = analysisData{a};
    valsB = analysisData{b};
    pooledStd = sqrt((var(valsA) + var(valsB)) / 2);
    d = (mean(valsA) - mean(valsB)) / pooledStd;

    key = sprintf('%d-%d', a, b);
    fprintf('%s  vs  %s\n', analysisNames{a}, analysisNames{b})
    fprintf('  d = %+.4f  ', d)
    if abs(d) < 0.2
        fprintf('(negligible)\n')
    elseif abs(d) < 0.5
        fprintf('(small)\n')
    elseif abs(d) < 0.8
        fprintf('(medium)\n')
    else
        fprintf('(large)\n')
    end
    fprintf('  Tests: %s\n\n', compMap(key))
end


%% Save results

outputPath = fullfile(pwd, 'augmentation_independence_results.mat');
save(outputPath, 'ssim_exVsEx', 'ssim_augVsAug', 'ssim_augVsCasey', ...
    'ssim_augVsAll', 'nSamples', 'nExemplars', 'nExemplarPairs', ...
    'analysisNames', 'analysisDescs')
fprintf('\nResults saved to: %s\n', outputPath)

fprintf('\nDone.\n')

%% ========================= Helper Functions ============================

function similarity = spectrogramSSIM(melSpec1, melSpec2)
% Compute SSIM between two mel spectrograms.
% Handles different durations by resampling the time axis of the shorter
% spectrogram to match the longer one.
%
% Reference:
%   Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
%   Image quality assessment: from error visibility to structural
%   similarity. IEEE Transactions on Image Processing, 13(4), 600-612.

    [nBands1, T1] = size(melSpec1);
    [nBands2, T2] = size(melSpec2);

    % Resample shorter spectrogram to match longer along time axis
    Tmax = max(T1, T2);
    if T1 ~= Tmax
        melSpec1 = imresize(melSpec1, [nBands1, Tmax], 'bilinear');
    end
    if T2 ~= Tmax
        melSpec2 = imresize(melSpec2, [nBands2, Tmax], 'bilinear');
    end

    % Normalize both to [0, 1] for SSIM
    melSpec1 = (melSpec1 - min(melSpec1(:))) / (max(melSpec1(:)) - min(melSpec1(:)) + eps);
    melSpec2 = (melSpec2 - min(melSpec2(:))) / (max(melSpec2(:)) - min(melSpec2(:)) + eps);

    similarity = ssim(melSpec1, melSpec2);
end

function melSpec = computeMelSpec(audio, fs, nBands, bw, winLen, hopLen, FFTLen, saturationRange)
% Compute mel spectrogram matching gavdNetPreprocess pipeline.
% Uses Hamming window, power spectrum, mel filterbank, dB conversion,
% and dynamic range saturation.
    winLen = min(winLen, length(audio));
    hopLen = min(hopLen, winLen - 1);

    % STFT (Hamming window, matching gavdNetPreprocess)
    [S, F, ~] = spectrogram(audio, hamming(winLen, 'periodic'), ...
        winLen - hopLen, FFTLen, fs);
    S = abs(S).^2;  % Power spectrum

    % Restrict to bandwidth
    fIdx = F >= bw(1) & F <= bw(2);
    S = S(fIdx, :);
    F = F(fIdx);

    % Mel filterbank
    melFilter = melFilterBank(nBands, F, bw);

    % Apply filterbank
    spectMel = melFilter * S;

    % Convert to dB and saturate (matching gavdNetPreprocess)
    melSpec = 10 * log10(max(spectMel, 1e-10));
    melSpec = max(melSpec, max(melSpec(:)) - saturationRange);
end

function fb = melFilterBank(nBands, freqs, bw)
% Construct a simple triangular mel filterbank.
    melMin = hz2mel(bw(1));
    melMax = hz2mel(bw(2));
    melCenters = linspace(melMin, melMax, nBands + 2);
    hzCenters = mel2hz(melCenters);

    nFreqs = length(freqs);
    fb = zeros(nBands, nFreqs);
    for b = 1:nBands
        lo = hzCenters(b);
        mid = hzCenters(b + 1);
        hi = hzCenters(b + 2);
        for f = 1:nFreqs
            if freqs(f) >= lo && freqs(f) <= mid
                fb(b, f) = (freqs(f) - lo) / (mid - lo);
            elseif freqs(f) > mid && freqs(f) <= hi
                fb(b, f) = (hi - freqs(f)) / (hi - mid);
            end
        end
    end
end
