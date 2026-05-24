% Inspect "Unknown" false positives at a chosen post-processing operating point.
%
% Purpose:
%   The adjudicated GT sweep (GAVDNet_SweepPostProcParamsAndCompareToAdjudicatedGT.m)
%   inherits analyst decisions onto new-model false positives via
%   sample-domain IoU against the original adjudicated FP set. New FPs whose
%   maximum IoU on the same file falls below the threshold inherit
%   analystDecision = 'Unknown' and remain FP under every decision logic.
%   They therefore drag precision down even though some may in fact be
%   real discrete calls that the prior model never produced (and so were
%   never adjudicated).
%
%   This script re-runs post-processing at the sweep's chosen operating
%   point, identifies the Unknown FPs, and renders one linear-frequency
%   spectrogram per detection so an analyst can manually classify each as
%   one of the four standard analyst decisions:
%       'DiscreteCallsPresent'         -> real call (TP under any logic)
%       'DiscreteCallsChorusPresent'   -> real call inside chorus (TP under
%                                         Discrete-only / Inclusive)
%       'ChorusPresent'                -> chorus only (FP under Discrete*)
%       'CallChorusAbsent'             -> nothing real (FP under any logic)
%
%   Two operating modes:
%
%     mode = 'render'  re-runs postproc, identifies Unknown FPs, writes one
%                      PNG spectrogram per FP and a CSV manifest with an
%                      empty analyst_decision column for the analyst to
%                      fill.
%
%     mode = 'tally'   reads the filled CSV and recomputes precision /
%                      recall under the user's stated TP definition
%                      ("discrete call, with or without chorus"). The
%                      audit can only increase recall and precision since
%                      Unknown FPs are converted from FP into either TP or
%                      remain FP, never the other way.
%
%   The recall numerator gain is small in absolute terms (32 Unknowns out
%   of ~7000 GT positives at the Run #1 chosen point) but the audit also
%   sanity-checks what the new model is firing on, beyond the inheritance
%   coverage of the prior model.
%
% Inputs:
%   USER INPUT block below specifies the chosen sweep run folder plus the
%   inference / model / audio paths needed to re-run postproc.
%
% Outputs (render mode, written under <chosenSweepRunFolder>/unknown_FPs/):
%   spectrograms/FP_<idx>_<basename>.png   per-FP spectrograms with the
%                                          detection time bounds overlaid
%                                          as red vertical lines
%   unknown_FPs_manifest.csv               manifest with index, filename,
%                                          time bounds, sample bounds,
%                                          confidence, plus a blank
%                                          analyst_decision column
%   inspection_log.txt                     diary of the run
%
% Outputs (tally mode, written under the same folder):
%   audited_metrics.txt                    summary of category counts and
%                                          recomputed precision / recall /
%                                          F1 under Discrete-only TP def
%
% Ben Jancovich, 2026
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

clear;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USER INPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Mode ---
% 'render'  : re-run postproc, identify Unknown FPs, write PNGs + CSV.
% 'tally'   : read filled CSV, recompute precision / recall.
mode                         = 'render';

% --- Sweep run to audit ---
% Folder containing chosen_postProcOptions_<ts>.mat and
% sweep_summary_<ts>.mat (created by
% GAVDNet_SweepPostProcParamsAndCompareToAdjudicatedGT.m).
chosenSweepRunFolder         = "C:\Users\z5439673\Git\GAVDNet\PostProc Tuning\ExcludeChorus\ExcludeChorus_2026-05-19_08-48-42_discrete-only";

% --- Inference / model artefacts (only used in 'render' mode) ---
trainedModelPath             = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar Exclude Chorus\GAVDNet_trained_18-May-2026_15-13.mat";
inferenceOutputPath          = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\Chagos_DGS\Test Results\Final Test - 2007subset\Exclude Chorus";   % expects detector_raw_results.mat here
audioSourceFolder            = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\Chagos_DGS\Test Data\2007subset";
groundtruthPath              = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\Chagos_DGS\Test Data\2007subset\test_dataset_detection_list.mat";
gtFormat                     = 'CTBTO';
adjudicatedDisagreementsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\GAVDNet Data for Publication\Post-Adjudication Results\CPBW_DiegoGarciaSouth_2007_ADJUDICATED\detector_vs_GT_disagreements_07-Jul-2025_08-57-43.mat";

% --- GT matching tolerances (must match the sweep that produced the chosen point) ---
detectionTolerance           = 30;    % seconds, Hungarian GT-match tolerance
maxDetectionDuration         = 40;    % seconds, FN window length

% --- Spectrogram window ---
contextSeconds               = 60;    % seconds of audio on either side of the detection centre
spectrogramFreqRangeHz       = [0 60]; % frequency band to display (DGS calls sit ~22 Hz)
spectrogramWindowSec         = 2;     % STFT window length, seconds
spectrogramOverlapFrac       = 0.90;  % STFT window overlap fraction
spectrogramDynRangeDB        = 60;    % dB span below max for display normalisation

% --- Tally mode ---
% Manifest column name that the analyst fills with one of:
%   DiscreteCallsPresent | DiscreteCallsChorusPresent |
%   ChorusPresent        | CallChorusAbsent
% Rows left blank are treated as "still Unknown" and remain FP.
analystColumnName            = 'analyst_decision';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.

%% Setup paths

% This script lives in <projectRoot>/Utilitiy Scripts/ so projectRoot is two
% levels up from the script file. Top-level entry-point scripts use pwd
% instead, but those run from the repo root.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, "Functions"));

% Validate mode
validModes = {'render', 'tally'};
if ~ismember(mode, validModes)
    error('Unknown mode ''%s''. Valid: %s.', mode, strjoin(validModes, ', '));
end

% Output subfolder under the chosen sweep run
auditFolder        = fullfile(chosenSweepRunFolder, 'unknown_FPs');
spectrogramFolder  = fullfile(auditFolder, 'spectrograms');
manifestPath       = fullfile(auditFolder, 'unknown_FPs_manifest.csv');
metricsPath        = fullfile(auditFolder, 'audited_metrics.txt');
logPath            = fullfile(auditFolder, sprintf('inspection_log_%s.txt', ...
                              char(datetime("now", "Format", "uuuu-MM-dd_HH-mm-ss"))));

if ~isfolder(auditFolder)
    mkdir(auditFolder);
end

diary(logPath);
fprintf('=== inspect_unknown_FPs_at_chosen_operating_point ===\n');
fprintf('Mode               : %s\n', mode);
fprintf('Chosen sweep folder: %s\n', chosenSweepRunFolder);
fprintf('Started            : %s\n', char(datetime("now")));

%% Load the chosen operating point (both modes)

chosenFiles = dir(fullfile(chosenSweepRunFolder, 'chosen_postProcOptions_*.mat'));
if isempty(chosenFiles)
    error('No chosen_postProcOptions_*.mat in %s', chosenSweepRunFolder);
end
chosenData = load(fullfile(chosenFiles(1).folder, chosenFiles(1).name));
postProcOptions = chosenData.postProcOptions;
fprintf('Chosen postProcOptions:\n');
disp(postProcOptions);

summaryFiles = dir(fullfile(chosenSweepRunFolder, 'sweep_summary_*.mat'));
if isempty(summaryFiles)
    error('No sweep_summary_*.mat in %s', chosenSweepRunFolder);
end
S = load(fullfile(summaryFiles(1).folder, summaryFiles(1).name));
chosenRow              = S.chosenRow;
decisionLogic          = S.decisionLogic;
originalFPMatch_IoUThr = S.originalFPMatch_IoUThreshold;
fprintf('decisionLogic used at chosen point: %s\n', decisionLogic);
fprintf('originalFPMatch_IoUThreshold      : %g\n\n', originalFPMatch_IoUThr);

switch mode
    case 'render'
        renderUnknownFPs( ...
            postProcOptions, chosenRow, originalFPMatch_IoUThr, ...
            trainedModelPath, inferenceOutputPath, audioSourceFolder, ...
            groundtruthPath, gtFormat, adjudicatedDisagreementsPath, ...
            detectionTolerance, maxDetectionDuration, ...
            spectrogramFolder, manifestPath, ...
            contextSeconds, spectrogramFreqRangeHz, ...
            spectrogramWindowSec, spectrogramOverlapFrac, ...
            spectrogramDynRangeDB);

    case 'tally'
        tallyAuditedDecisions( ...
            manifestPath, metricsPath, chosenRow, analystColumnName);
end

fprintf('\nFinished: %s\n', char(datetime("now")));
diary off;


%% ========================================================================
%% Local functions
%% ========================================================================

function renderUnknownFPs( ...
        postProcOptions, chosenRow, originalFPMatch_IoUThr, ...
        trainedModelPath, inferenceOutputPath, audioSourceFolder, ...
        groundtruthPath, gtFormat, adjudicatedDisagreementsPath, ...
        detectionTolerance, maxDetectionDuration, ...
        spectrogramFolder, manifestPath, ...
        contextSeconds, freqRangeHz, ...
        winSec, overlapFrac, dynRangeDB)
% Re-run post-processing at the chosen operating point, identify Unknown
% FPs by re-doing the IoU inheritance, and render one spectrogram per FP.

    %% Pre-flight checks
    rawDetectionsPath = fullfile(inferenceOutputPath, 'detector_raw_results.mat');
    if ~exist(rawDetectionsPath, 'file')
        error('Raw detection results not found:\n  %s', rawDetectionsPath);
    end
    if ~exist(trainedModelPath, 'file')
        error('Trained model file not found:\n  %s', trainedModelPath);
    end
    if ~exist(groundtruthPath, 'file')
        error('Ground truth file not found:\n  %s', groundtruthPath);
    end
    if ~exist(adjudicatedDisagreementsPath, 'file')
        error('Adjudicated disagreements file not found:\n  %s', adjudicatedDisagreementsPath);
    end
    if ~isfolder(audioSourceFolder)
        error('Audio source folder not found:\n  %s', audioSourceFolder);
    end

    if ~isfolder(spectrogramFolder)
        mkdir(spectrogramFolder);
    end

    %% Load model + raw + disagreements
    fprintf('Loading trained model...\n');
    modelData = load(trainedModelPath, 'model');
    model = modelData.model;
    preprocParams = model.preprocParams;

    fprintf('Loading cached raw detection results...\n');
    rawData    = load(rawDetectionsPath, 'results');
    rawResults = rawData.results;
    fprintf('  %d files of cached probabilities.\n', length(rawResults));

    fprintf('Loading adjudicated disagreements...\n');
    adjData       = load(adjudicatedDisagreementsPath, 'disagreements');
    disagreements = adjData.disagreements;
    fprintf('  %d adjudicated FPs.\n', numel(disagreements.falsePositives));

    %% Audio cache (needed by gavdNetPostprocess even when AEAVD=0)
    filesFromRaw = arrayfun(@(r) basenameOf(r.fileName),      rawResults,                  'UniformOutput', false);
    filesFromFP  = arrayfun(@(d) basenameOf(d.AudioFilename), disagreements.falsePositives, 'UniformOutput', false);
    uniqueFiles  = unique([filesFromRaw(:); filesFromFP(:)]);
    uniqueFiles  = uniqueFiles(~cellfun(@isempty, uniqueFiles));
    fprintf('Caching %d audio files into RAM...\n', length(uniqueFiles));

    audioCache = struct('name', {}, 'audio', {}, 'fileFs', {});
    audioCache(length(uniqueFiles)).name = '';
    nReadFail = 0;
    for k = 1:length(uniqueFiles)
        fname = uniqueFiles{k};
        audioCache(k).name = fname;
        try
            [audio, fs] = audioread(fullfile(audioSourceFolder, fname));
            if size(audio, 2) > 1
                audio = audio(:, 1);
            end
            audioCache(k).audio  = audio;
            audioCache(k).fileFs = fs;
        catch ME
            warning('Audio read failed for %s: %s', fname, ME.message);
            audioCache(k).audio  = [];
            audioCache(k).fileFs = NaN;
            nReadFail = nReadFail + 1;
        end
    end
    fprintf('  Cached %d files (%d read failures).\n', length(uniqueFiles), nReadFail);

    % map basename -> audioCache index
    audioCacheNames = {audioCache.name};
    audioIdxForRaw  = zeros(length(rawResults), 1);
    for f = 1:length(rawResults)
        nm = basenameOf(rawResults(f).fileName);
        idx = find(strcmp(audioCacheNames, nm), 1);
        if isempty(idx)
            audioIdxForRaw(f) = 0;
        else
            audioIdxForRaw(f) = idx;
        end
    end

    %% Re-run postprocess at the chosen options
    fprintf('Running gavdNetPostprocess at the chosen operating point...\n');
    currentResults = rawResults;
    nFiles = length(currentResults);
    for f = 1:nFiles
        [b, conf, n, t, ~] = postprocOneFile( ...
            currentResults(f), audioIdxForRaw(f), audioCache, ...
            preprocParams, postProcOptions);
        currentResults(f).eventSampleBoundaries = b;
        currentResults(f).confidence            = conf;
        currentResults(f).nDetections           = n;
        currentResults(f).eventTimesDT          = t;
    end

    %% Flatten + compare to GT
    flatDetections = flattenDetections(currentResults, preprocParams);
    if isempty(fieldnames(flatDetections))
        flatDetections = struct([]);
    end
    fprintf('  %d detections at this operating point.\n', numel(flatDetections));

    % compareDetectionsToSubsampledTestDataset reads results from a .mat file,
    % so write the flat detections to a temporary results file.
    tempPath = fullfile(spectrogramFolder, 'temp_results_for_compare.mat');
    results = flatDetections;
    featureFraming       = 'unknown';
    frameStandardization = 'unknown';
    currentPostProcOptions = postProcOptions;
    save(tempPath, 'results', 'featureFraming', 'frameStandardization', ...
        'currentPostProcOptions', '-v6');

    fprintf('Running compareDetectionsToSubsampledTestDataset...\n');
    [~, newFP, ~] = compareDetectionsToSubsampledTestDataset( ...
        groundtruthPath, tempPath, detectionTolerance, ...
        maxDetectionDuration, gtFormat);
    delete(tempPath);
    fprintf('  %d new-model FPs at the chosen operating point.\n', numel(newFP));

    %% Inherit adjudication and filter to Unknowns
    fpGrouping = buildFPGrouping(disagreements.falsePositives);
    newFPwithAdj = inheritAdjudication(newFP, fpGrouping, originalFPMatch_IoUThr);

    decs = string({newFPwithAdj.analystDecision});
    isUnknown = decs == "Unknown";
    unknownFPs = newFPwithAdj(isUnknown);
    fprintf('  %d Unknown FPs after inheritance (expected %d from chosenRow).\n', ...
        numel(unknownFPs), chosenRow.nUnknownFPs);

    if numel(unknownFPs) == 0
        fprintf('No Unknown FPs to render. Nothing to do.\n');
        return;
    end

    %% Render spectrograms + build manifest
    fprintf('Rendering %d spectrograms to:\n  %s\n', ...
        numel(unknownFPs), spectrogramFolder);

    % Basename lookup into rawResults so each render call can pull the
    % full-file probability trace for the bottom-axis overlay.
    rawResultsNames = arrayfun(@(r) basenameOf(r.fileName), rawResults, ...
        'UniformOutput', false);

    nU = numel(unknownFPs);
    manifestRows = cell(nU, 1);

    for i = 1:nU
        fp = unknownFPs(i);
        [pngName, manRow] = renderOneSpectrogram( ...
            fp, i, audioSourceFolder, audioCache, audioCacheNames, ...
            rawResults, rawResultsNames, preprocParams, postProcOptions, ...
            spectrogramFolder, contextSeconds, freqRangeHz, ...
            winSec, overlapFrac, dynRangeDB);
        fprintf('  [%3d/%3d] %s\n', i, nU, pngName);
        manifestRows{i} = manRow;
    end

    manifest = vertcat(manifestRows{:});
    % Append blank analyst_decision column for the analyst to fill in.
    manifest.analyst_decision = repmat({''}, height(manifest), 1);
    manifest.notes            = repmat({''}, height(manifest), 1);
    writetable(manifest, manifestPath);
    fprintf('Manifest written: %s\n', manifestPath);
end


function [pngFilename, manRow] = renderOneSpectrogram( ...
        fp, idx, audioSourceFolder, audioCache, audioCacheNames, ...
        rawResults, rawResultsNames, preprocParams, postProcOptions, ...
        spectrogramFolder, contextSeconds, freqRangeHz, ...
        winSec, overlapFrac, dynRangeDB)
% Render one Unknown FP as a two-panel figure: linear-frequency
% spectrogram on top, full-file probability trace on the bottom (sliced
% to the same time window), with AT / DT reference lines and detection
% bounds overlaid on both axes.

    fname = basenameOf(fp.AudioFilename);

    % Find audio in cache (prefer cache to a fresh read; fall back to a
    % range-limited audioread if the file was not in the cache).
    audioIdx = find(strcmp(audioCacheNames, fname), 1);
    if ~isempty(audioIdx) && ~isempty(audioCache(audioIdx).audio) ...
            && ~isnan(audioCache(audioIdx).fileFs)
        audio  = audioCache(audioIdx).audio;
        fs     = audioCache(audioIdx).fileFs;
        nTotal = length(audio);
    else
        info   = audioinfo(fullfile(audioSourceFolder, fname));
        fs     = info.SampleRate;
        nTotal = info.TotalSamples;
        audio  = [];   % deferred read below
    end

    % Compute the ±contextSeconds window centred on the detection.
    centreSamp = round((fp.DetectionStartSamp + fp.DetectionEndSamp) / 2);
    halfCtx    = round(contextSeconds * fs);
    readStart  = max(1, centreSamp - halfCtx);
    readEnd    = min(nTotal, centreSamp + halfCtx);

    if isempty(audio)
        audio = audioread(fullfile(audioSourceFolder, fname), [readStart, readEnd]);
        if size(audio, 2) > 1
            audio = audio(:, 1);
        end
        segOffset = readStart - 1;   % sample index in the file of audio(1)
        segment   = audio;
    else
        segment   = audio(readStart:readEnd);
        segOffset = readStart - 1;
    end

    % STFT
    winLen     = max(64, round(winSec * fs));
    overlapLen = round(winLen * overlapFrac);
    nfft       = 2^nextpow2(4 * winLen);
    if length(segment) < winLen
        warning('Segment shorter than window for FP %d; padding.', idx);
        segment = [segment(:); zeros(winLen - length(segment), 1)];
    end
    [Sxx, F, Tspec] = spectrogram(segment, hamming(winLen), overlapLen, nfft, fs);
    P_dB = 20 * log10(abs(Sxx) + eps);

    fMask = F >= freqRangeHz(1) & F <= freqRangeHz(2);
    Pdisp = P_dB(fMask, :);
    Fdisp = F(fMask);

    % Display dynamic range
    maxP = max(Pdisp(:));
    clim = [maxP - dynRangeDB, maxP];

    % Time axes: 0 == start of the read window
    detStartSecInSeg = (fp.DetectionStartSamp - 1 - segOffset) / fs;
    detEndSecInSeg   = (fp.DetectionEndSamp   - 1 - segOffset) / fs;
    windowDurSec     = (readEnd - readStart) / fs;

    % Probability trace within the same window.
    %
    % rawResults(k).probabilities is one entry per STFT frame at
    % targetFs/hopLen Hz (the detector frame rate). The canonical
    % frame -> file-time mapping is the one in
    % Functions/gavdNetPostprocess.m frame2sample (lines 333-346):
    %
    %   t_file_seconds(i) = (i-1) * (hopLen/fsTarget) - padLen/fsTarget
    %
    % where padLen = ceil(windowLen/2) is the pre-pad in samples at
    % fsTarget. NOTE: preprocParams.hopDur is stored separately and may
    % be slightly inconsistent with hopLen/fsTarget (in the Exclude-Chorus
    % model hopDur=0.05 but hopLen/fsTarget=12/250=0.048). gavdNetPostprocess
    % uses hopLen/fsTarget, so we must too. Without these corrections the
    % probability trace and the spectrogram's detection-bound verticals
    % visibly disagree (the trace was sliced from the wrong part of the
    % file).
    rawIdx = find(strcmp(rawResultsNames, fname), 1);
    probsAvailable = false;
    if ~isempty(rawIdx) && all(isfield(preprocParams, {'hopLen','fsTarget','windowLen'}))
        probsFull = rawResults(rawIdx).probabilities;
        if ~isempty(probsFull)
            hopLen    = preprocParams.hopLen;
            fsTarget  = preprocParams.fsTarget;
            windowLen = preprocParams.windowLen;
            padLen    = ceil(windowLen / 2);              % samples at fsTarget
            frameDur  = hopLen / fsTarget;                % seconds per frame
            padOffset = padLen / fsTarget;                % seconds

            probsFull  = probsFull(:);
            nProbs     = numel(probsFull);
            tProbsFull = (0:nProbs-1) * frameDur - padOffset;   % seconds from start of file

            tWinStart  = (readStart - 1) / fs;
            tWinEnd    = (readEnd   - 1) / fs;
            inWin      = tProbsFull >= tWinStart & tProbsFull <= tWinEnd;
            tProbsRel  = tProbsFull(inWin) - tWinStart;          % seconds from start of window
            probsRel   = probsFull(inWin);
            probsAvailable = ~isempty(tProbsRel);
        end
    end

    % Detection statistics (drawn from the per-detection probability
    % slice attached by compareDetectionsToSubsampledTestDataset).
    confScalar = NaN;
    if isfield(fp, 'Confidence') && isnumeric(fp.Confidence) && isscalar(fp.Confidence)
        confScalar = fp.Confidence;
    end
    probSlice = [];
    if isfield(fp, 'probabilities') && ~isempty(fp.probabilities)
        probSlice = fp.probabilities(:);
    end
    if ~isempty(probSlice)
        meanP    = mean(probSlice, 'omitnan');
        maxP_det = max(probSlice, [], 'omitnan');
    else
        meanP    = confScalar;
        maxP_det = confScalar;
    end

    durSec = (fp.DetectionEndSamp - fp.DetectionStartSamp + 1) / fs;
    titleLines = {sprintf('Unknown FP #%d   %s', idx, fname), ...
        sprintf('det: %s to %s (%.2f s)   mean p = %.3f   max p = %.3f', ...
                char(fp.DetectionStartTime, 'yyyy-MM-dd HH:mm:ss'), ...
                char(fp.DetectionEndTime,   'HH:mm:ss'), ...
                durSec, meanP, maxP_det)};

    %% Two-panel figure: spectrogram on top, probability trace below.
    fig = figure('Visible', 'off', 'Position', [100 100 1200 800], 'Color', 'w');
    tl  = tiledlayout(fig, 4, 1, 'TileSpacing', 'tight', 'Padding', 'compact');

    % --- Spectrogram (top 3 tiles) ---------------------------------------
    axSpec = nexttile(tl, 1, [3 1]);
    imagesc(axSpec, Tspec, Fdisp, Pdisp);
    axis(axSpec, 'xy');
    set(axSpec, 'CLim', clim);
    colormap(axSpec, parula);
    cb = colorbar(axSpec); ylabel(cb, 'Power (dB, file-relative)');
    ylabel(axSpec, 'Frequency (Hz)');
    hold(axSpec, 'on');
    yLim = freqRangeHz;
    plot(axSpec, [detStartSecInSeg detStartSecInSeg], yLim, ...
        'r-', 'LineWidth', 1.6);
    plot(axSpec, [detEndSecInSeg   detEndSecInSeg],   yLim, ...
        'r-', 'LineWidth', 1.6);
    plot(axSpec, [detStartSecInSeg detEndSecInSeg], [yLim(2) yLim(2)] - 1, ...
        'r-', 'LineWidth', 1.6);
    xlim(axSpec, [0, windowDurSec]);
    set(axSpec, 'XTickLabel', []);   % share x-axis labels with the bottom tile
    title(axSpec, titleLines, 'Interpreter', 'none');

    % --- Probability trace (bottom 1 tile) -------------------------------
    axProb = nexttile(tl, 4, [1 1]);
    if probsAvailable
        plot(axProb, tProbsRel, probsRel, 'k-', 'LineWidth', 1.0);
        hold(axProb, 'on');
        % AT / DT reference lines from the chosen postproc options.
        AT = NaN; DT = NaN;
        if isfield(postProcOptions, 'AT'), AT = postProcOptions.AT; end
        if isfield(postProcOptions, 'DT'), DT = postProcOptions.DT; end
        if ~isnan(AT)
            yline(axProb, AT, 'r--', sprintf('AT = %.3f', AT), ...
                'LabelHorizontalAlignment', 'left');
        end
        if ~isnan(DT)
            yline(axProb, DT, 'b--', sprintf('DT = %.3f', DT), ...
                'LabelHorizontalAlignment', 'left');
        end
        % Detection-bound verticals matched to the top axis.
        ylProb = [0, 1];
        plot(axProb, [detStartSecInSeg detStartSecInSeg], ylProb, ...
            'r-', 'LineWidth', 1.2);
        plot(axProb, [detEndSecInSeg   detEndSecInSeg],   ylProb, ...
            'r-', 'LineWidth', 1.2);
        ylim(axProb, [0, 1.02]);
    else
        text(axProb, 0.5, 0.5, 'No probability trace available for this file', ...
            'HorizontalAlignment', 'center', 'Units', 'normalized');
        ylim(axProb, [0, 1]);
    end
    xlim(axProb, [0, windowDurSec]);
    xlabel(axProb, 'Time within window (s)');
    ylabel(axProb, 'p (detector)');
    grid(axProb, 'on');

    % linkaxes so any zoom of the saved PNG is in sync (cosmetic for the
    % rendered PNG; useful if the figure is ever opened interactively).
    linkaxes([axSpec, axProb], 'x');

    pngFilename = sprintf('FP_%03d_%s.png', idx, regexprep(fname, '\.[^.]+$', ''));
    exportgraphics(fig, fullfile(spectrogramFolder, pngFilename), 'Resolution', 150);
    close(fig);

    % Build manifest row
    manRow = table( ...
        idx, ...
        string(fname), ...
        string(fp.DetectionStartTime, 'yyyy-MM-dd HH:mm:ss'), ...
        string(fp.DetectionEndTime,   'yyyy-MM-dd HH:mm:ss'), ...
        durSec, ...
        fp.DetectionStartSamp, fp.DetectionEndSamp, fs, ...
        meanP, maxP_det, ...
        getMatchedIoU(fp), ...
        string(pngFilename), ...
        'VariableNames', { ...
            'index', 'filename', 'detection_start_utc', 'detection_end_utc', ...
            'duration_sec', 'start_sample', 'end_sample', 'sample_rate_hz', ...
            'mean_probability', 'max_probability', 'matched_origFP_IoU', ...
            'spectrogram_png'});
end


function tallyAuditedDecisions(manifestPath, metricsPath, chosenRow, analystColumnName)
% Recompute precision / recall / F1 under the user's stated TP definition
% (DiscreteCallsPresent OR DiscreteCallsChorusPresent -> TP) after the
% analyst has filled in the analyst_decision column.

    if ~exist(manifestPath, 'file')
        error('Manifest not found:\n  %s\nRun this script in render mode first.', manifestPath);
    end
    M = readtable(manifestPath, 'TextType', 'string');

    if ~ismember(analystColumnName, M.Properties.VariableNames)
        error('Manifest is missing the analyst column ''%s''.', analystColumnName);
    end

    raw = M.(analystColumnName);
    raw = string(raw);

    validCategories = ["DiscreteCallsPresent", "DiscreteCallsChorusPresent", ...
                       "ChorusPresent", "CallChorusAbsent", ""];
    bad = ~ismember(raw, validCategories);
    if any(bad)
        fprintf('Rows with invalid analyst_decision values:\n');
        disp(table(M.index(bad), raw(bad), 'VariableNames', {'index','value'}));
        warning('Invalid analyst_decision values will be treated as blank (still Unknown).');
        raw(bad) = "";
    end

    % "Discrete-like" Unknowns are the ones that count as TP under the
    % user's TP def (DiscreteCallsPresent OR DiscreteCallsChorusPresent).
    % Chorus / no-call rows stay as FP; blank rows stay Unknown (FP).
    isTPlike = raw == "DiscreteCallsPresent" | raw == "DiscreteCallsChorusPresent";
    isBlank  = raw == "";

    nU       = height(M);
    nTPlike  = sum(isTPlike);
    nBlank   = sum(isBlank);

    %% Recompute metrics under the user's TP def
    %
    % At the chosen operating point, chosenRow reports nTP_adj / nFP_adj /
    % nFN_adj already including the IoU-based inheritance for non-Unknown
    % FPs. The audit only reclassifies Unknowns. Discrete-like Unknowns
    % become TPs (newly discovered ground-truth positives); chorus/no-call
    % Unknowns stay FP; blank rows stay Unknown (FP).

    TP_old = chosenRow.nTP_adj;
    FP_old = chosenRow.nFP_adj;
    FN_old = chosenRow.nFN_adj;

    TP_new = TP_old + nTPlike;
    FP_new = FP_old - nTPlike;    % the discrete-like ones leave FP
    FN_new = FN_old;              % we didn't miss anything new

    P_new  = TP_new / (TP_new + FP_new);
    R_new  = TP_new / (TP_new + FN_new);
    F1_new = 2 * P_new * R_new / (P_new + R_new);

    %% Report
    fid = fopen(metricsPath, 'w');
    cleanup = onCleanup(@() fclose(fid));
    printer = @(varargin) fprintf(fid, varargin{:});

    fprintf('--- Audit summary (%s) ---\n', char(datetime("now")));
    printer('--- Audit summary (%s) ---\n', char(datetime("now")));
    msg = sprintf(['  Manifest               : %s\n', ...
                   '  Total Unknown FPs      : %d\n', ...
                   '  DiscreteCallsPresent   : %d\n', ...
                   '  DiscreteCallsChorusPresent: %d\n', ...
                   '  ChorusPresent          : %d\n', ...
                   '  CallChorusAbsent       : %d\n', ...
                   '  Blank (still Unknown)  : %d\n\n'], ...
                   manifestPath, nU, ...
                   sum(raw == "DiscreteCallsPresent"), ...
                   sum(raw == "DiscreteCallsChorusPresent"), ...
                   sum(raw == "ChorusPresent"), ...
                   sum(raw == "CallChorusAbsent"), ...
                   nBlank);
    fprintf('%s', msg); printer('%s', msg);

    msg = sprintf(['Original chosen point (decision logic = %s):\n', ...
                   '  AT=%.3f DT=%.3f LT_scaler=%.3f MT=%.2fs\n', ...
                   '  TP=%d FP=%d FN=%d  P=%.4f R=%.4f F1=%.4f\n\n'], ...
                   'Discrete-only (user TP def)', ...
                   chosenRow.AT, chosenRow.DT, chosenRow.LT_scaler, chosenRow.MT_s, ...
                   TP_old, FP_old, FN_old, ...
                   chosenRow.precision_adj, chosenRow.recall_adj, chosenRow.f1_adj);
    fprintf('%s', msg); printer('%s', msg);

    msg = sprintf(['After Unknown-FP audit (Discrete-like Unknown -> TP):\n', ...
                   '  TP=%d FP=%d FN=%d  P=%.4f R=%.4f F1=%.4f\n', ...
                   '  Delta P=%+.4f  Delta R=%+.4f  Delta F1=%+.4f\n\n', ...
                   '  (Blank rows still Unknown contribute %d FPs.)\n'], ...
                   TP_new, FP_new, FN_new, P_new, R_new, F1_new, ...
                   P_new - chosenRow.precision_adj, ...
                   R_new - chosenRow.recall_adj, ...
                   F1_new - chosenRow.f1_adj, ...
                   nBlank);
    fprintf('%s', msg); printer('%s', msg);

    fprintf('\nMetrics written: %s\n', metricsPath);
end


%% ------------------------------------------------------------------------
%% Helpers copied verbatim from
%% GAVDNet_SweepPostProcParamsAndCompareToAdjudicatedGT.m so this audit
%% script is self-contained. If those helpers are ever promoted to
%% Functions/ they should be removed here.
%% ------------------------------------------------------------------------

function name = basenameOf(p)
    if isempty(p)
        name = '';
        return;
    end
    p = char(p);
    [~, n, e] = fileparts(p);
    name = [n, e];
end


function [boundaries, confidence, nDetections, eventTimesDT, skipped] = ...
        postprocOneFile(rawEntry, audioIdx, audioCache, preprocParams, postProcOptions)

    boundaries   = zeros(0, 2);
    confidence   = [];
    nDetections  = 0;
    eventTimesDT = NaT(0, 2);
    skipped      = false;

    if isfield(rawEntry, 'failComment') && ~isempty(rawEntry.failComment)
        skipped = true;
        return;
    end

    if audioIdx == 0 || isempty(audioCache(audioIdx).audio)
        skipped = true;
        return;
    end
    audioIn = audioCache(audioIdx).audio;
    fileFs  = audioCache(audioIdx).fileFs;
    if isnan(fileFs)
        skipped = true;
        return;
    end

    probs = rawEntry.probabilities;
    if isempty(probs) || all(isnan(probs))
        skipped = true;
        return;
    end

    try
        [boundaries, ~, confidence] = gavdNetPostprocess( ...
            audioIn, fileFs, probs, preprocParams, postProcOptions);
    catch ME
        warning('gavdNetPostprocess failed on %s: %s', rawEntry.fileName, ME.message);
        skipped = true;
        return;
    end

    nDetections = size(boundaries, 1);

    if nDetections > 0
        fileStart = rawEntry.fileStartDateTime;
        eventTimesDT = NaT(nDetections, 2);
        eventTimesDT(:, 1) = fileStart + seconds((boundaries(:, 1) - 1) / fileFs);
        eventTimesDT(:, 2) = fileStart + seconds((boundaries(:, 2) - 1) / fileFs);
    end
end


function fpGrouping = buildFPGrouping(origFP)
    n = numel(origFP);
    fpGrouping.startSamp = NaN(n, 1);
    fpGrouping.endSamp   = NaN(n, 1);
    fpGrouping.decision  = cell(n, 1);

    for k = 1:n
        s1 = origFP(k).DetectionStartSamp;
        s2 = origFP(k).DetectionEndSamp;
        if isnumeric(s1) && isscalar(s1) && ~isnan(s1)
            fpGrouping.startSamp(k) = s1;
        end
        if isnumeric(s2) && isscalar(s2) && ~isnan(s2)
            fpGrouping.endSamp(k) = s2;
        end
        fpGrouping.decision{k} = origFP(k).analystDecision;
    end

    fpGrouping.byName = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for k = 1:n
        nm = basenameOf(origFP(k).AudioFilename);
        if isempty(nm), continue; end
        if fpGrouping.byName.isKey(nm)
            fpGrouping.byName(nm) = [fpGrouping.byName(nm), int32(k)];
        else
            fpGrouping.byName(nm) = int32(k);
        end
    end
end


function newFPwithAdj = inheritAdjudication(newFP, fpGrouping, iouThreshold)
% Same logic as the sweep script's inheritAdjudication, minus the margin
% fields (not needed here). Tags each new FP with the inherited
% analystDecision or 'Unknown', and stores the IoU of the best match.

    if isempty(newFP)
        newFPwithAdj = struct( ...
            'AudioFilename', {}, 'DetectionStartTime', {}, 'DetectionEndTime', {}, ...
            'DetectionStartSamp', {}, 'DetectionEndSamp', {}, ...
            'analystDecision', {}, 'matchedIoU', {});
        return;
    end

    nNew = numel(newFP);
    newFPwithAdj = newFP;
    for i = 1:nNew
        s1 = newFP(i).DetectionStartSamp;
        s2 = newFP(i).DetectionEndSamp;
        fname = basenameOf(newFP(i).AudioFilename);

        bestIoU = 0;
        bestK   = 0;
        if ~isempty(fname) && fpGrouping.byName.isKey(fname) ...
                && isnumeric(s1) && isnumeric(s2) ...
                && ~isnan(s1) && ~isnan(s2)
            candIdx = fpGrouping.byName(fname);
            c1 = fpGrouping.startSamp(candIdx);
            c2 = fpGrouping.endSamp(candIdx);
            good = ~isnan(c1) & ~isnan(c2);
            if any(good)
                candIdxValid = candIdx(good);
                cg1 = c1(good);
                cg2 = c2(good);
                overlap = max(0, min(s2, cg2) - max(s1, cg1) + 1);
                unionN  = (s2 - s1 + 1) + (cg2 - cg1 + 1) - overlap;
                iouVec  = overlap ./ unionN;
                iouVec(overlap <= 0 | unionN <= 0) = 0;
                [bestIoU, localBest] = max(iouVec);
                if bestIoU > 0
                    bestK = candIdxValid(localBest);
                end
            end
        end

        if bestK > 0 && bestIoU >= iouThreshold
            newFPwithAdj(i).analystDecision = fpGrouping.decision{bestK};
            newFPwithAdj(i).matchedIoU      = bestIoU;
        else
            newFPwithAdj(i).analystDecision = 'Unknown';
            newFPwithAdj(i).matchedIoU      = bestIoU;   % may be 0 or sub-threshold
        end
    end
end


function v = getMatchedIoU(fp)
    if isfield(fp, 'matchedIoU') && isnumeric(fp.matchedIoU) && isscalar(fp.matchedIoU)
        v = fp.matchedIoU;
    else
        v = NaN;
    end
end
