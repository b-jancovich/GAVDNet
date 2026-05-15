% Tune Post-Processing Parameters
%
% This script tunes GAVDNet post-processing parameters by sweeping over a
% 4D grid (AT, DT, LT_scaler, MT) on cached raw detection probabilities
% (the neural network is NOT re-run) and scoring each combination against
% an adjudicated disagreements file produced by the GAVDNetAdjudicator app.
%
% The TP definition is selectable at runtime via the USER INPUT
% `decisionLogic`:
%   'Inclusive'                — DiscreteCallsPresent OR ChorusPresent OR
%                                DiscreteCallsChorusPresent all count as TP.
%                                Matches the methodology that produced the
%                                published post-adjudication numbers
%                                (Functions/reclassifyDisagreementsByLogic.m).
%   'Discrete-only'            — DiscreteCallsPresent OR
%                                DiscreteCallsChorusPresent count as TP.
%   'Strict-discrete'          — Only DiscreteCallsPresent counts as TP.
%   'StrictDiscreteWithMargin' — Pure DiscreteCallsPresent is TP unconditionally;
%                                DiscreteCallsChorusPresent is TP only when the
%                                discrete component is at least
%                                marginThreshold_dB above the surrounding
%                                chorus (measured in the model's training
%                                bandwidth via a band-limited Hilbert envelope).
%                                Strictest setting; rejects bare chorus.
% Anything not promoted to TP under the chosen logic remains a false positive
% (the 'Unknown' tag is reserved for new FPs that couldn't inherit any
% analyst decision via sample-domain IoU >= originalFPMatch_IoUThreshold; it
% remains FP under every logic).
%
% The script:
%   1. Loads raw probabilities, adjudicated disagreements, and trained model.
%   2. Pre-computes a dB margin for every 'DiscreteCallsChorusPresent' FP
%      once, then caches it to disk (used by 'StrictDiscreteWithMargin').
%   3. Sweeps post-processing parameters over the 4D grid, re-running only
%      post-processing for each combination (cached raw probabilities are
%      reused). Outer parfor over combos.
%   4. For every combination, matches new detections against ground truth,
%      then inherits adjudicator decisions for the resulting false positives
%      via sample-domain IoU against the original adjudicated FP set.
%      Unmatched ("Unknown") FPs are conservatively counted as FPs.
%   5. Reports adjusted precision/recall/F1 and chorus-rejection diagnostics
%      per combination to an Excel file, plus a scatter figure of recall vs
%      precision, and auto-selects an operating point using
%      "max recall subject to precision >= precisionFloor".
%
% Required toolboxes:
%   Audio Toolbox, Signal Processing Toolbox, Parallel Computing Toolbox
%   (optional but recommended).
%
% Companion utilities reused (read-only):
%   Functions/gavdNetPostprocess.m
%   Functions/flattenDetections.m
%   Functions/compareDetectionsToSubsampledTestDataset.m
%   Functions/reclassifyDisagreementsByLogic.m
%
% Ben Jancovich, 2026
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%% Init

clear
close all
clc
clear persistent

% **** USER INPUT ****t

% Replace these placeholder paths before running.

rawDetectionsPath             = "E:\GAVDNet\Chagos_DGS\Test Results\Final Test - 2007subset\-10 to 10 Single Exemplar\detector_raw_results.mat";
adjudicatedDisagreementsPath  = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\GAVDNet Data for Publication\Post-Adjudication Results\CPBW_DiegoGarciaSouth_2007_ADJUDICATED\detector_vs_GT_disagreements_07-Jul-2025_08-57-43.mat";
trainedModelPath              = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar\GAVDNet_trained_01-Jul-2025_16-08.mat";
groundtruthPath               = "E:\GAVDNet\Chagos_DGS\Test Data\2007subset\test_dataset_detection_list.mat";   % or .txt for SORP
gtFormat                      = 'CTBTO';                      % 'CTBTO' | 'SORP'
audioSourceFolder             = "E:\GAVDNet\Chagos_DGS\Test Data\2007subset";        % WAV files referenced by raw + disagreements
outputFolder                  = "C:\Users\z5439673\Git\GAVDNet\PostProc Tuning";

% (Optional) path to the original `detector_results_postprocessed.mat`. If
% provided, the script reads `featureFraming`, `frameStandardization`, and
% the original `postProcOptions` from it (used only for diagnostic Excel
% metadata and for verification step 2). Leave as "" to skip.
postprocessedResultsPath      = "";

% Hungarian-matching parameters. detectionTolerance must match the run
% that produced the disagreements file for the sweep results to be
% comparable with the published post-adjudication metrics.
detectionTolerance            = 30;      % seconds (GT match tolerance).
                                         % 30 matches the published
                                         % methodology; tighten to 10 for
                                         % a stricter evaluation.
maxDetectionDuration          = 40;      % seconds (FN window length)

% TP reclassification policy. Determines which adjudicated FP categories
% are promoted to TP under inheritance. Options:
%   'Inclusive'                — DiscreteCallsPresent OR ChorusPresent OR
%                                DiscreteCallsChorusPresent all count as TP.
%                                Matches the published post-adjudication
%                                methodology (recall ~0.98, precision ~0.91
%                                on this dataset at AT=0.5, DT=0.001,
%                                MT=0.1, LT_scaler=0.5).
%   'Discrete-only'            — DiscreteCallsPresent OR
%                                DiscreteCallsChorusPresent count as TP;
%                                bare ChorusPresent remains FP.
%   'Strict-discrete'          — Only DiscreteCallsPresent counts as TP.
%   'StrictDiscreteWithMargin' — In-script policy: DiscreteCallsPresent
%                                always TP; DiscreteCallsChorusPresent TP
%                                only if its dB margin above the
%                                surrounding chorus clears
%                                marginThreshold_dB. Strictest.
% First three delegate to Functions/reclassifyDisagreementsByLogic.m. The
% last uses the local margin-aware reclassifier (reclassifyWithMargin).
%
% PHASE 2 — chorus-rejection measurement run. Set to 'Strict-discrete':
% only DiscreteCallsPresent analyst decisions promote an FP to TP; chorus
% and combined chorus/discrete categories remain FPs. This is the run that
% actually measures the model's chorus-rejection capability under the
% current post-processing chain (Phase 1's Inclusive run promoted chorus
% FPs to TPs and so could not).
%
% 'Strict-discrete' is preferred over 'StrictDiscreteWithMargin' because
% Phase 1 Run 1 showed the dataset has zero DiscreteCallsChorusPresent FPs,
% so the 3 dB margin test never fires; the two policies are equivalent on
% this dataset and 'Strict-discrete' is the cleaner delegating path
% (Functions/reclassifyDisagreementsByLogic.m).
decisionLogic                 = 'Strict-discrete';

% Margin-aware reclassification rule (only used when
% decisionLogic == 'StrictDiscreteWithMargin'; informational otherwise).
marginThreshold_dB            = 3.0;     % discrete must be >= chorus + this (dB)
originalFPMatch_IoUThreshold  = 0.1;     % new FP inherits an original adjudication
                                         % only if max IoU >= this on the same file.
                                         % Lowered from 0.3 -> 0.1 so detections
                                         % that are slightly time-shifted from
                                         % an adjudicated FP still inherit its
                                         % analyst decision (avoids over-counting
                                         % Unknown FPs as real FPs).

% 4D sweep grid (108 combos = 3 x 4 x 3 x 3), focused on the Run-2 winners'
% parameter neighbourhood under decisionLogic = 'Strict-discrete'. This run
% measures the true Pareto front for discrete-only detection (chorus FPs
% remain FPs; only DiscreteCallsPresent analyst decisions promote new FPs
% to TPs).
%
% Coverage rationale:
%   AT in [0.4, 0.5, 0.6]              — centred on Run-2 optimum AT=0.5
%   DT_offsets in [0.001..0.2]         — dense near tight hysteresis
%                                        (Run-2 winners spanned 0.001 -> 0.2)
%   LT_scaler in [0.1, 0.25, 0.5]      — Run-2 winners used 0.10 or 0.50;
%                                        0.25 fills the gap; 0.75 dropped
%   MT in [0.1, 0.3, 0.5]              — Run 2: MT=0.1 was universally
%                                        optimal; 2.0 s dropped
AT_sweep_values               = [0.4, 0.5, 0.6];               % centred on Run-2 optimum AT=0.5
DT_offsets_from_AT            = [0.001, 0.05, 0.1, 0.2];       % dense around tight hysteresis.
                                                                % DT = max(0, min(AT - offset, AT - 1e-3));
                                                                % gavdNetPostprocess asserts DT<AT.
LTscaler_sweep_values         = [0.1, 0.25, 0.5];              % drops 0.75 (Run-2 winners used 0.10 or 0.50)
MT_sweep_values               = [0.1, 0.3, 0.5];               % drops 2.0 s (universally MT=0.1 optimal in Run 2)
AEAVD                         = 0;                              % fixed; AEAVD=1 is expensive

% Operating-point selector
precisionFloor                = 0.95;    % required precision_adj for "chosen"

% dB-margin computation (Stage 2)
marginBandHz                  = [];      % [] -> model.preprocParams.bandwidth (Hz)
marginFlankPad_s              = 5;       % seconds of audio padding before+after FP window
marginMinFlank_s              = 2;       % below this, fall back to 'inside-only' mode
marginEnvelopeSmooth_s        = [];      % [] -> max(0.05, min(0.5, meanTargetCallDuration/4)) (s)
marginPeakWindow_s            = [];      % [] -> meanTargetCallDuration (s); rolling-max window
marginPeakPercentile          = 95;      % robust peak inside FP window (percentile)
marginBaselinePercentile      = 25;      % robust baseline in flanks (percentile)

% Performance
useParallel                   = true;    % parfor over combos (requires Parallel
                                         % Computing Toolbox); inner per-file
                                         % loop is serial inside each combo
audioCacheInRAM               = true;    % audioread every file once at startup
maxDetectionsPerCombo         = 30000;   % combos producing more flat detections
                                         % than this are skipped before
                                         % compareDetectionsToSubsampledTestDataset
                                         % is called. The Hungarian matcher in
                                         % matchpairs allocates a square cost
                                         % matrix sized ~(nDet+nGT)^2 doubles
                                         % (~20 GB at 50000), so combos near the
                                         % low-AT end of the sweep can OOM
                                         % without this guard.

% Output options
saveFullDiagnostics           = false;   % save per-combo augmented disagreement
                                         % struct (large; off by default)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.

%% Determinism

% Seed kept for reproducibility, though calculateAdjudicatedMetricsFast does
% not invoke any random-number generators (the temperature-scaling step from
% the original Functions/calculateAdjudicatedMetrics is intentionally
% skipped). Harmless to leave in place.
rng(0, 'twister');

%% Setup paths

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, "Functions"))

%% Stage 1 - Load and validate inputs

% Create output folder + diary
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
ts = char(datetime("now", "Format", "uuuu-MM-dd_HH-mm-ss"));
diary(fullfile(outputFolder, sprintf('tune_postproc_log_%s.txt', ts)));

fprintf('=== tune_postproc_params_for_discrete_only ===\n');
fprintf('Started: %s\n\n', char(datetime("now")));

% Raw detection results
fprintf('Loading raw detection results:\n  %s\n', rawDetectionsPath);
rawData = load(rawDetectionsPath, 'results');
rawResults = rawData.results;
fprintf('  Loaded %d files of cached probabilities.\n', length(rawResults));

% Adjudicated disagreements
fprintf('Loading adjudicated disagreements:\n  %s\n', adjudicatedDisagreementsPath);
adjData = load(adjudicatedDisagreementsPath, 'disagreements');
disagreements = adjData.disagreements;
nFP_adj = length(disagreements.falsePositives);
nFN_adj = length(disagreements.falseNegatives);
fprintf('  Loaded %d adjudicated false positives, %d adjudicated false negatives.\n', ...
    nFP_adj, nFN_adj);

% Verify adjudication completeness
nUnadj_FP = sum(cellfun(@isempty, {disagreements.falsePositives.analystDecision}));
nUnadj_FN = sum(cellfun(@isempty, {disagreements.falseNegatives.analystDecision}));
if nUnadj_FP > 0 || nUnadj_FN > 0
    error(['Incomplete adjudication: %d FPs and %d FNs have empty analystDecision. ', ...
           'All disagreements must be adjudicated before running this analysis.'], ...
           nUnadj_FP, nUnadj_FN);
end
fprintf('  All disagreements have an analystDecision.\n');

% Trained model
fprintf('Loading trained model:\n  %s\n', trainedModelPath);
modelData = load(trainedModelPath, 'model');
model = modelData.model;
meanTargetCallDuration = model.dataSynthesisParams.meanTargetCallDuration;   % seconds
maxTargetCallDuration  = model.dataSynthesisParams.maxTargetCallDuration;    % seconds
preprocParams = model.preprocParams;
fprintf('  meanTargetCallDuration = %.3f s, maxTargetCallDuration = %.3f s.\n', ...
    meanTargetCallDuration, maxTargetCallDuration);

% Optional: postprocessed results, used only for metadata flags
featureFraming = 'unknown';
frameStandardization = 'unknown';
originalPostProcOptions = [];
if strlength(postprocessedResultsPath) > 0 && exist(postprocessedResultsPath, 'file')
    fprintf('Loading metadata from postprocessed results:\n  %s\n', postprocessedResultsPath);
    postData = load(postprocessedResultsPath, 'featureFraming', 'frameStandardization', 'postProcOptions');
    if isfield(postData, 'featureFraming'),        featureFraming        = postData.featureFraming;        end
    if isfield(postData, 'frameStandardization'),  frameStandardization  = postData.frameStandardization;  end
    if isfield(postData, 'postProcOptions'),       originalPostProcOptions = postData.postProcOptions;     end
end

% Resolve [] defaults that depend on model
if isempty(marginBandHz)
    marginBandHz = preprocParams.bandwidth;
end
if isempty(marginEnvelopeSmooth_s)
    marginEnvelopeSmooth_s = max(0.05, min(0.5, meanTargetCallDuration / 4));
end
if isempty(marginPeakWindow_s)
    marginPeakWindow_s = meanTargetCallDuration;
end
fprintf('Margin band: [%.2f, %.2f] Hz; envelope smoothing %.3f s; peak window %.3f s.\n', ...
    marginBandHz(1), marginBandHz(2), marginEnvelopeSmooth_s, marginPeakWindow_s);

% Combine the set of files we need audio for: every file present in
% rawResults AND every file referenced by an adjudicated FP.
filesFromRaw = arrayfun(@(r) basenameOf(r.fileName), rawResults, 'UniformOutput', false);
filesFromFP  = arrayfun(@(d) basenameOf(d.AudioFilename), disagreements.falsePositives, 'UniformOutput', false);
uniqueFiles = unique([filesFromRaw(:); filesFromFP(:)]);
uniqueFiles = uniqueFiles(~cellfun(@isempty, uniqueFiles));
fprintf('Need audio for %d unique files.\n', length(uniqueFiles));

% Audio cache (filename -> {audio, fileFs}); empty audio means read failed.
audioCache = struct('name', {}, 'audio', {}, 'fileFs', {});
if audioCacheInRAM
    fprintf('Caching audio into RAM (one read per file)...\n');
    audioCache(length(uniqueFiles)).name = '';   % preallocate struct array
    nReadFail = 0;
    for k = 1:length(uniqueFiles)
        fname = uniqueFiles{k};
        audioCache(k).name = fname;
        try
            [audio, fs] = audioread(fullfile(audioSourceFolder, fname));
            % Ensure column vector for gavdNetPostprocess (which requires (:,1))
            if size(audio, 2) > 1
                audio = audio(:, 1);
            end
            audioCache(k).audio = audio;
            audioCache(k).fileFs = fs;
        catch ME
            warning('Audio read failed for %s: %s', fname, ME.message);
            audioCache(k).audio = [];
            audioCache(k).fileFs = NaN;
            nReadFail = nReadFail + 1;
        end
    end
    fprintf('  Cached %d files (%d read failures).\n', length(uniqueFiles), nReadFail);
end
audioCacheNames = {audioCache.name};

%% Stage 2 - Pre-compute dB margins for DiscreteCallsChorusPresent FPs

% Cache file (tied to the adjudication file + model so re-runs skip Stage 2)
[~, adjStem, ~] = fileparts(char(adjudicatedDisagreementsPath));
[~, modelStem, ~] = fileparts(char(trainedModelPath));
marginCachePath = fullfile(outputFolder, ...
    sprintf('margin_cache_%s_%s.mat', adjStem, modelStem));

if exist(marginCachePath, 'file')
    fprintf('Loading cached dB margins from:\n  %s\n', marginCachePath);
    marginData = load(marginCachePath, 'falsePositives');
    disagreements.falsePositives = marginData.falsePositives;
else
    fprintf('Computing dB margins for DiscreteCallsChorusPresent FPs...\n');

    for i = 1:nFP_adj
        fp = disagreements.falsePositives(i);
        % Initialise margin fields for every FP (NaN where N/A)
        disagreements.falsePositives(i).discreteAboveChorus_dB = NaN;
        disagreements.falsePositives(i).baseline_median_dB    = NaN;
        disagreements.falsePositives(i).marginEstimateMode    = 'n/a';

        if ~strcmp(fp.analystDecision, 'DiscreteCallsChorusPresent')
            continue;
        end

        % Need valid sample indices
        if ~isnumeric(fp.DetectionStartSamp) || ~isnumeric(fp.DetectionEndSamp) || ...
           ~isscalar(fp.DetectionStartSamp) || ~isscalar(fp.DetectionEndSamp) || ...
           isnan(fp.DetectionStartSamp) || isnan(fp.DetectionEndSamp)
            disagreements.falsePositives(i).marginEstimateMode = 'nan-samples';
            continue;
        end

        % Look up audio
        baseName = basenameOf(fp.AudioFilename);
        cacheIdx = find(strcmp(audioCacheNames, baseName), 1);
        if isempty(cacheIdx) || isempty(audioCache(cacheIdx).audio)
            % Try direct audioread if cache miss (or cache disabled)
            try
                [audio, fs] = audioread(fullfile(audioSourceFolder, baseName));
                if size(audio, 2) > 1, audio = audio(:, 1); end
            catch
                disagreements.falsePositives(i).marginEstimateMode = 'audio-read-failed';
                continue;
            end
        else
            audio = audioCache(cacheIdx).audio;
            fs    = audioCache(cacheIdx).fileFs;
        end

        % Compute the margin
        [margin_dB, baselineMed_dB, mode] = computeDiscreteAboveChorus(...
            audio, fs, fp.DetectionStartSamp, fp.DetectionEndSamp, ...
            marginBandHz, marginFlankPad_s, marginMinFlank_s, ...
            marginEnvelopeSmooth_s, marginPeakWindow_s, ...
            marginPeakPercentile, marginBaselinePercentile);
        disagreements.falsePositives(i).discreteAboveChorus_dB = margin_dB;
        disagreements.falsePositives(i).baseline_median_dB     = baselineMed_dB;
        disagreements.falsePositives(i).marginEstimateMode     = mode;
    end

    % Save cache
    falsePositives = disagreements.falsePositives; %#ok<NASGU>  (saved name)
    save(marginCachePath, 'falsePositives', '-v7.3');
    fprintf('  Saved margin cache to:\n  %s\n', marginCachePath);
end

% Margin sanity histogram (verification step 1)
plotMarginHistogram(disagreements.falsePositives, marginThreshold_dB, ...
    fullfile(outputFolder, sprintf('margin_histogram_%s.png', ts)));

% Report
mFP = disagreements.falsePositives;
isDCC = strcmp({mFP.analystDecision}, 'DiscreteCallsChorusPresent');
dBvals = [mFP(isDCC).discreteAboveChorus_dB];
fprintf('  DiscreteCallsChorusPresent FPs: %d (with valid margin: %d, NaN: %d)\n', ...
    sum(isDCC), sum(~isnan(dBvals)), sum(isnan(dBvals)));
if any(~isnan(dBvals))
    fprintf('  Margin dB: median = %.2f, IQR = [%.2f, %.2f], range = [%.2f, %.2f]\n', ...
        median(dBvals, 'omitnan'), prctile(dBvals, 25), prctile(dBvals, 75), ...
        min(dBvals), max(dBvals));
    fprintf('  Of these, %d (%.1f%%) meet the %.1f dB threshold.\n', ...
        sum(dBvals >= marginThreshold_dB), ...
        100 * sum(dBvals >= marginThreshold_dB) / sum(~isnan(dBvals)), ...
        marginThreshold_dB);
end

%% Stage 3 - Pre-build adjudication-inheritance groupings

% The originals (disagreements.falsePositives / falseNegatives) are invariant
% across the sweep, so the basename->indices buckets that inheritAdjudication
% / inheritFNAdjudication need can be built once here and reused for every
% combo. This turns the per-combo O(nNew * nOrig) strcmp loop into an O(1)
% bucket lookup plus a vectorised score over the bucket members.
fprintf('Pre-building FP/FN inheritance groupings...\n');
fpGrouping = buildFPGrouping(disagreements.falsePositives);
fnGrouping = buildFNGrouping(disagreements.falseNegatives);
fprintf('  Indexed %d FPs across %d files; %d FNs across %d files.\n', ...
    length(disagreements.falsePositives), fpGrouping.byName.Count, ...
    length(disagreements.falseNegatives), fnGrouping.byName.Count);

%% Stage 4 - 4D sweep over (AT, DT, LT_scaler, MT)

nAT  = length(AT_sweep_values);
nDT  = length(DT_offsets_from_AT);
nLT  = length(LTscaler_sweep_values);
nMT  = length(MT_sweep_values);
combosTotal = nAT * nDT * nLT * nMT;
fprintf('\nStarting %d-combination sweep (%d AT x %d DT x %d LT x %d MT)\n', ...
    combosTotal, nAT, nDT, nLT, nMT);

% Result buffer (will become the output table)
sweepRows = cell(combosTotal, 1);

% Diagnostic per-combo augmented disagreements (only kept in memory if
% saveFullDiagnostics; otherwise we discard between iterations)
diagFolder = '';
if saveFullDiagnostics
    diagFolder = fullfile(outputFolder, 'sweep_disagreements');
    if ~isfolder(diagFolder), mkdir(diagFolder); end
end

% Map rawFileName -> audio cache index, computed once outside the parfor
audioIdxForRaw = zeros(length(rawResults), 1);
for f = 1:length(rawResults)
    baseName = basenameOf(rawResults(f).fileName);
    idx = find(strcmp(audioCacheNames, baseName), 1);
    if ~isempty(idx)
        audioIdxForRaw(f) = idx;
    end
end

% Pre-compute postProcOptions skeleton
postProcSkel = struct( ...
    'AT',    NaN, ...
    'DT',    NaN, ...
    'AEAVD', AEAVD, ...
    'MT',    NaN, ...
    'LT',    NaN, ...
    'LT_scaler', NaN, ...
    'maxTargetCallDuration', maxTargetCallDuration * 1.2);

% Bundle all per-combo-invariant scalars/strings into a single struct so the
% per-combo worker function takes a flat argument list. Built once here and
% broadcast to workers as a small (~hundreds of bytes) frozen value.
inv = struct( ...
    'postProcSkel',                postProcSkel, ...
    'meanTargetCallDuration',      meanTargetCallDuration, ...
    'groundtruthPath',             groundtruthPath, ...
    'detectionTolerance',          detectionTolerance, ...
    'maxDetectionDuration',        maxDetectionDuration, ...
    'gtFormat',                    gtFormat, ...
    'featureFraming',              featureFraming, ...
    'frameStandardization',        frameStandardization, ...
    'originalFPMatch_IoUThreshold', originalFPMatch_IoUThreshold, ...
    'marginThreshold_dB',          marginThreshold_dB, ...
    'decisionLogic',               decisionLogic, ...
    'AEAVD',                       AEAVD, ...
    'precisionFloor',              precisionFloor, ...
    'outputFolder',                char(outputFolder), ...
    'saveFullDiagnostics',         saveFullDiagnostics, ...
    'diagFolder',                  diagFolder, ...
    'maxDetectionsPerCombo',       maxDetectionsPerCombo);

% Unroll the 4D sweep grid to a flat 1D combo index so we can run a single
% parfor over combosTotal. Each component is a column vector that parfor
% recognises as a sliced input.
[iAT_g, iDT_g, iLT_g, iMT_g] = ndgrid(1:nAT, 1:nDT, 1:nLT, 1:nMT);
comboAT     = AT_sweep_values(iAT_g(:));
comboDTbase = DT_offsets_from_AT(iDT_g(:));
comboLT     = LTscaler_sweep_values(iLT_g(:));
comboMT     = MT_sweep_values(iMT_g(:));

% Tier-C speedup: wrap all per-combo-invariant inputs in parallel.pool.Constant
% so they are broadcast to each worker ONCE for the whole pool session, not
% re-sent on every parfor invocation. The audio cache dominates the broadcast
% volume; rawResults, audioIdxForRaw, and the inheritance groupings are also
% invariant across combos.
audioCacheConst     = [];
preprocParamsConst  = [];
rawResultsConst     = [];
audioIdxForRawConst = [];
fpGroupingConst     = [];
fnGroupingConst     = [];
if useParallel
    if isempty(gcp('nocreate'))
        % 4 workers is the memory-safe cap on 64 GB given the ~3.7 GB
        % audio cache per worker plus per-combo matchpairs cost matrices
        % (worst-case ~7 GB at maxDetectionsPerCombo=30000).
        % The 'Processes' profile's NumWorkers ceiling has dropped to 2
        % on this machine; bump it for this session (no saveProfile, so
        % the change is in-memory only) before opening the pool.
        nWorkersWanted = 2;
        cluster = parcluster('Processes');
        if cluster.NumWorkers < nWorkersWanted
            cluster.NumWorkers = nWorkersWanted;
        end
        parpool(cluster, nWorkersWanted);
    end
    audioCacheConst     = parallel.pool.Constant(audioCache);
    preprocParamsConst  = parallel.pool.Constant(preprocParams);
    rawResultsConst     = parallel.pool.Constant(rawResults);
    audioIdxForRawConst = parallel.pool.Constant(audioIdxForRaw);
    fpGroupingConst     = parallel.pool.Constant(fpGrouping);
    fnGroupingConst     = parallel.pool.Constant(fnGrouping);
end

% Progress reporter: workers send their combo index to this DataQueue when
% they finish; afterEach runs the client-side callback in arrival order so
% the diary gets a clean progress line every few combos.
sweepStartTime = datetime('now');
progressQueue = parallel.pool.DataQueue;
clear reportProgress   % reset persistent state in case the script is rerun
afterEach(progressQueue, @(c) reportProgress(c, combosTotal, sweepStartTime));

if useParallel
    parfor c = 1:combosTotal
        currentAT = comboAT(c);
        currentDT = max(0, min(currentAT - comboDTbase(c), currentAT - 1e-3));
        currentLT = comboLT(c);
        currentMT = comboMT(c);
        try
            sweepRows{c} = doOneCombo(c, currentAT, currentDT, currentLT, currentMT, ...
                rawResultsConst.Value, audioIdxForRawConst.Value, audioCacheConst.Value, ...
                preprocParamsConst.Value, fpGroupingConst.Value, fnGroupingConst.Value, inv);
        catch ME
            warning('Combo %d hard failure: %s', c, ME.message);
            sweepRows{c} = [];
        end
        send(progressQueue, c);
    end
else
    for c = 1:combosTotal
        currentAT = comboAT(c);
        currentDT = max(0, min(currentAT - comboDTbase(c), currentAT - 1e-3));
        currentLT = comboLT(c);
        currentMT = comboMT(c);
        try
            sweepRows{c} = doOneCombo(c, currentAT, currentDT, currentLT, currentMT, ...
                rawResults, audioIdxForRaw, audioCache, ...
                preprocParams, fpGrouping, fnGrouping, inv);
        catch ME
            warning('Combo %d hard failure: %s', c, ME.message);
            sweepRows{c} = [];
        end
        send(progressQueue, c);
    end
end

% Cleanup of the per-combo temp comparison files (one per combo under
% outputFolder). Files only exist for combos that reached the save() call
% inside doOneCombo before any try/catch return.
tempPattern = fullfile(outputFolder, 'temp_combo_*.mat');
tempFiles = dir(tempPattern);
for k = 1:numel(tempFiles)
    delete(fullfile(tempFiles(k).folder, tempFiles(k).name));
end

% Drop empty rows from failures, concatenate
sweepRows = sweepRows(~cellfun(@isempty, sweepRows));
if isempty(sweepRows)
    error('Sweep produced no rows; check raw data and configuration.');
end
sweepTable = vertcat(sweepRows{:});

sweepEndTime = datetime('now');
fprintf('\nSweep elapsed: %s\n', char(sweepEndTime - sweepStartTime));

%% Stage 5 - Outputs

% --- Operating-point picker -----------------------------------------------
filtered = sweepTable(sweepTable.precision_adj >= precisionFloor, :);
if ~isempty(filtered)
    [~, idxSort] = sortrows(filtered, ...
        {'recall_adj', 'f1_adj', 'nUnknownFPs'}, {'descend', 'descend', 'ascend'});
    chosenRow = filtered(idxSort(1), :);
    metPrecisionFloor = true;
    fprintf('\nChosen operating point (precision >= %.3f, max recall):\n', precisionFloor);
else
    [~, idxSort] = sortrows(sweepTable, ...
        {'precision_adj', 'recall_adj'}, {'descend', 'descend'});
    chosenRow = sweepTable(idxSort(1), :);
    metPrecisionFloor = false;
    warning('tune_postproc:noComboMetFloor', ...
        'No combo met precision floor %.3f; returning highest-precision combo.', precisionFloor);
    fprintf('\nFallback operating point (highest precision; no combo met %.3f):\n', precisionFloor);
end
fprintf(['  AT=%.4f DT=%.4f LT_scaler=%.3f MT=%.3f s\n', ...
         '  precision_adj=%.4f  recall_adj=%.4f  f1_adj=%.4f  nUnknownFPs=%d\n'], ...
    chosenRow.AT, chosenRow.DT, chosenRow.LT_scaler, chosenRow.MT_s, ...
    chosenRow.precision_adj, chosenRow.recall_adj, chosenRow.f1_adj, chosenRow.nUnknownFPs);

% Also report the max-F1 combo
[~, idxF1] = max(sweepTable.f1_adj);
maxF1Row = sweepTable(idxF1, :);
fprintf('Max-F1 operating point:\n');
fprintf(['  AT=%.4f DT=%.4f LT_scaler=%.3f MT=%.3f s\n', ...
         '  precision_adj=%.4f  recall_adj=%.4f  f1_adj=%.4f\n'], ...
    maxF1Row.AT, maxF1Row.DT, maxF1Row.LT_scaler, maxF1Row.MT_s, ...
    maxF1Row.precision_adj, maxF1Row.recall_adj, maxF1Row.f1_adj);

% --- Write Excel (single fresh write) -------------------------------------
excelPath = fullfile(outputFolder, sprintf('tune_postproc_results_%s.xlsx', ts));
writetable(sweepTable, excelPath, 'Sheet', 'Sweep');

% Recommendations sheet
rec = chosenRow;
rec.Selection = "ChosenByPrecisionFloor";
rec.MetPrecisionFloor = metPrecisionFloor;
rec = movevars(rec, {'Selection', 'MetPrecisionFloor'}, 'Before', 'ComboIdx');
recF1 = maxF1Row;
recF1.Selection = "MaxF1";
recF1.MetPrecisionFloor = true;
recF1 = movevars(recF1, {'Selection', 'MetPrecisionFloor'}, 'Before', 'ComboIdx');
writetable([rec; recF1], excelPath, 'Sheet', 'Recommendations');
fprintf('\nExcel written: %s\n', excelPath);

% --- Scatter figure -------------------------------------------------------
figPath = fullfile(outputFolder, sprintf('operating_point_%s.png', ts));
plotOperatingPointScatter(sweepTable, chosenRow, maxF1Row, precisionFloor, decisionLogic, figPath);

% --- Sweep summary .mat ---------------------------------------------------
summaryPath = fullfile(outputFolder, sprintf('sweep_summary_%s.mat', ts));
chosenPostProcOptions = struct( ...
    'AT',    chosenRow.AT, ...
    'DT',    chosenRow.DT, ...
    'AEAVD', chosenRow.AEAVD, ...
    'MT',    chosenRow.MT_s, ...
    'LT',    chosenRow.LT_s, ...
    'LT_scaler', chosenRow.LT_scaler, ...
    'maxTargetCallDuration', maxTargetCallDuration * 1.2);
save(summaryPath, 'sweepTable', 'chosenRow', 'maxF1Row', ...
    'chosenPostProcOptions', 'metPrecisionFloor', ...
    'marginThreshold_dB', 'precisionFloor', ...
    'AT_sweep_values', 'DT_offsets_from_AT', 'LTscaler_sweep_values', 'MT_sweep_values', ...
    'originalPostProcOptions', '-v7.3');
fprintf('Sweep summary saved: %s\n', summaryPath);

%% Verification readout (text only)

fprintf('\n--- Verification snapshot ---\n');
fprintf('DT range used: [%.4f, %.4f]; all DT < AT: %s\n', ...
    min(sweepTable.DT), max(sweepTable.DT), ...
    string(all(sweepTable.DT < sweepTable.AT)));
fprintf('Combos with zero Unknown FPs: %d / %d\n', ...
    sum(sweepTable.nUnknownFPs == 0), height(sweepTable));
fprintf('Precision floor met by %d / %d combos.\n', height(filtered), height(sweepTable));

fprintf('\nFinished: %s\n', char(datetime("now")));
diary off

clearvars -except sweepTable chosenRow maxF1Row chosenPostProcOptions ...
    metPrecisionFloor marginThreshold_dB precisionFloor outputFolder ...
    excelPath summaryPath figPath
clear persistent

% =========================================================================
% Local functions
% =========================================================================

function row = doOneCombo(c, currentAT, currentDT, currentLT, currentMT, ...
        rawResults, audioIdxForRaw, audioCache, preprocParams, ...
        fpGrouping, fnGrouping, inv)
% Per-combo worker function. Returns the result-table row, or [] if the
% combo couldn't be scored (GT comparison failed, or adjusted-metrics
% calculation failed). The caller filters out empty cells before
% vertcat()ing the rows into the final sweepTable.
%
% Designed for parfor: takes only plain values (Constants are unwrapped at
% the call site before doOneCombo is invoked) so it is also callable from
% the serial (useParallel=false) branch with no per-call cost.

    row = [];

    currentPostProcOptions = inv.postProcSkel;
    currentPostProcOptions.AT        = currentAT;
    currentPostProcOptions.DT        = currentDT;
    currentPostProcOptions.MT        = currentMT;
    currentPostProcOptions.LT_scaler = currentLT;
    currentPostProcOptions.LT        = inv.meanTargetCallDuration * currentLT;

    % --- Step 2: Per-file postprocess (serial inside parfor; nested parfor
    % is not supported by MATLAB).
    currentResults = rawResults;   % local writable copy of the struct array
    nFiles = length(currentResults);
    ppEventSampleBoundaries = cell(nFiles, 1);
    ppConfidence            = cell(nFiles, 1);
    ppNDetections           = zeros(nFiles, 1);
    ppEventTimesDT          = cell(nFiles, 1);
    ppSkipped               = false(nFiles, 1);

    for f = 1:nFiles
        [b, conf, n, t, s] = postprocOneFile(currentResults(f), ...
            audioIdxForRaw(f), audioCache, ...
            preprocParams, currentPostProcOptions);
        ppEventSampleBoundaries{f} = b;
        ppConfidence{f}            = conf;
        ppNDetections(f)           = n;
        ppEventTimesDT{f}          = t;
        ppSkipped(f)               = s;
    end

    for f = 1:nFiles
        currentResults(f).eventSampleBoundaries = ppEventSampleBoundaries{f};
        currentResults(f).confidence            = ppConfidence{f};
        currentResults(f).nDetections           = ppNDetections(f);
        currentResults(f).eventTimesDT          = ppEventTimesDT{f};
    end

    % --- Step 3: Flatten detections to per-detection rows ----------------
    flatDetections = flattenDetections(currentResults, preprocParams);
    if isempty(fieldnames(flatDetections))
        flatDetections = struct([]);
    end
    nPositivesDetector_thisCombo = length(flatDetections);

    % Predicted-OOM short-circuit: compareDetectionsToSubsampledTestDataset
    % calls matchpairs, which allocates a ~(nDet+nGT)^2 dense double cost
    % matrix. At low AT values (near 0.5) nDet can run into the hundreds of
    % thousands, blowing past MATLAB's max-array-size preference. Skip the
    % combo cleanly here rather than paying for save() + GT load + matchpairs
    % init only for matchpairs to throw.
    if nPositivesDetector_thisCombo > inv.maxDetectionsPerCombo
        warning(['Combo %d: %d detections exceeds maxDetectionsPerCombo (%d) -- ', ...
                 'skipping to avoid OOM in matchpairs.'], ...
                 c, nPositivesDetector_thisCombo, inv.maxDetectionsPerCombo);
        return;
    end

    % --- Step 4: Compare to GT (writes a per-combo temp .mat) -----------
    % Per-combo unique filename so parallel workers do not collide. -v6 is
    % uncompressed and noticeably faster than -v7.3 for this small ephemeral
    % file (~1 MB); it also avoids the HDF5 init overhead. The temp file is
    % cleaned up by a single glob delete after the sweep finishes.
    tempResultsPath = fullfile(inv.outputFolder, sprintf('temp_combo_%05d.mat', c));
    results = flatDetections; %#ok<NASGU>  (variable name read by compareDetectionsToSubsampledTestDataset)
    featureFraming = inv.featureFraming;
    frameStandardization = inv.frameStandardization;
    save(tempResultsPath, 'results', 'featureFraming', ...
        'frameStandardization', 'currentPostProcOptions', '-v6');

    try
        [metrics_orig, newFP, newFN] = compareDetectionsToSubsampledTestDataset( ...
            inv.groundtruthPath, tempResultsPath, inv.detectionTolerance, ...
            inv.maxDetectionDuration, inv.gtFormat);
    catch ME
        warning('Combo %d: comparison failed: %s', c, ME.message);
        return;
    end

    % --- Step 5: Inherit adjudication for new FPs/FNs --------------------
    comboDisagreements = struct( ...
        'falsePositives', inheritAdjudication(newFP, fpGrouping, inv.originalFPMatch_IoUThreshold), ...
        'falseNegatives', inheritFNAdjudication(newFN, fnGrouping, inv.detectionTolerance));

    % --- Step 6: Reclassify FP/FN per the configured decision logic -----
    % Dispatch on inv.decisionLogic:
    %   - 'StrictDiscreteWithMargin' uses the local margin-aware path
    %     (DiscreteCallsChorusPresent only counts as TP when its dB margin
    %     above the surrounding chorus clears inv.marginThreshold_dB).
    %   - All other recognised policies delegate to
    %     Functions/reclassifyDisagreementsByLogic.m, the same routine
    %     that produces the published post-adjudication numbers via
    %     Utilitiy Scripts/GAVDNet_adjudicated_performance_metrics.m.
    switch inv.decisionLogic
        case 'StrictDiscreteWithMargin'
            [FP_to_TP, FN_to_TN] = reclassifyWithMargin( ...
                comboDisagreements, inv.marginThreshold_dB);
        case {'Inclusive', 'Discrete-only', 'Strict-discrete', 'Chorus-aware'}
            [~, FP_to_TP, FN_to_TN] = reclassifyDisagreementsByLogic( ...
                comboDisagreements, inv.decisionLogic);
        otherwise
            error('Unknown decisionLogic: %s', inv.decisionLogic);
    end

    % --- Step 7: Calculate adjusted metrics (fast local version) --------
    try
        metrics_adj = calculateAdjudicatedMetricsFast( ...
            metrics_orig.nPositivesGT, ...
            metrics_orig.nTruePositives, ...
            nPositivesDetector_thisCombo, ...
            comboDisagreements, FP_to_TP, FN_to_TN, ...
            flatDetections, ...
            metrics_orig.totalAudioDuration_sec, ...
            inv.detectionTolerance, ...
            inv.decisionLogic);
    catch ME
        warning('Combo %d: adjusted-metrics failed: %s', c, ME.message);
        return;
    end

    % --- Step 8: Per-combo chorus-rejection diagnostics ------------------
    decs = {comboDisagreements.falsePositives.analystDecision};
    marginVals = [comboDisagreements.falsePositives.discreteAboveChorus_dB];

    isUnknown  = strcmp(decs, 'Unknown');
    isDiscrete = strcmp(decs, 'DiscreteCallsPresent');
    isChorus   = strcmp(decs, 'ChorusPresent');
    isDPCh     = strcmp(decs, 'DiscreteCallsChorusPresent');
    isNoCall   = strcmp(decs, 'CallChorusAbsent');

    dpchPass = isDPCh & (marginVals >= inv.marginThreshold_dB);
    dpchFail = isDPCh & ~(marginVals >= inv.marginThreshold_dB);   % NaN >= -> false

    nUnknownFPs                   = sum(isUnknown);
    nDiscreteRecovered            = sum(isDiscrete);
    nChorusKeptFP                 = sum(isChorus);
    nDiscretePlusChorusRecovered  = sum(dpchPass);
    nDiscretePlusChorusRejected   = sum(dpchFail);
    nNoCallKeptFP                 = sum(isNoCall);

    % --- Step 9: Build the result row ------------------------------------
    row = table( ...
        c, currentAT, currentDT, currentMT, currentLT, ...
        inv.meanTargetCallDuration * currentLT, inv.AEAVD, ...
        inv.marginThreshold_dB, inv.precisionFloor, ...
        metrics_adj.precision, metrics_adj.recall, metrics_adj.f1Score, ...
        metrics_adj.auc, ...
        metrics_adj.nTruePositives, metrics_adj.nFalsePositives, ...
        metrics_adj.nFalseNegatives, metrics_adj.nTrueNegatives, ...
        nUnknownFPs, nDiscreteRecovered, nChorusKeptFP, ...
        nDiscretePlusChorusRecovered, nDiscretePlusChorusRejected, ...
        nNoCallKeptFP, ...
        metrics_orig.precision, metrics_orig.recall, metrics_orig.f1Score, ...
        metrics_orig.nFalsePositives, metrics_orig.nFalseNegatives, ...
        nPositivesDetector_thisCombo, ...
        'VariableNames', { ...
        'ComboIdx', 'AT', 'DT', 'MT_s', 'LT_scaler', 'LT_s', 'AEAVD', ...
        'MarginThreshold_dB', 'PrecisionFloor', ...
        'precision_adj', 'recall_adj', 'f1_adj', 'auc_adj', ...
        'nTP_adj', 'nFP_adj', 'nFN_adj', 'nTN_adj', ...
        'nUnknownFPs', 'nDiscreteRecovered', 'nChorusKeptFP', ...
        'nDiscretePlusChorusRecovered', 'nDiscretePlusChorusRejected', ...
        'nNoCallKeptFP', ...
        'precision_orig', 'recall_orig', 'f1_orig', ...
        'nFP_orig', 'nFN_orig', ...
        'nDetections_thisCombo'});

    % --- Step 10: Optional per-combo diagnostics -------------------------
    if inv.saveFullDiagnostics
        diagFile = fullfile(inv.diagFolder, sprintf( ...
            'combo_%04d_AT%.4f_DT%.4f_LT%.3f_MT%.3f.mat', ...
            c, currentAT, currentDT, currentLT, currentMT));
        save(diagFile, 'comboDisagreements', 'currentPostProcOptions', '-v6');
    end
end

function reportProgress(~, totalCombos, startTime)
% Client-side callback invoked by afterEach on the progress DataQueue.
% Emits a single diary line every 5 combos with elapsed time and ETA.
    persistent nDone
    if isempty(nDone), nDone = 0; end
    nDone = nDone + 1;
    if mod(nDone, 5) == 0 || nDone == totalCombos
        elapsed = datetime('now') - startTime;
        if nDone > 0
            eta = elapsed * (totalCombos - nDone) / nDone;
        else
            eta = duration(0, 0, 0);
        end
        fprintf('  Sweep progress: %d/%d combos done (elapsed %s, ETA %s)\n', ...
            nDone, totalCombos, char(elapsed), char(eta));
    end
end

function metrics = calculateAdjudicatedMetricsFast(...
    nPositivesGT_original, nTruePositives_original, nPositivesDetector, ...
    disagreements, FP_becomes_TP, FN_becomes_TN, detectorResults, ...
    totalAudioDuration, detectionTolerance, logicName) %#ok<INUSD>
% Fast in-script replacement for Functions/calculateAdjudicatedMetrics that
% computes only the fields the sweep table actually reads:
%   precision, recall, f1Score, auc, nTruePositives, nFalsePositives,
%   nFalseNegatives, nTrueNegatives.
%
% Speedups vs the shared function:
%   1) FP-to-detection matching is O(N+M) via a containers.Map keyed on
%      "fileName|posixtime", replacing an O(N x M) nested loop with
%      per-iteration isfield / strcmp / datetime arithmetic.
%   2) The unused performance curve, PR curve, temperature-scaling
%      optimisation, second perfcurve call, and confidence-distribution
%      analysis are not computed at all.
%
% Counts and precision/recall/F1 reproduce the original arithmetic exactly.
% AUC matches the original to within perfcurve's tolerance (same inputs,
% one perfcurve call).
%
% Unused arguments (totalAudioDuration, detectionTolerance, logicName) are
% kept for call-site signature parity with calculateAdjudicatedMetrics.

    FP = disagreements.falsePositives;
    FN = disagreements.falseNegatives;
    nFP_to_TP = sum(FP_becomes_TP);
    nFN_to_TN = sum(FN_becomes_TN);

    nTP_adj = nTruePositives_original + nFP_to_TP;
    nFP_adj = length(FP) - nFP_to_TP;
    nFN_adj = length(FN) - nFN_to_TN;
    nTN_adj = nFN_to_TN;
    nPositivesGT_adj = nPositivesGT_original + nFP_to_TP - nFN_to_TN;

    if nPositivesGT_adj > 0
        recall = nTP_adj / nPositivesGT_adj;
    else
        recall = NaN;
    end
    if nPositivesDetector > 0
        precision = nTP_adj / nPositivesDetector;
    else
        precision = NaN;
    end
    if ~isnan(precision) && ~isnan(recall) && (precision + recall) > 0
        f1Score = 2 * precision * recall / (precision + recall);
    else
        f1Score = NaN;
    end

    % --- AUC via fast label assignment ----------------------------------
    AUC = NaN;
    nDet = length(detectorResults);
    if nDet == 0
        metrics = packAdjMetrics(precision, recall, f1Score, AUC, ...
            nTP_adj, nFP_adj, nFN_adj, nTN_adj);
        return;
    end

    % Pre-extract confidence + composite key (filename|posixtime) per
    % detection. Sub-microsecond precision is well below the original
    % 1e-6 s match tolerance.
    confidenceScores = NaN(nDet, 1);
    detKeys = strings(nDet, 1);
    for j = 1:nDet
        entry = detectorResults(j);
        if isfield(entry, 'confidence') && isnumeric(entry.confidence) && isscalar(entry.confidence)
            confidenceScores(j) = entry.confidence;
        end
        if isfield(entry, 'eventStartTime') && isfield(entry, 'fileName') ...
                && isdatetime(entry.eventStartTime) && isscalar(entry.eventStartTime) ...
                && ~isnat(entry.eventStartTime) && ~isempty(entry.fileName)
            detKeys(j) = string(entry.fileName) + "|" + ...
                sprintf("%.7f", posixtime(entry.eventStartTime));
        end
    end

    % Build map from key -> detection index. Preserve "first wins" on
    % duplicates to match the break-on-first-match semantics of the
    % original function.
    keyToIdx = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    for j = 1:nDet
        if detKeys(j) ~= ""
            k = char(detKeys(j));
            if ~keyToIdx.isKey(k)
                keyToIdx(k) = int32(j);
            end
        end
    end

    % Labels default to 1 (TP). For each adjudicated FP, find its detection
    % and downgrade to 0 only if the analyst did not promote it to TP.
    resultLabels = ones(nDet, 1);
    for i = 1:length(FP)
        fpStart = FP(i).DetectionStartTime;
        if isdatetime(fpStart) && isscalar(fpStart) && ~isnat(fpStart) ...
                && isfield(FP, 'AudioFilename') && ~isempty(FP(i).AudioFilename)
            key = char(string(FP(i).AudioFilename) + "|" + ...
                sprintf("%.7f", posixtime(fpStart)));
            if keyToIdx.isKey(key) && ~FP_becomes_TP(i)
                resultLabels(keyToIdx(key)) = 0;
            end
        end
    end

    validIdx = ~isnan(confidenceScores);
    cv = confidenceScores(validIdx);
    lv = resultLabels(validIdx);
    if ~isempty(cv) && numel(unique(lv)) >= 2
        try
            [~, ~, ~, AUC] = perfcurve(lv, cv, 1, ...
                'XCrit', 'fpr', 'YCrit', 'tpr');
        catch
            AUC = NaN;
        end
    end

    metrics = packAdjMetrics(precision, recall, f1Score, AUC, ...
        nTP_adj, nFP_adj, nFN_adj, nTN_adj);
end

function m = packAdjMetrics(precision, recall, f1Score, auc, nTP, nFP, nFN, nTN)
% Minimal-field metrics struct consumed by the sweep table.
    m = struct( ...
        'precision', precision, 'recall', recall, 'f1Score', f1Score, ...
        'auc', auc, ...
        'nTruePositives',  nTP, 'nFalsePositives', nFP, ...
        'nFalseNegatives', nFN, 'nTrueNegatives',  nTN);
end

function name = basenameOf(p)
% Returns the bare filename (no path) from a char/string filename or path.
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
% Run gavdNetPostprocess on one file's cached probabilities. Returns the
% boundaries, per-region confidence, count, datetime boundaries, and a
% skipped-flag (true if we couldn't run because of bad input).

    boundaries   = zeros(0, 2);
    confidence   = [];
    nDetections  = 0;
    eventTimesDT = NaT(0, 2);
    skipped      = false;

    % Skip files that failed during inference
    if isfield(rawEntry, 'failComment') && ~isempty(rawEntry.failComment)
        skipped = true;
        return;
    end

    % Audio (required by gavdNetPostprocess, even when AEAVD=0)
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

    % Sanity guard against length mismatches: trim or skip if implausible
    try
        [boundaries, ~, confidence] = gavdNetPostprocess( ...
            audioIn, fileFs, probs, preprocParams, postProcOptions);
    catch ME
        warning('gavdNetPostprocess failed on %s: %s', rawEntry.fileName, ME.message);
        skipped = true;
        return;
    end

    nDetections = size(boundaries, 1);

    % Convert sample boundaries to datetime (without using sampleDomainTimeVector
    % to avoid broadcasting a huge per-sample datetime array to each worker)
    if nDetections > 0
        fileStart = rawEntry.fileStartDateTime;
        eventTimesDT = NaT(nDetections, 2);
        eventTimesDT(:, 1) = fileStart + seconds((boundaries(:, 1) - 1) / fileFs);
        eventTimesDT(:, 2) = fileStart + seconds((boundaries(:, 2) - 1) / fileFs);
    end
end

function [margin_dB, baseline_med_dB, modeTag] = computeDiscreteAboveChorus( ...
    audio, fs, startSamp, endSamp, bandHz, flankPad_s, minFlank_s, ...
    smooth_s, peakWin_s, peakPct, basePct)
% Bandpassed-envelope dB margin between a discrete call inside [startSamp,
% endSamp] and the surrounding chorus in padded flanks. Returns NaN with a
% modeTag explaining the fallback path when the estimate is degraded.

    margin_dB       = NaN;
    baseline_med_dB = NaN;
    modeTag         = 'two-flank';

    fileSamps = numel(audio);
    if startSamp < 1 || endSamp > fileSamps || endSamp <= startSamp
        modeTag = 'invalid-window';
        return;
    end

    % Padded read window
    padSamps  = round(flankPad_s * fs);
    readStart = max(1, startSamp - padSamps);
    readStop  = min(fileSamps, endSamp + padSamps);
    seg = audio(readStart:readStop);
    preFlankN  = startSamp - readStart;
    postFlankN = readStop  - endSamp;

    % Bandpass to the model's training band (Butterworth, order 4, filtfilt)
    nyq = fs / 2;
    bandNorm = bandHz / nyq;
    bandNorm = max(min(bandNorm, 1 - 1e-6), 1e-6);
    [bb, aa] = butter(4, bandNorm, 'bandpass');
    try
        filtered = filtfilt(bb, aa, seg);
    catch
        % Segment too short for filtfilt
        modeTag = 'segment-too-short';
        return;
    end

    % Hilbert envelope, smoothed
    env = abs(hilbert(filtered));
    smoothN = max(1, round(smooth_s * fs));
    env = movmean(env, smoothN);

    % Indices of the inside window within the read segment
    insideStart = preFlankN + 1;
    insideStop  = numel(seg) - postFlankN;
    if insideStop < insideStart
        modeTag = 'inside-empty';
        return;
    end
    env_in = env(insideStart:insideStop);

    % Robust peak (rolling-max over a call-duration window, then percentile)
    peakWinN = max(1, round(peakWin_s * fs));
    if peakWinN > 1 && numel(env_in) >= peakWinN
        env_peakwin = movmax(env_in, peakWinN);
    else
        env_peakwin = env_in;
    end
    peak_linear = prctile(env_peakwin, peakPct);

    % Baseline from flanks (preferred) or inside-only (fallback)
    preFlank_s  = preFlankN  / fs;
    postFlank_s = postFlankN / fs;
    if (preFlank_s >= minFlank_s) && (postFlank_s >= minFlank_s)
        flanks = [env(1:preFlankN); env(end-postFlankN+1:end)];
        modeTag = 'two-flank';
    elseif preFlank_s >= minFlank_s
        flanks = env(1:preFlankN);
        modeTag = 'one-flank-pre';
    elseif postFlank_s >= minFlank_s
        flanks = env(end-postFlankN+1:end);
        modeTag = 'one-flank-post';
    else
        % No usable flank; use the bottom percentile of the inside as baseline.
        flanks = env_in;
        modeTag = 'inside-only';
    end
    baseline_linear = prctile(flanks, basePct);
    baseline_med_dB = 20 * log10(median(flanks) + eps);

    % dB margin
    if baseline_linear <= 0 || peak_linear <= 0 || ~isfinite(peak_linear) || ~isfinite(baseline_linear)
        margin_dB = NaN;
    else
        margin_dB = 20 * log10(peak_linear / baseline_linear);
    end
end

function fpGrouping = buildFPGrouping(origFP)
% Pre-compute a basename->indices bucket plus packed numeric/cell columns for
% the original (invariant) adjudicated false-positive list. Used by
% inheritAdjudication to do O(1) bucket lookup + vectorised IoU scoring.
% Built once outside the combo sweep and reused across all combos.

    n = numel(origFP);
    fpGrouping.startSamp = NaN(n, 1);
    fpGrouping.endSamp   = NaN(n, 1);
    fpGrouping.decision  = cell(n, 1);
    fpGrouping.discreteAboveChorus_dB = NaN(n, 1);
    fpGrouping.marginEstimateMode     = cell(n, 1);

    hasMarginField    = isfield(origFP, 'discreteAboveChorus_dB');
    hasModeField      = isfield(origFP, 'marginEstimateMode');

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
        if hasMarginField
            v = origFP(k).discreteAboveChorus_dB;
            if isnumeric(v) && isscalar(v)
                fpGrouping.discreteAboveChorus_dB(k) = v;
            end
        end
        if hasModeField
            fpGrouping.marginEstimateMode{k} = origFP(k).marginEstimateMode;
        else
            fpGrouping.marginEstimateMode{k} = 'n/a';
        end
    end

    % basename -> int32 vector of indices into origFP
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

function fnGrouping = buildFNGrouping(origFN)
% Pre-compute a basename->indices bucket and packed datetime column for the
% original adjudicated false-negative list. Used by inheritFNAdjudication.

    n = numel(origFN);
    fnGrouping.startTime = NaT(n, 1);
    fnGrouping.decision  = cell(n, 1);
    for k = 1:n
        t = origFN(k).DetectionStartTime;
        if isdatetime(t) && isscalar(t) && ~isnat(t)
            fnGrouping.startTime(k) = t;
        end
        fnGrouping.decision{k} = origFN(k).analystDecision;
    end

    fnGrouping.byName = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for k = 1:n
        nm = basenameOf(origFN(k).AudioFilename);
        if isempty(nm), continue; end
        if fnGrouping.byName.isKey(nm)
            fnGrouping.byName(nm) = [fnGrouping.byName(nm), int32(k)];
        else
            fnGrouping.byName(nm) = int32(k);
        end
    end
end

function newFPwithAdj = inheritAdjudication(newFP, fpGrouping, iouThreshold)
% For each new (sweep-produced) FP, find the original adjudicated FP on the
% same audio file with the largest sample-domain IoU. If max IoU is at least
% iouThreshold, inherit analystDecision and discreteAboveChorus_dB. Otherwise
% tag analystDecision = 'Unknown'.
%
% Fast path: takes a pre-built fpGrouping (see buildFPGrouping) so the
% basename->indices mapping is O(1) per new FP and the IoU scoring against
% candidates is vectorised. Equivalent in behaviour to the original
% O(nNew * nOrig) implementation, including the strict-greater-than
% tie-break (MATLAB's max returns the first index on ties, matching the
% original accumulator pattern).

    if isempty(newFP)
        newFPwithAdj = struct( ...
            'AudioFilename', {}, 'DetectionStartTime', {}, 'DetectionEndTime', {}, ...
            'DetectionStartSamp', {}, 'DetectionEndSamp', {}, ...
            'analystDecision', {}, 'discreteAboveChorus_dB', {}, ...
            'marginEstimateMode', {}, 'matchedIoU', {});
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
            newFPwithAdj(i).analystDecision        = fpGrouping.decision{bestK};
            newFPwithAdj(i).discreteAboveChorus_dB = fpGrouping.discreteAboveChorus_dB(bestK);
            newFPwithAdj(i).marginEstimateMode     = fpGrouping.marginEstimateMode{bestK};
            newFPwithAdj(i).matchedIoU             = bestIoU;
        else
            newFPwithAdj(i).analystDecision        = 'Unknown';
            newFPwithAdj(i).discreteAboveChorus_dB = NaN;
            newFPwithAdj(i).marginEstimateMode     = 'n/a';
            newFPwithAdj(i).matchedIoU             = bestIoU;   % may be 0 or sub-threshold
        end
    end
end

function newFNwithAdj = inheritFNAdjudication(newFN, fnGrouping, detectionTolerance)
% For each new FN, inherit analystDecision from the original FN with the
% smallest |DetectionStartTime - newFN.DetectionStartTime| on the same file
% within detectionTolerance seconds. Otherwise tag as 'Unknown'.
%
% Fast path using a pre-built fnGrouping (see buildFNGrouping). Vectorised
% time-difference scoring per file bucket; preserves the first-wins tie
% semantics of the original (min returns the first index on ties).

    if isempty(newFN)
        newFNwithAdj = struct( ...
            'AudioFilename', {}, 'DetectionStartTime', {}, 'DetectionEndTime', {}, ...
            'DetectionStartSamp', {}, 'DetectionEndSamp', {}, ...
            'analystDecision', {});
        return;
    end

    newFNwithAdj = newFN;
    for i = 1:numel(newFN)
        fname    = basenameOf(newFN(i).AudioFilename);
        bestDiff = inf;
        bestK    = 0;
        if ~isempty(fname) && fnGrouping.byName.isKey(fname) ...
                && isdatetime(newFN(i).DetectionStartTime) ...
                && isscalar(newFN(i).DetectionStartTime) ...
                && ~isnat(newFN(i).DetectionStartTime)
            candIdx = fnGrouping.byName(fname);
            ct = fnGrouping.startTime(candIdx);
            good = ~isnat(ct);
            if any(good)
                candIdxValid = candIdx(good);
                diffs = abs(seconds(newFN(i).DetectionStartTime - ct(good)));
                [bestDiff, localBest] = min(diffs);
                bestK = candIdxValid(localBest);
            end
        end
        if bestK > 0 && bestDiff <= detectionTolerance
            newFNwithAdj(i).analystDecision = fnGrouping.decision{bestK};
        else
            newFNwithAdj(i).analystDecision = 'Unknown';
        end
    end
end

function [FP_to_TP, FN_to_TN] = reclassifyWithMargin(disagreements, marginThreshold_dB)
% Margin-aware reclassification: FP becomes TP if analyst said
% DiscreteCallsPresent, OR if analyst said DiscreteCallsChorusPresent AND
% the dB margin meets the threshold. FN becomes TN only if analyst said
% CallChorusAbsent. 'Unknown' decisions are conservative -> remain FP/FN.

    FP = disagreements.falsePositives;
    FN = disagreements.falseNegatives;
    nFP = length(FP);
    nFN = length(FN);

    FP_to_TP = false(nFP, 1);
    for i = 1:nFP
        d = FP(i).analystDecision;
        if strcmp(d, 'DiscreteCallsPresent')
            FP_to_TP(i) = true;
        elseif strcmp(d, 'DiscreteCallsChorusPresent')
            m = getField(FP(i), 'discreteAboveChorus_dB', NaN);
            if isnumeric(m) && isscalar(m) && ~isnan(m) && (m >= marginThreshold_dB)
                FP_to_TP(i) = true;
            end
        end
        % All other decisions (ChorusPresent, CallChorusAbsent, Unknown) remain FP.
    end

    FN_to_TN = false(nFN, 1);
    for i = 1:nFN
        if strcmp(FN(i).analystDecision, 'CallChorusAbsent')
            FN_to_TN(i) = true;
        end
    end
end

function v = getField(s, name, default)
% Defensive struct field access with default value.
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default;
    end
end

function plotMarginHistogram(falsePositives, threshold_dB, savePath)
% Saves a histogram of the dB-margin distribution for
% DiscreteCallsChorusPresent FPs, split by estimate mode.

    decs = {falsePositives.analystDecision};
    isDPCh = strcmp(decs, 'DiscreteCallsChorusPresent');
    if ~any(isDPCh)
        return;
    end
    dBvals = [falsePositives(isDPCh).discreteAboveChorus_dB];
    modes  = {falsePositives(isDPCh).marginEstimateMode};
    valid  = ~isnan(dBvals);

    if ~any(valid), return; end

    fig = figure('Visible', 'off', 'Position', [100 100 800 480]);
    edges = linspace(min(dBvals(valid)) - 1, max(dBvals(valid)) + 1, 30);
    hold on;
    uniqueModes = unique(modes(valid));
    for k = 1:length(uniqueModes)
        m = uniqueModes{k};
        sel = strcmp(modes, m) & valid;
        histogram(dBvals(sel), edges, 'DisplayName', m, 'FaceAlpha', 0.6);
    end
    yl = ylim;
    line([threshold_dB threshold_dB], yl, 'Color', 'r', 'LineStyle', '--', ...
        'LineWidth', 1.5, 'DisplayName', sprintf('threshold = %.1f dB', threshold_dB));
    xlabel('Discrete-above-chorus margin (dB)');
    ylabel('Count');
    title(sprintf('Margin distribution for DiscreteCallsChorusPresent FPs (n=%d, valid=%d)', ...
        sum(isDPCh), sum(valid)));
    legend('Location', 'best', 'Interpreter', 'none');
    grid on;
    exportgraphics(fig, savePath, 'Resolution', 150);
    close(fig);
end

function plotOperatingPointScatter(T, chosen, maxF1, precisionFloor, decisionLogic, savePath)
% Scatter of recall_adj (x) vs precision_adj (y), coloured by AT, with the
% precision floor and selected operating points annotated. The axis labels
% and title include the actual decisionLogic so the figure is unambiguous
% across policies.

    fig = figure('Visible', 'off', 'Position', [100 100 900 640]);
    scatter(T.recall_adj, T.precision_adj, 50, T.AT, 'filled', ...
        'MarkerEdgeColor', 'k', 'MarkerEdgeAlpha', 0.4);
    hold on;
    cb = colorbar; ylabel(cb, 'Activation Threshold (AT)');
    yl = ylim;
    line(xlim, [precisionFloor precisionFloor], 'Color', 'r', 'LineStyle', '--', ...
        'LineWidth', 1.2, 'DisplayName', sprintf('precision floor = %.2f', precisionFloor));
    ylim(yl);

    plot(chosen.recall_adj, chosen.precision_adj, 'pentagram', ...
        'MarkerSize', 16, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', ...
        'DisplayName', 'Chosen (precision-floor)');
    plot(maxF1.recall_adj, maxF1.precision_adj, 'diamond', ...
        'MarkerSize', 14, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', ...
        'DisplayName', 'Max F1');

    txtChosen = sprintf('AT=%.3f DT=%.3f\nLT=%.2f MT=%.2fs', ...
        chosen.AT, chosen.DT, chosen.LT_scaler, chosen.MT_s);
    text(chosen.recall_adj, chosen.precision_adj, ['  ' txtChosen], ...
        'FontSize', 8, 'VerticalAlignment', 'top');

    policyTag = char(decisionLogic);
    xlabel(sprintf('Recall (adjusted, %s)', policyTag));
    ylabel(sprintf('Precision (adjusted, %s)', policyTag));
    title(sprintf('Post-processing operating points  (logic = %s)', policyTag));
    legend('Location', 'southwest');
    grid on;
    exportgraphics(fig, savePath, 'Resolution', 150);
    close(fig);
end
