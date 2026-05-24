% GAVDNet Post-Processing Parameter Sweep vs Adjudicated Ground Truth
%
% This script sweeps the four GAVDNet post-processing parameters
% (AT, DT, LT_scaler, MT) on cached raw detection probabilities (the
% neural network is NOT re-run) and scores every combination against an
% adjudication-enriched ground-truth. The adjudicated disagreements file
% comes from a previous model run; the existing analyst decisions are
% propagated onto the new model's detections via sample-domain IoU
% matching, so a fresh adjudication pass is not required.
%
% Motivation: the raw test-set ground truth is known to under-call discrete
% calls (which is why the analyst adjudication exists in the first place).
% Scoring solely against the raw GT therefore overstates the false-positive
% rate. Using the adjudicated decisions to reclassify the new model's FPs
% and FNs yields the most reliable performance estimate currently
% available, while acknowledging that the adjudication is not exhaustive
% (any new TP the analyst never saw will still be counted as FP via the
% 'Unknown' tag).
%
% This script is the adjudication-aware sibling of
% GAVDNet_SweepPostProcParamsAndCompareToGT.m. The raw-GT sibling sweeps
% only AT (or AT x LT_scaler) and reports unadjusted precision / recall.
%
% The TP definition is selectable at runtime via the USER INPUT
% `decisionLogic`. The four supported logics all delegate to the shared
% Functions/reclassifyDisagreementsByLogic.m:
%
%   'Inclusive'       - DiscreteCallsPresent OR ChorusPresent OR
%                       DiscreteCallsChorusPresent -> TP.
%   'Discrete-only'   - DiscreteCallsPresent OR DiscreteCallsChorusPresent
%                       -> TP. (DEFAULT — matches the user's TP definition:
%                       a discrete call, with or without chorus, counts.)
%   'Strict-discrete' - Only pure DiscreteCallsPresent -> TP. Chorus and
%                       Discrete+chorus both stay FP.
%   'Chorus-aware'    - Same TP definition as Discrete-only; adds chorus
%                       prevalence reporting.
%
% No dB-margin variant is offered: the user's spec is "use analyst
% decisions only, no arbitrary 3 dB margin".
%
% Anything not promoted to TP under the chosen logic remains a false
% positive. The 'Unknown' tag is reserved for new FPs whose
% sample-domain IoU against every original adjudicated FP on the same
% file is below originalFPMatch_IoUThreshold; Unknown FPs remain FP
% under every logic.
%
% The script:
%   1. Loads raw probabilities, adjudicated disagreements, and trained model.
%   2. Builds an in-RAM audio cache (required by gavdNetPostprocess even
%      when AEAVD=0).
%   3. Pre-builds basename->indices buckets for the original FPs/FNs so
%      inheritance is O(1) per new FP/FN.
%   4. Sweeps post-processing parameters over the 4D grid, re-running
%      only post-processing for each combination (cached raw probabilities
%      are reused). Outer parfor over combos.
%   5. For every combination, matches new detections against ground truth,
%      then inherits adjudicator decisions for the resulting false positives
%      via sample-domain IoU against the original adjudicated FP set.
%      Unmatched ("Unknown") FPs are conservatively counted as FPs.
%   6. Reports adjusted precision / recall / F1 and inheritance-category
%      breakdown per combination to an Excel file, plus a scatter figure
%      of recall vs precision, and auto-selects an operating point using
%      "max recall subject to precision >= precisionFloor" (aligns with
%      the user's priority: minimise FP first, then FN).
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

%% **** USER INPUT ****

% Replace these placeholder paths before running.

% --- Paths ---
trainedModelPath             = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar Exclude Chorus\GAVDNet_trained_20-May-2026_12-16.mat";
inferenceOutputPath          = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\Chagos_DGS\Test Results\Final Test - 2007subset\Exclude Chorus"; % must contain detector_raw_results.mat (run GAVDNet_Run_Detector.m first if absent)
audioSourceFolder            = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\Chagos_DGS\Test Data\2007subset"; % WAV files referenced by raw results + adjudicated FPs
groundtruthPath              = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\Chagos_DGS\Test Data\2007subset\test_dataset_detection_list.mat";                                 % or .txt for SORP
gtFormat                     = 'CTBTO';                                                                                                      % 'CTBTO' | 'SORP'
adjudicatedDisagreementsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\GAVDNet Data for Publication\Post-Adjudication Results\CPBW_DiegoGarciaSouth_2007_ADJUDICATED\detector_vs_GT_disagreements_07-Jul-2025_08-57-43.mat";
outputFolder                 = "C:\Users\z5439673\Git\GAVDNet\PostProc Tuning\ExcludeChorus";                                                % a per-run timestamped subfolder is created beneath this

% --- TP-reclassification policy ---
% One of four logics; all are dispatched to
% Functions/reclassifyDisagreementsByLogic.m.
%   'Inclusive'       : Discrete OR Chorus OR Discrete+Chorus -> TP
%   'Discrete-only'   : Discrete OR Discrete+Chorus -> TP   (DEFAULT)
%   'Strict-discrete' : Only Discrete -> TP
%   'Chorus-aware'    : same TPs as Discrete-only; adds chorus-prevalence stats
decisionLogic                = 'Discrete-only';

% --- Inheritance / matching ---
originalFPMatch_IoUThreshold = 0.1;   % new FP inherits an original adjudication
                                      % only if max sample-domain IoU on the
                                      % same file >= this. Lowered from 0.3 to
                                      % 0.1 in the prior tune-script run so
                                      % slightly time-shifted detections still
                                      % inherit (avoids over-counting Unknown
                                      % FPs).
detectionTolerance           = 30;    % seconds (Hungarian GT-match tolerance)
maxDetectionDuration         = 40;    % seconds (FN window length)

% --- 4D sweep grid (default: 6 x 4 x 4 x 3 = 288 combos) ---
% Coverage rationale:
%   AT in [0.30, 0.80]              - broader than the prior 108-combo grid,
%                                     which was anchored to the old model's
%                                     R-P winners. The chorus-rejecting new
%                                     model may have its elbow elsewhere.
%   DT_offsets in [0.001, 0.20]     - dense near tight hysteresis (the prior
%                                     Run-2 / Run-3 winners spanned this
%                                     range); DT = max(0, min(AT-offset,
%                                     AT-1e-3)).
%   LT_scaler in [0.10, 0.75]       - four levels; prior runs settled on 0.5
%                                     but the new model may need longer LT.
%   MT in [0.10, 0.50]              - prior runs found MT=0.1 universally
%                                     optimal; 0.3 and 0.5 included as a
%                                     safety net.
AT_sweep_values              = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80];
DT_offsets_from_AT           = [0.001, 0.05, 0.10, 0.20];
LTscaler_sweep_values        = [0.10, 0.25, 0.50, 0.75];
MT_sweep_values              = [0.10, 0.30, 0.50];
AEAVD                        = 0;     % fixed; AEAVD=1 is expensive

% --- Operating-point selector ---
% Optimise to minimise FP first (ie, achieve > precisionFloor), then maximise recall.
precisionFloor               = 0.95;

% --- Performance ---
useParallel                  = true;  % parfor over combos
audioCacheInRAM              = true;  % audioread every file once at startup
maxDetectionsPerCombo        = 30000; % combos producing more flat detections
                                      % than this are skipped before
                                      % compareDetectionsToSubsampledTestDataset
                                      % is called. The Hungarian matcher in
                                      % matchpairs allocates a square cost
                                      % matrix sized ~(nDet+nGT)^2 doubles
                                      % (~20 GB at 50000), so combos near the
                                      % low-AT end of the sweep can OOM
                                      % without this guard.

% --- Output options ---
saveFullDiagnostics          = true; % save per-combo augmented disagreement
                                      % struct (large; off by default)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.

%% Determinism

% Seed kept for reproducibility, though calculateAdjudicatedMetricsFast does
% not invoke any random-number generators.
rng(0, 'twister');

%% Setup paths

projectRoot = pwd;
addpath(fullfile(projectRoot, "Functions"))

%% Stage 1 - Pre-flight checks

validLogics = {'Inclusive', 'Discrete-only', 'Strict-discrete', 'Chorus-aware'};
if ~ismember(decisionLogic, validLogics)
    error('Unknown decisionLogic ''%s''. Valid options: %s.', ...
        decisionLogic, strjoin(validLogics, ', '));
end

rawDetectionsPath = fullfile(inferenceOutputPath, 'detector_raw_results.mat');
if ~exist(rawDetectionsPath, 'file')
    error(['Raw detection results file not found:\n  %s\n', ...
           'Run GAVDNet_Run_Detector.m on this model first to populate the ', ...
           'inference-output folder.'], rawDetectionsPath);
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
if isempty(dir(fullfile(audioSourceFolder, '*.wav')))
    error('Audio source folder contains no .wav files:\n  %s', audioSourceFolder);
end

%% Stage 2 - Output folder + diary

if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
ts = char(datetime("now", "Format", "uuuu-MM-dd_HH-mm-ss"));
runFolder = fullfile(outputFolder, ['ExcludeChorus_', ts]);
if ~isfolder(runFolder)
    mkdir(runFolder);
end

% Writability probe (delete immediately).
probePath = fullfile(runFolder, '.write_probe');
fid = fopen(probePath, 'w');
if fid < 0
    error('Output folder is not writable:\n  %s', runFolder);
end
fclose(fid);
delete(probePath);

diary(fullfile(runFolder, sprintf('sweep_log_%s.txt', ts)));

fprintf('=== GAVDNet_SweepPostProcParamsAndCompareToAdjudicatedGT ===\n');
fprintf('Started: %s\n', char(datetime("now")));
fprintf('Decision logic: %s\n', decisionLogic);
fprintf('Output folder:\n  %s\n\n', runFolder);

%% Stage 3 - Load inputs

% Raw detection results
fprintf('Loading raw detection results:\n  %s\n', rawDetectionsPath);
rawData    = load(rawDetectionsPath, 'results');
rawResults = rawData.results;
fprintf('  Loaded %d files of cached probabilities.\n', length(rawResults));

% Adjudicated disagreements
fprintf('Loading adjudicated disagreements:\n  %s\n', adjudicatedDisagreementsPath);
adjData       = load(adjudicatedDisagreementsPath, 'disagreements');
disagreements = adjData.disagreements;
nFP_adj = length(disagreements.falsePositives);
nFN_adj = length(disagreements.falseNegatives);
fprintf('  Loaded %d adjudicated FPs, %d adjudicated FNs.\n', nFP_adj, nFN_adj);

% Completeness assertion
nUnadj_FP = sum(cellfun(@isempty, {disagreements.falsePositives.analystDecision}));
nUnadj_FN = sum(cellfun(@isempty, {disagreements.falseNegatives.analystDecision}));
if nUnadj_FP > 0 || nUnadj_FN > 0
    error(['Incomplete adjudication: %d FPs and %d FNs have empty ', ...
           'analystDecision. All disagreements must be adjudicated before ', ...
           'running this analysis.'], nUnadj_FP, nUnadj_FN);
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

% featureFraming / frameStandardization are saved into the per-combo temp
% .mat file as metadata for compareDetectionsToSubsampledTestDataset; they
% are not used by the matching logic. Default to 'unknown' (matches the
% existing tune-script convention).
featureFraming       = 'unknown';
frameStandardization = 'unknown';

%% Stage 4 - Audio cache

% Combine the set of files we need audio for: every file present in
% rawResults AND every file referenced by an adjudicated FP.
filesFromRaw = arrayfun(@(r) basenameOf(r.fileName),         rawResults,                  'UniformOutput', false);
filesFromFP  = arrayfun(@(d) basenameOf(d.AudioFilename),    disagreements.falsePositives, 'UniformOutput', false);
uniqueFiles  = unique([filesFromRaw(:); filesFromFP(:)]);
uniqueFiles  = uniqueFiles(~cellfun(@isempty, uniqueFiles));
fprintf('\nNeed audio for %d unique files.\n', length(uniqueFiles));

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
            % gavdNetPostprocess requires a column vector
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
end
audioCacheNames = {audioCache.name};

%% Stage 5 - Pre-build adjudication-inheritance groupings

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

%% Stage 6 - 4D sweep over (AT, DT, LT_scaler, MT)

nAT  = length(AT_sweep_values);
nDT  = length(DT_offsets_from_AT);
nLT  = length(LTscaler_sweep_values);
nMT  = length(MT_sweep_values);
combosTotal = nAT * nDT * nLT * nMT;
fprintf('\nStarting %d-combination sweep (%d AT x %d DT x %d LT x %d MT)\n', ...
    combosTotal, nAT, nDT, nLT, nMT);

% Result buffer (will become the output table)
sweepRows = cell(combosTotal, 1);

% Diagnostic per-combo augmented disagreements (only kept on disk if
% saveFullDiagnostics; otherwise discarded between iterations)
diagFolder = '';
if saveFullDiagnostics
    diagFolder = fullfile(runFolder, 'sweep_disagreements'); %#ok<UNRCH>
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
% broadcast to workers as a small frozen value.
inv = struct( ...
    'postProcSkel',                 postProcSkel, ...
    'meanTargetCallDuration',       meanTargetCallDuration, ...
    'groundtruthPath',              groundtruthPath, ...
    'detectionTolerance',           detectionTolerance, ...
    'maxDetectionDuration',         maxDetectionDuration, ...
    'gtFormat',                     gtFormat, ...
    'featureFraming',               featureFraming, ...
    'frameStandardization',         frameStandardization, ...
    'originalFPMatch_IoUThreshold', originalFPMatch_IoUThreshold, ...
    'decisionLogic',                decisionLogic, ...
    'AEAVD',                        AEAVD, ...
    'precisionFloor',               precisionFloor, ...
    'outputFolder',                 char(runFolder), ...
    'saveFullDiagnostics',          saveFullDiagnostics, ...
    'diagFolder',                   diagFolder, ...
    'maxDetectionsPerCombo',        maxDetectionsPerCombo);

% Unroll the 4D sweep grid to a flat 1D combo index so we can run a single
% parfor over combosTotal. Each component is a column vector that parfor
% recognises as a sliced input.
[iAT_g, iDT_g, iLT_g, iMT_g] = ndgrid(1:nAT, 1:nDT, 1:nLT, 1:nMT);
comboAT     = AT_sweep_values(iAT_g(:));
comboDTbase = DT_offsets_from_AT(iDT_g(:));
comboLT     = LTscaler_sweep_values(iLT_g(:));
comboMT     = MT_sweep_values(iMT_g(:));

% Wrap all per-combo-invariant inputs in parallel.pool.Constant so they are
% broadcast to each worker ONCE for the whole pool session. These handles
% are only referenced inside the `if useParallel` parfor block below; the
% serial path uses the raw workspace values directly.
if useParallel
    % Match the existing tune-script worker count: 4 workers triggered video
    % memory BSODs on the 64 GB host; 2 is the memory-safe ceiling.
    nWorkersWanted = 2;
    if ~isempty(gcp('nocreate'))
        delete(gcp('nocreate'));
    end
    cluster = parcluster('Processes');
    if cluster.NumWorkers < nWorkersWanted
        cluster.NumWorkers = nWorkersWanted;
    end
    parpool(cluster, nWorkersWanted);
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
progressQueue  = parallel.pool.DataQueue;
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
    for c = 1:combosTotal %#ok<UNRCH>
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

% Cleanup of the per-combo temp comparison files
tempPattern = fullfile(runFolder, 'temp_combo_*.mat');
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

%% Stage 7 - Operating-point picker + outputs

% --- Operating-point picker -----------------------------------------------
% Aligns with user priority: precision_adj >= floor first (minimise FP),
% then maximise recall_adj (minimise FN), then minimise nUnknownFPs as a
% tie-breaker for robustness.
filtered = sweepTable(sweepTable.precision_adj >= precisionFloor, :);
if ~isempty(filtered)
    [~, idxSort] = sortrows(filtered, ...
        {'recall_adj', 'f1_adj', 'nUnknownFPs'}, {'descend', 'descend', 'ascend'});
    chosenRow = filtered(idxSort(1), :);
    metPrecisionFloor = true;
    fprintf('\nChosen operating point (precision_adj >= %.3f, max recall_adj):\n', precisionFloor);
else
    [~, idxSort] = sortrows(sweepTable, ...
        {'precision_adj', 'recall_adj'}, {'descend', 'descend'});
    chosenRow = sweepTable(idxSort(1), :);
    metPrecisionFloor = false;
    warning('sweepAdjGT:noComboMetFloor', ...
        'No combo met precision floor %.3f; returning highest-precision combo.', precisionFloor);
    fprintf('\nFallback operating point (highest precision_adj; no combo met %.3f):\n', precisionFloor);
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

% --- Write Excel ----------------------------------------------------------
excelPath = fullfile(runFolder, sprintf('sweep_results_%s.xlsx', ts));
writetable(sweepTable, excelPath, 'Sheet', 'Sweep');

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
figPath = fullfile(runFolder, sprintf('operating_point_%s.png', ts));
plotOperatingPointScatter(sweepTable, chosenRow, maxF1Row, precisionFloor, decisionLogic, figPath);

% --- Sweep summary .mat ---------------------------------------------------
summaryPath = fullfile(runFolder, sprintf('sweep_summary_%s.mat', ts));
chosenPostProcOptions = struct( ...
    'AT',                    chosenRow.AT, ...
    'DT',                    chosenRow.DT, ...
    'AEAVD',                 chosenRow.AEAVD, ...
    'MT',                    chosenRow.MT_s, ...
    'LT',                    chosenRow.LT_s, ...
    'LT_scaler',             chosenRow.LT_scaler, ...
    'maxTargetCallDuration', maxTargetCallDuration * 1.2);
save(summaryPath, 'sweepTable', 'chosenRow', 'maxF1Row', ...
    'chosenPostProcOptions', 'metPrecisionFloor', ...
    'precisionFloor', 'decisionLogic', 'originalFPMatch_IoUThreshold', ...
    'AT_sweep_values', 'DT_offsets_from_AT', 'LTscaler_sweep_values', ...
    'MT_sweep_values', '-v7.3');
fprintf('Sweep summary saved: %s\n', summaryPath);

% --- Drop-in chosen postProcOptions .mat ---------------------------------
% Saved as a single-variable file so downstream callers (e.g.
% GAVDNet_Run_Detector.m) can `load(...)` it without unpacking.
chosenOptionsPath = fullfile(runFolder, sprintf('chosen_postProcOptions_%s.mat', ts));
postProcOptions = chosenPostProcOptions;
save(chosenOptionsPath, 'postProcOptions', '-v7.3');
fprintf('Chosen postProcOptions saved: %s\n', chosenOptionsPath);

%% Stage 8 - Verification readout

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
    metPrecisionFloor precisionFloor runFolder ...
    excelPath summaryPath figPath chosenOptionsPath
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

    row = [];

    currentPostProcOptions = inv.postProcSkel;
    currentPostProcOptions.AT        = currentAT;
    currentPostProcOptions.DT        = currentDT;
    currentPostProcOptions.MT        = currentMT;
    currentPostProcOptions.LT_scaler = currentLT;
    currentPostProcOptions.LT        = inv.meanTargetCallDuration * currentLT;

    % --- Step 1: Per-file postprocess (serial inside parfor) -------------
    currentResults = rawResults;
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

    % --- Step 2: Flatten detections to per-detection rows ----------------
    flatDetections = flattenDetections(currentResults, preprocParams);
    if isempty(fieldnames(flatDetections))
        flatDetections = struct([]);
    end
    nPositivesDetector_thisCombo = length(flatDetections);

    % Predicted-OOM short-circuit: compareDetectionsToSubsampledTestDataset
    % calls matchpairs, which allocates a ~(nDet+nGT)^2 dense double cost
    % matrix.
    if nPositivesDetector_thisCombo > inv.maxDetectionsPerCombo
        warning(['Combo %d: %d detections exceeds maxDetectionsPerCombo (%d) -- ', ...
                 'skipping to avoid OOM in matchpairs.'], ...
                 c, nPositivesDetector_thisCombo, inv.maxDetectionsPerCombo);
        return;
    end

    % --- Step 3: Compare to GT (writes a per-combo temp .mat) -----------
    % Per-combo unique filename so parallel workers do not collide. -v6 is
    % uncompressed and noticeably faster than -v7.3 for this small ephemeral
    % file (~1 MB); it also avoids the HDF5 init overhead.
    tempResultsPath = fullfile(inv.outputFolder, sprintf('temp_combo_%05d.mat', c));
    results = flatDetections;
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

    % --- Step 4: Inherit adjudication for new FPs/FNs --------------------
    comboDisagreements = struct( ...
        'falsePositives', inheritAdjudication(newFP, fpGrouping, inv.originalFPMatch_IoUThreshold), ...
        'falseNegatives', inheritFNAdjudication(newFN, fnGrouping, inv.detectionTolerance));

    % --- Step 5: Reclassify FP/FN per the configured decision logic -----
    % All four logics delegate to the shared function. Margin-aware
    % reclassification is intentionally not supported in this script.
    [~, FP_to_TP, FN_to_TN] = reclassifyDisagreementsByLogic( ...
        comboDisagreements, inv.decisionLogic);

    % --- Step 6: Calculate adjusted metrics (fast local version) --------
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

    % --- Step 7: Per-combo inheritance-category diagnostics --------------
    % Pure counts of each analyst-decision category among the new combo's
    % FPs, plus the count that became TP under the active logic. Naming is
    % logic-neutral; interpret via the decisionLogic in the summary file.
    %
    % The original adjudicated FPs store analystDecision as a MATLAB string
    % scalar; inheritAdjudication writes 'Unknown' as a char. The mixed
    % cell that results would silently return 0 for every strcmp against a
    % char literal. string() unwraps both into a string array so == works
    % element-wise.
    decs = string({comboDisagreements.falsePositives.analystDecision});
    nUnknownFPs                  = sum(decs == "Unknown");
    nDiscreteInherited           = sum(decs == "DiscreteCallsPresent");
    nChorusInherited             = sum(decs == "ChorusPresent");
    nDiscretePlusChorusInherited = sum(decs == "DiscreteCallsChorusPresent");
    nNoCallInherited             = sum(decs == "CallChorusAbsent");
    nInheritedFPs_to_TP          = sum(FP_to_TP);

    % --- Step 8: Build the result row ------------------------------------
    row = table( ...
        c, currentAT, currentDT, currentMT, currentLT, ...
        inv.meanTargetCallDuration * currentLT, inv.AEAVD, ...
        inv.precisionFloor, ...
        metrics_adj.precision, metrics_adj.recall, metrics_adj.f1Score, ...
        metrics_adj.auc, ...
        metrics_adj.nTruePositives, metrics_adj.nFalsePositives, ...
        metrics_adj.nFalseNegatives, metrics_adj.nTrueNegatives, ...
        nUnknownFPs, nDiscreteInherited, nChorusInherited, ...
        nDiscretePlusChorusInherited, nNoCallInherited, nInheritedFPs_to_TP, ...
        metrics_orig.precision, metrics_orig.recall, metrics_orig.f1Score, ...
        metrics_orig.nFalsePositives, metrics_orig.nFalseNegatives, ...
        nPositivesDetector_thisCombo, ...
        'VariableNames', { ...
        'ComboIdx', 'AT', 'DT', 'MT_s', 'LT_scaler', 'LT_s', 'AEAVD', ...
        'PrecisionFloor', ...
        'precision_adj', 'recall_adj', 'f1_adj', 'auc_adj', ...
        'nTP_adj', 'nFP_adj', 'nFN_adj', 'nTN_adj', ...
        'nUnknownFPs', 'nDiscreteInherited', 'nChorusInherited', ...
        'nDiscretePlusChorusInherited', 'nNoCallInherited', 'nInheritedFPs_to_TP', ...
        'precision_orig', 'recall_orig', 'f1_orig', ...
        'nFP_orig', 'nFN_orig', ...
        'nDetections_thisCombo'});

    % --- Step 9: Optional per-combo diagnostics --------------------------
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

    keyToIdx = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    for j = 1:nDet
        if detKeys(j) ~= ""
            k = char(detKeys(j));
            if ~keyToIdx.isKey(k)
                keyToIdx(k) = int32(j);
            end
        end
    end

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
% Pre-compute a basename->indices bucket plus packed numeric/cell columns for
% the original (invariant) adjudicated false-positive list. Used by
% inheritAdjudication to do O(1) bucket lookup + vectorised IoU scoring.

    n = numel(origFP);
    fpGrouping.startSamp = NaN(n, 1);
    fpGrouping.endSamp   = NaN(n, 1);
    fpGrouping.decision  = cell(n, 1);
    fpGrouping.discreteAboveChorus_dB = NaN(n, 1);
    fpGrouping.marginEstimateMode     = cell(n, 1);

    hasMarginField = isfield(origFP, 'discreteAboveChorus_dB');
    hasModeField   = isfield(origFP, 'marginEstimateMode');

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
% original adjudicated false-negative list.

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
% iouThreshold, inherit analystDecision (and any cached margin fields).
% Otherwise tag analystDecision = 'Unknown'.

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
