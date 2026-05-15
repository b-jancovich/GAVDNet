% Recover failed-read entries in detector_raw_results_<year>.mat
%
% A long inference run over external-drive storage can leave entries in the
% raw results .mat where audio could not be read at the time (transient
% drive blips, OneDrive sync races, antivirus locks, etc.). Such entries
% have empty fileName and/or empty probabilities. This script identifies
% those entries, retries inference for just those files (with the same
% audioReadWithRetry helper used in the main pipeline), and patches the
% existing raw .mat in place. The valid entries are not touched.
%
% Run from the GAVDNet project root.
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

plotting = false;

% Path to the config file (must match the original inference run):
configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos.m";

% Trained model data path (must match the original inference run):
gavdNetDataPath = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar";

% Root folder containing per-year subfolders of audio:
audioRoot = "E:\Diego Garcia South 3Ch";

% Channel prefix used by the original inference run:
channelPrefix = "H08S1";

% Output path containing the detector_raw_results_<year>.mat file(s) to patch:
inferenceOutputPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_DGS_Detections_2000_to_2025";

% Year(s) to recover. Each year's raw .mat is patched in place.
yearsToRecover = 2019;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO MORE USER TUNABLE PARAMETERS. DO NOT MODIFY THE CODE BELOW THIS POINT.

%% One-time setup (model + GPU loaded ONCE, outside the recovery loop)

% Snapshot user inputs (config will overwrite some)
userGavdNetDataPath     = gavdNetDataPath;
userInferenceOutputPath = inferenceOutputPath;

% Add dependencies to path & load config. This script lives in
% <projectRoot>/Utilitiy Scripts/, not at the project root, so derive
% projectRoot from the script's own location instead of pwd. That makes
% the script work regardless of how it was launched (Run button, F5,
% command-line invocation from any directory).
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);
[projectRoot, ~, ~] = fileparts(scriptDir);
addpath(fullfile(projectRoot, "Functions"))
run(configPath)

% Re-apply user inputs (config clobbered them)
gavdNetDataPath     = userGavdNetDataPath;
inferenceOutputPath = userInferenceOutputPath;
clear userGavdNetDataPath userInferenceOutputPath

fprintf('\tGAVDNet model data path: %s\n', gavdNetDataPath)
fprintf('\tAudio root: %s\n', audioRoot)
fprintf('\tChannel prefix: %s\n', channelPrefix)
fprintf('\tInference output path: %s\n', inferenceOutputPath)

% Load the trained model
modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
if isscalar(modelList)
    load(fullfile(modelList.folder, modelList.name))
    fprintf('Loading model: %s\n', modelList.name)
else
    [file, location] = uigetfile(gavdNetDataPath, 'Select a model to load:');
    load(fullfile(location, file))
end

% LT and maxTargetCallDuration (kept for parity with main script context;
% recovery itself does not run postprocessing).
postProcOptions.LT = model.dataSynthesisParams.meanTargetCallDuration .* postProcOptions.LT_scaler;
postProcOptions.maxTargetCallDuration = model.dataSynthesisParams.maxTargetCallDuration * 1.2;

% GPU config (one device for the whole recovery run)
[useGPU, gpuDeviceID, ~, bytesAvailable] = gpuConfig();

%% Per-year recovery loop

for yearIdx = 1:numel(yearsToRecover)
    year = yearsToRecover(yearIdx);
    yearDir = fullfile(audioRoot, num2str(year));
    saveNamePathRaw = fullfile(inferenceOutputPath, ...
        sprintf('detector_raw_results_%d.mat', year));

    fprintf('=== Year %d ===\n', year)

    if ~exist(saveNamePathRaw, 'file')
        warning('No raw results file at %s. Skipping year %d.', saveNamePathRaw, year)
        continue
    end

    %% Load existing raw results
    fprintf('Loading existing raw results from %s\n', saveNamePathRaw)
    raw = load(saveNamePathRaw, 'results');
    results = raw.results;
    clear raw

    %% Identify broken entries
    brokenIdx = find(arrayfun(@(r) isempty(r.fileName) || isempty(r.probabilities), results));
    fprintf('Found %d broken entries (out of %d total).\n', numel(brokenIdx), numel(results))

    if isempty(brokenIdx)
        fprintf('Nothing to recover for year %d.\n', year)
        continue
    end

    %% Reconstruct the original file ordering
    % Original inference iterated chFilePaths in dir() order; results(i)
    % therefore corresponds to chFiles(i).
    chFiles = dir(fullfile(yearDir, sprintf('%s_*.wav', channelPrefix)));
    if numel(chFiles) ~= numel(results)
        error(['File count mismatch for year %d: %d files on disk vs %d ' ...
            'entries in results. The audio folder contents may have ' ...
            'changed since the original inference run; cannot map broken ' ...
            'indices safely.'], year, numel(chFiles), numel(results))
    end
    chFilePaths = fullfile({chFiles.folder}, {chFiles.name});

    % Sanity check: a known-good entry must match the file list at the same index
    sampleGoodIdx = find(arrayfun(@(r) ~isempty(r.fileName), results), 1, 'first');
    if ~isempty(sampleGoodIdx) && ...
            ~strcmp(results(sampleGoodIdx).fileName, chFiles(sampleGoodIdx).name)
        error(['File ordering check failed for year %d: results(%d).fileName ' ...
            'is "%s" but dir() position %d is "%s". The file list may ' ...
            'have changed since the original inference run.'], ...
            year, sampleGoodIdx, results(sampleGoodIdx).fileName, ...
            sampleGoodIdx, chFiles(sampleGoodIdx).name)
    end

    %% Retry each broken entry
    nRecovered = 0;
    nStillBroken = 0;

    for k = 1:numel(brokenIdx)
        fileIdx = brokenIdx(k);
        filePath = chFilePaths{fileIdx};
        [~, fName, fExt] = fileparts(filePath);

        fprintf('[%d/%d] Retrying entry %d: %s%s\n', ...
            k, numel(brokenIdx), fileIdx, fName, fExt)

        % Refresh GPU memory snapshot
        if useGPU
            wait(gpuDevice(gpuDeviceID));
            reset(gpuDevice(gpuDeviceID));
            bytesAvailable = gpuDevice(gpuDeviceID).AvailableMemory;
        end

        % Read audio with retry (60/180/300 s waits, then 300 s cooldown)
        [audioIn, sampleRate, readErr] = audioReadWithRetry(filePath);

        % Always set fileName so the entry is identifiable downstream
        results(fileIdx).fileName = [fName, fExt];

        if isempty(audioIn)
            warning('\tCould not recover %s%s: %s', fName, fExt, readErr)
            results(fileIdx).failComment = sprintf( ...
                'Read failed after retries: %s', readErr);
            nStillBroken = nStillBroken + 1;
            continue
        end

        % Validate audio
        if ~isValidAudio(audioIn)
            warning('\tFile %s%s does not contain valid audio.', fName, fExt)
            results(fileIdx).failComment = 'Could not read valid audio from file';
            nStillBroken = nStillBroken + 1;
            continue
        end

        % Parse start datetime
        fileStartDateTime = extractDatetimeFromFilename(filePath, 'datetime');
        if isempty(fileStartDateTime) || isnat(fileStartDateTime)
            warning('\tCould not parse start datetime from %s%s.', fName, fExt)
            results(fileIdx).failComment = 'Could not read valid recording start data-time from filename';
            nStillBroken = nStillBroken + 1;
            continue
        end

        % Populate scalar fields (mirror main inference loop's schema)
        results(fileIdx).fileFs = sampleRate;
        results(fileIdx).fileSamps = numel(audioIn);
        results(fileIdx).fileDuration = numel(audioIn) / sampleRate;
        results(fileIdx).fileStartDateTime = fileStartDateTime;
        results(fileIdx).failComment = '';   % clear any prior failure note

        % Run preprocessing and inference
        [results(fileIdx).probabilities, ~, execTime, ...
            results(fileIdx).silenceMask, numAudioSegments] = gavdNetInference(...
            audioIn, sampleRate, model, bytesAvailable, ...
            featureFraming, frameStandardization, minSilenceDuration, plotting);

        % Report execution time and bins above activation threshold
        numTimeBinsProbHigh = sum(results(fileIdx).probabilities > postProcOptions.AT);
        secondsBinsProbHigh = numTimeBinsProbHigh * windowDur;
        fprintf('\tInference completed in %.2f s. Duration > AT: %.2f s.\n', ...
            execTime, secondsBinsProbHigh)

        % Diagnostic flags (mirror main script's schema)
        results(fileIdx).probsAllNan = all(isnan(results(fileIdx).probabilities));
        results(fileIdx).probsAnyNan = any(isnan(results(fileIdx).probabilities));
        results(fileIdx).audioAllSilence = all(results(fileIdx).silenceMask == true);
        results(fileIdx).audioAnySilence = any(results(fileIdx).silenceMask == true);
        results(fileIdx).numSplitAudioSegments = numAudioSegments;

        nRecovered = nRecovered + 1;
    end

    %% Save back, replacing the original raw .mat
    fprintf('Recovered %d of %d entries (%d still broken). Saving...\n', ...
        nRecovered, numel(brokenIdx), nStillBroken)
    save(saveNamePathRaw, 'results', '-v7.3')
    fprintf('Saved updated raw results to %s\n', saveNamePathRaw)
end

fprintf('Recovery run complete.\n')
