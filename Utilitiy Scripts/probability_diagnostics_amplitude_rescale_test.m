%% 2-file amplitude-rescale test (auto-select hot/cold from raw results)
  % Hypothesis: 2019/2020 audio amplitude (peak ~5e-4, DC ~6e-5) is collapsing
  % mel-spectrogram features into a near-constant input, driving the GRU to
  % its no-call rest state (~0.128). Test by DC-removing + peak-normalising
  % one hot and one cold 2019 file and re-running gavdNetInference.

  clear; close all; clc;

  projectRoot = "C:\Users\z5439673\Git\GAVDNet";
  addpath(fullfile(projectRoot, "Functions"));

  % --- Paths ---
  modelPath      = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar Exclude Chorus\GAVDNet_trained_20-May-2026_12-16.mat";
  rawResultsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_DGS_Detections_2000_to_2025\-10 to 10 single exemplar exclude chorus\detector_raw_results_2019.mat";
  audioFolder    = "E:\Diego Garcia South 3Ch\2019"; % 2019 audio folder

  % --- Inference settings (mirror the config) ---
  featureFraming       = 'event-split';
  frameStandardization = 'true';
  minSilenceDuration   = 1;
  plotting             = false;

  % --- Auto-pick hot and cold files from the raw results ---
  fprintf('Loading raw results from:\n  %s\n', rawResultsPath);
  R = load(rawResultsPath, 'results');
  results = R.results;
  nFiles = numel(results);

  maxP = nan(nFiles,1);
  for i = 1:nFiles
      p = results(i).probabilities;
      if ~isempty(p)
          maxP(i) = max(p, [], 'omitnan');
      end
  end

  % Hot: highest max p (with audio present on disk)
  [~, sortIdxDesc] = sort(maxP, 'descend');
  hotIdx = NaN;
  hotCandidatesShown = {};
  nShown = 0;
  for k = 1:numel(sortIdxDesc)
      cand = sortIdxDesc(k);
      if isnan(maxP(cand)), continue, end
      candPath = fullfile(audioFolder, results(cand).fileName);
      if ~isfile(candPath), continue, end
      if isnan(hotIdx)
          hotIdx = cand;
      end
      nShown = nShown + 1;
      hotCandidatesShown{end+1,1} = sprintf('    [%d] %s   (max p = %.3f)', ...
          cand, results(cand).fileName, maxP(cand)); %#ok<SAGROW>
      if nShown >= 5, break, end
  end

  % Cold: a file with max p in the modal band [0.12, 0.15] (the ~0.128 cluster)
  coldBand = find(maxP >= 0.12 & maxP <= 0.15);
  if isempty(coldBand)
      coldBand = find(maxP < 0.20);
  end
  coldBand = coldBand(randperm(numel(coldBand)));
  coldIdx = NaN;
  coldCandidatesShown = {};
  nShownC = 0;
  for k = coldBand(:).'
      candPath = fullfile(audioFolder, results(k).fileName);
      if ~isfile(candPath), continue, end
      if isnan(coldIdx)
          coldIdx = k;
      end
      nShownC = nShownC + 1;
      coldCandidatesShown{end+1,1} = sprintf('    [%d] %s   (max p = %.3f)', ...
          k, results(k).fileName, maxP(k)); %#ok<SAGROW>
      if nShownC >= 5, break, end
  end

  if isnan(hotIdx) || isnan(coldIdx)
      error(['Could not auto-select hot and/or cold files. ' ...
             'Check audioFolder contains the .wav files referenced in raw results.']);
  end

  fprintf('\n=== Selected candidates ===\n');
  fprintf('Distribution over %d files: min=%.3f, median=%.3f, max=%.3f, n>=0.5=%d\n', ...
      sum(~isnan(maxP)), min(maxP,[],'omitnan'), median(maxP,'omitnan'), ...
      max(maxP,[],'omitnan'), sum(maxP>=0.5));

  fprintf('\nTop 5 HOT candidates (highest max p, audio present):\n');
  fprintf('%s\n', hotCandidatesShown{:});
  fprintf('  -> default HOT pick: index %d (%s)\n', hotIdx, results(hotIdx).fileName);

  fprintf('\nUp to 5 COLD candidates (max p in [0.12, 0.15], audio present, random order):\n');
  fprintf('%s\n', coldCandidatesShown{:});
  fprintf('  -> default COLD pick: index %d (%s)\n', coldIdx, results(coldIdx).fileName);

  % --- Confirmation ---
  reply = input('\nProceed with the defaults? y = yes / n = abort / m = manually enter different indices: ', 's');
  reply = lower(strtrim(reply));
  if strcmp(reply, 'n')
      fprintf('Aborted by user.\n');
      return
  elseif strcmp(reply, 'm')
      hStr = input('  Enter HOT index from raw-results: ', 's');
      cStr = input('  Enter COLD index from raw-results: ', 's');
      hotIdx = str2double(hStr);
      coldIdx = str2double(cStr);
      if isnan(hotIdx) || isnan(coldIdx) || hotIdx<1 || coldIdx<1 || hotIdx>nFiles || coldIdx>nFiles
          error('Invalid indices entered.');
      end
      if ~isfile(fullfile(audioFolder, results(hotIdx).fileName))
          error('Hot audio file not found: %s', results(hotIdx).fileName);
      end
      if ~isfile(fullfile(audioFolder, results(coldIdx).fileName))
          error('Cold audio file not found: %s', results(coldIdx).fileName);
      end
      fprintf('  HOT  : %s   (max p = %.3f)\n', results(hotIdx).fileName, maxP(hotIdx));
      fprintf('  COLD : %s   (max p = %.3f)\n', results(coldIdx).fileName, maxP(coldIdx));
  elseif ~strcmp(reply, 'y')
      fprintf('Unrecognised response. Aborted.\n');
      return
  end

  hotFile  = fullfile(audioFolder, results(hotIdx).fileName);
  coldFile = fullfile(audioFolder, results(coldIdx).fileName);

  % --- Load model + GPU config ---
  fprintf('\nLoading model and GPU config...\n');
  S = load(modelPath, 'model'); model = S.model;
  [~, ~, ~, bytesAvailable] = gpuConfig();

  % --- Run inference on all 4 variants ---
  fprintf('\nRunning inference (4 passes total). Expect ~10-15 minutes on GPU.\n\n');
  [probsH_raw, statsH_raw] = runVariant("HOT  raw     ", hotFile,  false, model, bytesAvailable, featureFraming, frameStandardization, minSilenceDuration, plotting);
  [probsH_nrm, statsH_nrm] = runVariant("HOT  rescaled", hotFile,  true,  model, bytesAvailable, featureFraming, frameStandardization, minSilenceDuration, plotting);
  [probsC_raw, statsC_raw] = runVariant("COLD raw     ", coldFile, false, model, bytesAvailable, featureFraming, frameStandardization, minSilenceDuration, plotting);
  [probsC_nrm, statsC_nrm] = runVariant("COLD rescaled", coldFile, true,  model, bytesAvailable, featureFraming, frameStandardization, minSilenceDuration, plotting);

  % --- Summary ---
  fprintf('\n=== Result summary ===\n');
  fprintf('  %-13s  %-10s  %-10s  %-10s  %-8s\n', 'variant', 'audio RMS', 'audio peak', 'max p', 'med p');
  fprintf('  %-13s  %-10.5f  %-10.4f  %-10.3f  %-8.3f\n', 'HOT  raw',      statsH_raw.rms, statsH_raw.peak, statsH_raw.maxP, statsH_raw.medP);
  fprintf('  %-13s  %-10.5f  %-10.4f  %-10.3f  %-8.3f\n', 'HOT  rescaled', statsH_nrm.rms, statsH_nrm.peak, statsH_nrm.maxP, statsH_nrm.medP);
  fprintf('  %-13s  %-10.5f  %-10.4f  %-10.3f  %-8.3f\n', 'COLD raw',      statsC_raw.rms, statsC_raw.peak, statsC_raw.maxP, statsC_raw.medP);
  fprintf('  %-13s  %-10.5f  %-10.4f  %-10.3f  %-8.3f\n', 'COLD rescaled', statsC_nrm.rms, statsC_nrm.peak, statsC_nrm.maxP, statsC_nrm.medP);

  % --- Plot all 4 probability traces ---
  figure('Color','w','Position',[100 100 1300 700]);
  tiledlayout(2,1)
  nexttile;
  plot(probsH_raw,'k-','DisplayName','raw'); hold on
  plot(probsH_nrm,'r-','DisplayName','rescaled');
  ylim([0 1]); grid on; ylabel('p(call)'); xlabel('Time bin')
  legend('Location','best')
  title(sprintf('HOT file: %s', results(hotIdx).fileName), 'Interpreter','none')
  nexttile;
  plot(probsC_raw,'k-','DisplayName','raw'); hold on
  plot(probsC_nrm,'r-','DisplayName','rescaled');
  ylim([0 1]); grid on; ylabel('p(call)'); xlabel('Time bin')
  legend('Location','best')
  title(sprintf('COLD file: %s', results(coldIdx).fileName), 'Interpreter','none')

  % ------------------- helpers --------------------
  function [probs, stats] = runVariant(label, fp, doNormalise, model, bytesAvailable, featureFraming, frameStandardization, minSilenceDuration, plotting)
      [a, fs] = audioread(fp);
      if doNormalise
          a = a - mean(a);
          a = a / max(abs(a));
      end
      fprintf('%s | RMS=%.5f peak=%.4f DC=%.6f ... ', label, rms(a), max(abs(a)), mean(a));
      t0 = tic;
      [probs, ~, ~, ~, ~] = gavdNetInference(a, fs, model, bytesAvailable, ...
          featureFraming, frameStandardization, minSilenceDuration, plotting);
      elapsed = toc(t0);
      stats.rms  = rms(a);
      stats.peak = max(abs(a));
      if isempty(probs)
          stats.maxP = NaN; stats.medP = NaN;
          fprintf('probs empty (%.1fs)\n', elapsed);
      else
          stats.maxP = max(probs,[],'omitnan');
          stats.medP = median(probs,'omitnan');
          fprintf('max p=%.3f, med p=%.3f (%.1fs)\n', stats.maxP, stats.medP, elapsed);
      end
  end