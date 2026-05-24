 % Compare new-model probability distributions: 2019 vs 2007
  % Plus audio-property and silence-mask comparison.
  % Set both paths to detector_raw_results.mat from the NEW model.

  rawPath2019 = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_DGS_Detections_2000_to_2025\-10 to 10 single exemplar exclude chorus\detector_raw_results_2020.mat";
  rawPath2007 = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet Data Backup\Chagos_DGS\Test Results\Final Test - 2007subset\Exclude Chorus\detector_raw_results.mat"; % adjust if different

  audioFolder2019 = "E:\Diego Garcia South 3Ch\2019"; % adjust to your 2019 audio folder
  audioFolder2007 = "E:\GAVDNet\Chagos_DGS\Test Data\2007subset";

  S19 = load(rawPath2019, 'results'); r19 = S19.results;
  S07 = load(rawPath2007, 'results'); r07 = S07.results;

  % --- Per-file max p comparison ---
  maxP19 = arrayfun(@(s) safeMax(s.probabilities), r19);
  maxP07 = arrayfun(@(s) safeMax(s.probabilities), r07);

  fprintf('\n=== 2019 vs 2007 (new model) ===\n');
  fprintf('Files: %d (2019) vs %d (2007)\n', numel(r19), numel(r07));
  for t = [0.30 0.50 0.70 0.90]
      fprintf('  p>=%.2f:  %d (%.1f%%) 2019   vs   %d (%.1f%%) 2007\n', ...
          t, sum(maxP19>=t), 100*mean(maxP19>=t), ...
             sum(maxP07>=t), 100*mean(maxP07>=t));
  end
  fprintf('Median max p: %.3f (2019) vs %.3f (2007)\n', median(maxP19,'omitnan'), median(maxP07,'omitnan'));

  % --- Silence-mask fraction per file ---
  silFrac19 = arrayfun(@(s) safeMean(s.silenceMask), r19);
  silFrac07 = arrayfun(@(s) safeMean(s.silenceMask), r07);
  fprintf('\nSilence-mask fraction per file:\n');
  fprintf('  2019 mean=%.3f  median=%.3f  90%%=%.3f\n', ...
      mean(silFrac19,'omitnan'), median(silFrac19,'omitnan'), prctile(silFrac19,90));
  fprintf('  2007 mean=%.3f  median=%.3f  90%%=%.3f\n', ...
      mean(silFrac07,'omitnan'), median(silFrac07,'omitnan'), prctile(silFrac07,90));

  % --- Audio properties spot-check (10 random files each) ---
  fprintf('\nAudio properties (10 random files per year):\n');
  checkAudio('2019', audioFolder2019, r19, 10);
  checkAudio('2007', audioFolder2007, r07,  10);

  % --- The 11 "successful" 2019 files: what are they? ---
  fprintf('\n2019 files with max p >= 0.50:\n');
  hot = find(maxP19 >= 0.50);
  for k = hot(:).'
      fprintf('  %s  (max p=%.3f, dur=%.0fs, silFrac=%.2f)\n', ...
          r19(k).fileName, maxP19(k), r19(k).fileDuration, silFrac19(k));
  end

  % --- Quick histogram comparison ---
  figure('Color','w');
  histogram(maxP07, 0:0.02:1, 'FaceColor',[0.3 0.7 0.3], 'FaceAlpha',0.5, 'DisplayName','2007'); hold on
  histogram(maxP19, 0:0.02:1, 'FaceColor',[0.8 0.3 0.3], 'FaceAlpha',0.5, 'DisplayName','2019');
  xlabel('Per-file max probability'); ylabel('Count'); legend; grid on
  title('New model: 2007 vs 2019 per-file max probability');

  % --- Local helpers ---
  function v = safeMax(p)
      if isempty(p), v = NaN; return, end
      v = max(p, [], 'omitnan');
  end
  function v = safeMean(m)
      if isempty(m), v = NaN; return, end
      v = mean(double(m), 'omitnan');
  end
  function checkAudio(label, folder, res, n)
      idx = randperm(numel(res), min(n, numel(res)));
      fprintf('  %s sample:\n', label);
      for i = idx
          fp = fullfile(folder, res(i).fileName);
          if ~isfile(fp), continue, end
          info = audioinfo(fp);
          [a,~] = audioread(fp);
          fprintf('    %s | Fs=%d | bits=%d | ch=%d | dur=%.0fs | RMS=%.4f | DC=%.5f | peak=%.3f\n', ...
              res(i).fileName, info.SampleRate, info.BitsPerSample, ...
              info.NumChannels, info.TotalSamples/info.SampleRate, ...
              rms(a(:,1)), mean(a(:,1)), max(abs(a(:,1))));
      end
  end
