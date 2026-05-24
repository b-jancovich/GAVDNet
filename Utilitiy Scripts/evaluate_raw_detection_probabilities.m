  % Quick diagnostic of per-file max probability distribution from
  % detector_raw_results.mat. Compare against old-model 2019 baseline:
  %   Files with max p >= 0.70: 2 / 2197
  %   Files with max p >= 0.50: 7 / 2197
  %   Percentiles: 50%=0.140  90%=0.168  99%=0.218

  rawPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_DGS_Detections_2000_to_2025\-10 to 10 single exemplar exclude chorus\detector_raw_results_2020.mat";

  S = load(rawPath, 'results');
  results = S.results;
  nFiles = numel(results);

  % Per-file max probability (NaN where file failed / no probs)
  maxP = nan(nFiles, 1);
  hasValidProbs = false(nFiles, 1);
  isAllNaN = false(nFiles, 1);
  isAllSilence = false(nFiles, 1);
  failedFiles = false(nFiles, 1);

  hasFailField = isfield(results, 'failComment');

  for i = 1:nFiles
      p = results(i).probabilities;

      if hasFailField
          failedFiles(i) = ~isempty(results(i).failComment) || isempty(p);
      else
          failedFiles(i) = isempty(p);
      end

      if isempty(p)
          continue
      end
      isAllNaN(i) = all(isnan(p));
      if isfield(results, 'audioAllSilence')
          isAllSilence(i) = isequal(results(i).audioAllSilence, true);
      end
      if isAllNaN(i)
          continue
      end
      maxP(i) = max(p, [], 'omitnan');
      hasValidProbs(i) = true;
  end

  mp = maxP(hasValidProbs);

  % Threshold counts
  thr = [0.30 0.40 0.50 0.60 0.70 0.80 0.90];
  fprintf('\n=== Raw detector probability diagnostic ===\n');
  fprintf('File: %s\n', rawPath);
  fprintf('Total files in results struct: %d\n', nFiles);
  fprintf('  Files with valid probabilities: %d\n', sum(hasValidProbs));
  fprintf('  Files marked failed/skipped:    %d\n', sum(failedFiles));
  fprintf('  Files with all-NaN probs:       %d\n', sum(isAllNaN));
  fprintf('  Files with all-silence audio:   %d\n\n', sum(isAllSilence));

  fprintf('Per-file max probability (over %d files with valid probs):\n', numel(mp));
  for t = thr
      n = sum(mp >= t);
      fprintf('  Files with max p >= %.2f: %4d / %d  (%.2f%%)\n', ...
          t, n, numel(mp), 100*n/numel(mp));
  end

  pct = prctile(mp, [10 25 50 75 90 95 99]);
  fprintf('\nPercentiles of per-file max p:\n');
  fprintf('  10%%=%.3f  25%%=%.3f  50%%=%.3f  75%%=%.3f  90%%=%.3f  95%%=%.3f  99%%=%.3f\n', ...
      pct(1), pct(2), pct(3), pct(4), pct(5), pct(6), pct(7));
  fprintf('  min=%.3f  mean=%.3f  max=%.3f  std=%.3f\n\n', ...
      min(mp), mean(mp), max(mp), std(mp));

  % Quick histogram
  figure('Name','2019 per-file max probability', 'Color','w');
  histogram(mp, 0:0.02:1, 'FaceColor', [0.2 0.5 0.8]);
  hold on
  yl = ylim;
  plot([0.70 0.70], yl, 'r--', 'LineWidth', 1.5);
  plot([0.50 0.50], yl, 'Color', [0.6 0.4 0.1], 'LineStyle','--', 'LineWidth', 1.5);
  text(0.71, yl(2)*0.95, sprintf('AT=0.70 (n=%d)', sum(mp >= 0.70)), 'Color', 'r');
  text(0.51, yl(2)*0.85, sprintf('p=0.50 (n=%d)', sum(mp >= 0.50)), 'Color', [0.6 0.4 0.1]);
  xlabel('Per-file max probability');
  ylabel('Number of files');
  title(sprintf('Per-file max probability — %d files', numel(mp)));
  grid on
  hold off