% Compare detection counts between the legacy detector (2002-2018, CTBTO
% 8-column format from Leroy et al. 2021) and the new GAVDNet detector
% (2019-2025, flattened postprocessed results) on the Diego Garcia H08S1
% hydrophone dataset.
%
% Builds a year-by-month detection-count table for each detector, prints
% per-year totals side-by-side, and writes both tables to CSV next to this
% script.
%
% Ben Jancovich, 2026
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

clear; clc;

%% Paths and year ranges
newResultsDir = ['C:\Users\z5439673\OneDrive - UNSW\H0419778\' ...
    'GAVDNet_DGS_Detections_2000_to_2025\-10 to 10 single exemplar exclude chorus'];
oldResultsDir = 'C:\Users\z5439673\OneDrive - UNSW\H0419778\Manue_Chagos_RawData\DGS';

newYears = 2019:2025;
oldYears = 2002:2018;

outputDir = fileparts(mfilename('fullpath'));

%% Tables: rows = years, columns = months 1..12
newCounts = zeros(numel(newYears), 12);
oldCounts = zeros(numel(oldYears), 12);

%% New detector (GAVDNet): one struct entry per detection with eventStartTime
fprintf('--- New detector (GAVDNet, postprocessed) ---\n');
for k = 1:numel(newYears)
    yr = newYears(k);
    matPath = fullfile(newResultsDir, ...
        sprintf('detector_results_postprocessed_%d.mat', yr));

    if ~isfile(matPath)
        warning('Missing new-detector file for %d: %s', yr, matPath);
        continue
    end

    finfo = dir(matPath);
    fprintf('Year %d: loading %.1f MB ...\n', yr, finfo.bytes / 1e6);
    tLoad = tic;
    S = load(matPath, 'results');
    fprintf('Year %d: load took %.1f s\n', yr, toc(tLoad));

    if ~isfield(S, 'results') || isempty(S.results)
        fprintf('Year %d: 0 detections (empty results)\n', yr);
        continue
    end

    % flattenDetections output: one entry per detection. Vectorised field
    % concatenation - per-element struct indexing on large arrays is
    % effectively O(N^2), which is what was making earlier runs hang.
    % [S.results.eventStartTime] auto-drops empty entries.
    tExtract = tic;
    eventTimes = [S.results.eventStartTime].';
    fprintf('Year %d: extracted %d eventStartTime values in %.1f s\n', ...
        yr, numel(eventTimes), toc(tExtract));

    % Keep only detections that actually fall in the expected year
    yMask = year(eventTimes) == yr;
    if any(~yMask)
        fprintf('Year %d: %d detections fell outside the year and were excluded\n', ...
            yr, sum(~yMask));
    end
    eventTimes = eventTimes(yMask);

    months = month(eventTimes);
    for m = 1:12
        newCounts(k, m) = sum(months == m);
    end

    fprintf('Year %d: %d detections\n\n', yr, sum(newCounts(k, :)));

    % Free the big struct before next iteration (probability subsequences
    % per detection can dominate memory).
    clear S eventTimes
end

%% Old detector (CTBTO 8-column): col 1 = year, col 3 = month
fprintf('\n--- Old detector (CTBTO 8-column format) ---\n');
for k = 1:numel(oldYears)
    yr = oldYears(k);

    % Prefer cleaned version if present (only 2015 has one at time of writing)
    candidates = {
        fullfile(oldResultsDir, sprintf('detections_H08S1_DiegoGarciaS_%d_cleaned.mat', yr))
        fullfile(oldResultsDir, sprintf('detections_H08S1_DiegoGarciaS_%d.mat', yr))
    };
    matPath = '';
    for c = 1:numel(candidates)
        if isfile(candidates{c})
            matPath = candidates{c};
            break
        end
    end
    if isempty(matPath)
        warning('Missing old-detector file for %d', yr);
        continue
    end

    S = load(matPath, 'detections');
    if ~isfield(S, 'detections') || isempty(S.detections)
        fprintf('Year %d: 0 detections (empty detections)\n', yr);
        continue
    end

    det = S.detections;
    yrCol    = det(:, 1);
    monthCol = det(:, 3);

    % Sanity-check: any rows where col 1 disagrees with the filename year?
    inYearMask = (yrCol == yr);
    if any(~inYearMask)
        fprintf('Year %d: %d rows had col-1 year mismatching filename, excluded\n', ...
            yr, sum(~inYearMask));
    end
    monthCol = monthCol(inYearMask);

    for m = 1:12
        oldCounts(k, m) = sum(monthCol == m);
    end

    if ~isempty(strfind(matPath, '_cleaned')) %#ok<STREMP>
        tag = ' (cleaned)';
    else
        tag = '';
    end
    fprintf('Year %d%s: %d detections\n', yr, tag, sum(oldCounts(k, :)));
end

%% Assemble pretty tables
monthNames = {'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'};

newTable = array2table([newCounts, sum(newCounts, 2)], ...
    'VariableNames', [monthNames, {'Total'}], ...
    'RowNames', cellstr(string(newYears(:))));
oldTable = array2table([oldCounts, sum(oldCounts, 2)], ...
    'VariableNames', [monthNames, {'Total'}], ...
    'RowNames', cellstr(string(oldYears(:))));

%% Print summary
fprintf('\n========== OLD DETECTOR (2002-2018) ==========\n');
disp(oldTable);

fprintf('\n========== NEW DETECTOR (2019-2025) ==========\n');
disp(newTable);

fprintf('\n========== YEARLY TOTALS COMPARISON ==========\n');
oldTotal = sum(oldCounts(:));
newTotal = sum(newCounts(:));
nOldYears = sum(any(oldCounts > 0, 2));
nNewYears = sum(any(newCounts > 0, 2));

fprintf('Old detector: %d total detections across %d years (%d - %d)\n', ...
    oldTotal, nOldYears, oldYears(1), oldYears(end));
fprintf('              mean = %.1f detections/year\n', oldTotal / max(nOldYears, 1));
fprintf('New detector: %d total detections across %d years (%d - %d)\n', ...
    newTotal, nNewYears, newYears(1), newYears(end));
fprintf('              mean = %.1f detections/year\n', newTotal / max(nNewYears, 1));

if nOldYears > 0 && nNewYears > 0
    fprintf('Ratio (new mean / old mean) = %.2fx\n', ...
        (newTotal / nNewYears) / (oldTotal / nOldYears));
end

%% Save CSVs
oldCsvPath = fullfile(outputDir, 'detector_counts_old_2002_2018.csv');
newCsvPath = fullfile(outputDir, 'detector_counts_new_2019_2025.csv');
writetable(oldTable, oldCsvPath, 'WriteRowNames', true);
writetable(newTable, newCsvPath, 'WriteRowNames', true);
fprintf('\nWrote:\n  %s\n  %s\n', oldCsvPath, newCsvPath);

%% Quick visual: yearly totals
figure('Name', 'Detection counts: old vs new detector', 'Color', 'w');
allYears  = [oldYears, newYears];
allTotals = [sum(oldCounts, 2); sum(newCounts, 2)]';
allColors = [repmat([0.4 0.4 0.7], numel(oldYears), 1); ...
             repmat([0.85 0.4 0.2], numel(newYears), 1)];
b = bar(allYears, allTotals, 'FaceColor', 'flat');
b.CData = allColors;
xlabel('Year'); ylabel('Detections per year');
title('Diego Garcia H08S1 - annual detection counts');
grid on;
% Legend proxy
hold on;
hOld = bar(nan, nan, 'FaceColor', [0.4 0.4 0.7]);
hNew = bar(nan, nan, 'FaceColor', [0.85 0.4 0.2]);
legend([hOld, hNew], {'Old detector (CTBTO)', 'New detector (GAVDNet)'}, ...
    'Location', 'northwest');
hold off;
