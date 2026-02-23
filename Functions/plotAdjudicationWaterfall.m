function plotAdjudicationWaterfall(metricsArray, logicNames, nTP_orig, nFP_orig, nFN_orig, figPath)
% PLOTADJUDICATIONWATERFALL Grouped bars showing pre/post adjudication changes
%
% Creates a 1x3 tiled layout (one per decision logic) with grouped bars
% showing pre-adjudication and post-adjudication counts for TP, FP, FN,
% and TN. Pre-adj bars use outline style; post-adj bars are filled. Delta
% annotations show the change between pre- and post-adjudication counts.
%
% Pre-adjudication TN is omitted (undefined in detection problems).
%
% Inputs:
%   metricsArray - Cell array of metrics structs (one per logic)
%   logicNames   - Cell array of logic name strings
%   nTP_orig     - Pre-adjudication true positive count
%   nFP_orig     - Pre-adjudication false positive count
%   nFN_orig     - Pre-adjudication false negative count
%   figPath      - String path for saving figure (without extension)
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

nLogics = length(metricsArray);

%% Create figure

fontSz = 12;
fig = figure('Units', 'inches', 'Position', [2, 2, 12, 4]);
set(fig, 'DefaultAxesFontName', 'Aptos', 'DefaultAxesFontSize', fontSz);
set(fig, 'DefaultTextFontName', 'Aptos', 'DefaultTextFontSize', fontSz);

tiledlayout(1, nLogics, 'TileSpacing', 'compact', 'Padding', 'compact');

categoryLabels = {'TP', 'FP', 'FN', 'TN'};
nCategories = length(categoryLabels);

%% Plot each logic

for iLogic = 1:nLogics
    nexttile;

    metrics = metricsArray{iLogic};

    % Pre-adjudication counts (TN = 0 placeholder, bar will not be drawn)
    preCounts  = [nTP_orig, nFP_orig, nFN_orig, 0];
    postCounts = [metrics.nTruePositives, metrics.nFalsePositives, ...
                  metrics.nFalseNegatives, metrics.nTrueNegatives];

    % Grouped bar: rows = categories, cols = [pre, post]
    barData = [preCounts; postCounts]';
    b = bar(barData);

    % Style: pre-adj = outline only, post-adj = filled
    b(1).FaceColor = 'none';
    b(1).EdgeColor = [0.3, 0.3, 0.3];
    b(1).LineWidth = 1.5;
    b(2).FaceColor = [0.2, 0.5, 0.7];
    b(2).EdgeColor = [0.2, 0.5, 0.7];

    set(gca, 'XTickLabel', categoryLabels);
    ylabel('Count');
    title(logicNames{iLogic});
    grid on;

    % Add delta annotations above post-adj bars
    hold on;
    for k = 1:nCategories
        % Skip TN pre-adj bar (no meaningful pre-adj value)
        if k == nCategories
            delta = postCounts(k);  % TN didn't exist before
            deltaStr = sprintf('+%d', delta);
        else
            delta = postCounts(k) - preCounts(k);
            if delta > 0
                deltaStr = sprintf('+%d', delta);
            elseif delta < 0
                deltaStr = sprintf('%d', delta);  % minus sign included
            else
                deltaStr = '0';
            end
        end

        % Position text above the taller bar in each group
        yPos = max(preCounts(k), postCounts(k));
        xPos = b(2).XEndPoints(k);
        text(xPos, yPos, deltaStr, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', fontSz - 2, ...
            'FontWeight', 'bold');
    end
    hold off;

    if iLogic == 1
        legend({'Pre-adjudication', 'Post-adjudication'}, 'Location', 'best');
    end
end

sgtitle('Pre- vs Post-Adjudication Detection Counts', ...
    'FontSize', fontSz + 2, 'FontWeight', 'bold');

%% Save figure

savefig(fig, strcat(figPath, '.fig'));
print(fig, strcat(figPath, '.emf'), '-dmeta');
print(fig, strcat(figPath, '.svg'), '-dsvg');
print(fig, strcat(figPath, '.tif'), '-dtiff', '-r300');

fprintf('Adjudication waterfall figure saved:\n');
fprintf('  .fig: %s\n', strcat(figPath, '.fig'));
fprintf('  .emf: %s\n', strcat(figPath, '.emf'));
fprintf('  .svg: %s\n', strcat(figPath, '.svg'));
fprintf('  .tif: %s\n\n', strcat(figPath, '.tif'));

end
