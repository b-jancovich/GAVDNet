function plotAdjudicatedROCCurves(metricsArray, logicNames, figPath)
% PLOTADJUDICATEDROCCURVES Overlaid ROC curves for all decision logics
%
% Creates a single axes with one ROC curve per decision logic, a diagonal
% reference line, and AUC values in the legend.
%
% Inputs:
%   metricsArray - Cell array of metrics structs (one per logic), each
%                  containing .roc.fpr, .roc.tpr, and .auc
%   logicNames   - Cell array of logic name strings
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
fig = figure('Units', 'inches', 'Position', [2, 2, 5, 4.5]);
set(fig, 'DefaultAxesFontName', 'Aptos', 'DefaultAxesFontSize', fontSz);
set(fig, 'DefaultTextFontName', 'Aptos', 'DefaultTextFontSize', fontSz);

%% Plot ROC curves

colours = [0.2, 0.4, 0.8;   % blue
           0.8, 0.4, 0.2;   % orange
           0.4, 0.7, 0.3];  % green

hold on;

% Diagonal reference line
plot([0, 1], [0, 1], '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1, ...
    'HandleVisibility', 'off');

legendEntries = cell(nLogics, 1);

for iLogic = 1:nLogics
    metrics = metricsArray{iLogic};
    fpr = metrics.roc.fpr;
    tpr = metrics.roc.tpr;
    aucVal = metrics.auc;

    if isscalar(fpr) && isnan(fpr)
        % No valid ROC data
        legendEntries{iLogic} = sprintf('%s (no data)', logicNames{iLogic});
        plot(NaN, NaN, '-', 'Color', colours(iLogic, :), 'LineWidth', 2);
    else
        plot(fpr, tpr, '-', 'Color', colours(iLogic, :), 'LineWidth', 2);
        legendEntries{iLogic} = sprintf('%s (AUC = %.3f)', logicNames{iLogic}, aucVal);
    end
end

hold off;

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Post-Adjudication ROC Curves');
legend(legendEntries, 'Location', 'southeast');
grid on;
xlim([0, 1]);
ylim([0, 1]);
set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);
axis square;

%% Save figure

savefig(fig, strcat(figPath, '.fig'));
print(fig, strcat(figPath, '.emf'), '-dmeta');
print(fig, strcat(figPath, '.svg'), '-dsvg');
print(fig, strcat(figPath, '.tif'), '-dtiff', '-r300');

fprintf('ROC curves figure saved:\n');
fprintf('  .fig: %s\n', strcat(figPath, '.fig'));
fprintf('  .emf: %s\n', strcat(figPath, '.emf'));
fprintf('  .svg: %s\n', strcat(figPath, '.svg'));
fprintf('  .tif: %s\n\n', strcat(figPath, '.tif'));

end
