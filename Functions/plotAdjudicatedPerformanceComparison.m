function plotAdjudicatedPerformanceComparison(metricsArray, logicNames, figPath)
% plotAdjudicatedPerformanceComparison
% Creates comparison figure showing performance metrics across decision logics.
%
% Inputs:
%   metricsArray - Cell array of metrics structures (one per logic)
%   logicNames   - Cell array of logic name strings
%   figPath      - String path for saving figure (without extension)
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

nLogics = length(metricsArray);

% Extract metrics
recall = zeros(nLogics, 1);
precision = zeros(nLogics, 1);
f1Score = zeros(nLogics, 1);
auc = zeros(nLogics, 1);

for i = 1:nLogics
    recall(i) = metricsArray{i}.recall;
    precision(i) = metricsArray{i}.precision;
    f1Score(i) = metricsArray{i}.f1Score;
    auc(i) = metricsArray{i}.auc;
end

% Set font size
fontSz = 12;

% Create figure sized appropriately for A4 page
fig = figure('Units', 'inches', 'Position', [2, 2, 6.5, 3.5]);

% Set font for all text
set(fig, 'DefaultAxesFontName', 'Aptos', 'DefaultAxesFontSize', fontSz);
set(fig, 'DefaultTextFontName', 'Aptos', 'DefaultTextFontSize', fontSz);


%% Grouped Bar Chart

% Create grouped bar data
barData = [recall, precision, f1Score];

b = bar(barData);
b(1).FaceColor = [0.2, 0.4, 0.8];
b(2).FaceColor = [0.8, 0.4, 0.2];
b(3).FaceColor = [0.4, 0.8, 0.2];

xlabel('Decision Logic');
ylabel('Metric Value');
set(gca, 'XTickLabel', logicNames, 'XTickLabelRotation', 45);
legend({'Recall', 'Precision', 'F1-Score'}, 'Location', 'best');
grid on;
ylim([0, 1]);
set(gca, 'YTick', 0:0.1:1);
title('Performance Metrics by Decision Logic');

%% Save Figure

savefig(fig, strcat(figPath, '.fig'));
print(fig, strcat(figPath, '.emf'), '-dmeta');
print(fig, strcat(figPath, '.svg'), '-dsvg');
print(fig, strcat(figPath, '.tif'), '-dtiff', '-r300');

fprintf('Performance comparison figure saved:\n');
fprintf('  .fig: %s\n', strcat(figPath, '.fig'));
fprintf('  .emf: %s\n', strcat(figPath, '.emf'));
fprintf('  .svg: %s\n', strcat(figPath, '.svg'));
fprintf('  .tif: %s\n\n', strcat(figPath, '.tif'));

end