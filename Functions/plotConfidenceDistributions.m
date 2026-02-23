function plotConfidenceDistributions(metricsArray, logicNames, figPath)
% PLOTCONFIDENCEDISTRIBUTIONS Overlapping histograms of TP vs FP confidence
%
% Creates a 1x3 tiled layout (one per decision logic) showing overlapping
% semi-transparent histograms of confidence scores for true positive and
% false positive detections.
%
% Inputs:
%   metricsArray - Cell array of metrics structs (one per logic), each
%                  containing .confidenceScores and .resultLabels
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
fig = figure('Units', 'inches', 'Position', [2, 2, 12, 3.5]);
set(fig, 'DefaultAxesFontName', 'Aptos', 'DefaultAxesFontSize', fontSz);
set(fig, 'DefaultTextFontName', 'Aptos', 'DefaultTextFontSize', fontSz);

tiledlayout(1, nLogics, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Plot histograms

nBins = 30;
tpColour = [0.2, 0.4, 0.8];
fpColour = [0.8, 0.3, 0.3];

for iLogic = 1:nLogics
    nexttile;

    metrics = metricsArray{iLogic};
    scores = metrics.confidenceScores;
    labels = metrics.resultLabels;

    tpScores = scores(labels == 1);
    fpScores = scores(labels == 0);

    % Compute common bin edges across both distributions
    allScores = [tpScores; fpScores];
    if isempty(allScores)
        title(sprintf('%s\n(no data)', logicNames{iLogic}));
        continue;
    end
    edges = linspace(min(allScores), max(allScores), nBins + 1);

    % Plot histograms
    hold on;
    histogram(tpScores, edges, 'FaceColor', tpColour, 'FaceAlpha', 0.5, ...
        'EdgeColor', 'none', 'DisplayName', sprintf('TP (n=%d)', numel(tpScores)));
    histogram(fpScores, edges, 'FaceColor', fpColour, 'FaceAlpha', 0.5, ...
        'EdgeColor', 'none', 'DisplayName', sprintf('FP (n=%d)', numel(fpScores)));
    hold off;

    xlabel('Confidence Score');
    ylabel('Count');
    title(logicNames{iLogic});
    legend('Location', 'best');
    grid on;
    xlim([0, 1]);
end

sgtitle('Confidence Score Distributions: TP vs FP', ...
    'FontSize', fontSz + 2, 'FontWeight', 'bold');

%% Save figure

savefig(fig, strcat(figPath, '.fig'));
print(fig, strcat(figPath, '.emf'), '-dmeta');
print(fig, strcat(figPath, '.svg'), '-dsvg');
print(fig, strcat(figPath, '.tif'), '-dtiff', '-r300');

fprintf('Confidence distribution figure saved:\n');
fprintf('  .fig: %s\n', strcat(figPath, '.fig'));
fprintf('  .emf: %s\n', strcat(figPath, '.emf'));
fprintf('  .svg: %s\n', strcat(figPath, '.svg'));
fprintf('  .tif: %s\n\n', strcat(figPath, '.tif'));

end
