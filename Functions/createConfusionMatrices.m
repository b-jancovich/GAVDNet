function createConfusionMatrices(metricsArray, logicNames, figPath)
% createConfusionMatrices
% Creates confusion matrix visualization for each decision logic.
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

% Set font size
fontSz = 10;

% Create figure
fig = figure('Units', 'inches', 'Position', [1, 1, 10, 8]);

% Set font for all text
set(fig, 'DefaultAxesFontName', 'Aptos', 'DefaultAxesFontSize', fontSz);
set(fig, 'DefaultTextFontName', 'Aptos', 'DefaultTextFontSize', fontSz);

% Create 1x3 grid of tiles
tiledlayout(1, nLogics)
for iLogic = 1:nLogics
    nexttile
    
    metrics = metricsArray{iLogic};
    
    % Extract confusion matrix values
    TP = metrics.nTruePositives;
    FP = metrics.nFalsePositives;
    FN = metrics.nFalseNegatives;
    TN = metrics.nTrueNegatives;
    
    % Create confusion matrix
    confMat = [TP, FN; FP, TN];
    
    % Calculate percentages
    total = sum(confMat(:));
    confMatPercent = confMat / total * 100;
    
    % Create custom visualization
    imagesc(confMat);
    colormap(gca, [0.9, 0.9, 0.9; 0.4, 0.8, 0.4; 0.8, 0.4, 0.4; 0.9, 0.9, 0.9]);
    axis square;
    
    % Add text annotations
    for i = 1:2
        for j = 1:2
            % Count
            text(j, i, sprintf('%d', confMat(i,j)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', fontSz + 2, ...
                'FontWeight', 'bold', ...
                'Color', 'k');
            
            % Percentage
            text(j, i + 0.3, sprintf('(%.1f%%)', confMatPercent(i,j)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', fontSz - 1, ...
                'Color', 'k');
        end
    end
    
    % Labels
    set(gca, 'XTick', [1, 2], 'XTickLabel', {'Predicted Positive', 'Predicted Negative'});
    set(gca, 'YTick', [1, 2], 'YTickLabel', {'Actual Positive', 'Actual Negative'});
    xlabel('Detector Classification');
    ylabel('Ground Truth (Adjudicated)');
    
    % Title with logic name and metrics
    titleStr = sprintf('%s\nRecall=%.3f, Precision=%.3f, F1=%.3f', ...
        logicNames{iLogic}, metrics.recall, metrics.precision, metrics.f1Score);
    title(titleStr, 'FontSize', fontSz);
    
    % Add grid
    hold on;
    plot([1.5, 1.5], [0.5, 2.5], 'k-', 'LineWidth', 1.5);
    plot([0.5, 2.5], [1.5, 1.5], 'k-', 'LineWidth', 1.5);
    hold off;
end

% Overall title
sgtitle('Confusion Matrices for Decision Logics', 'FontSize', fontSz + 2, 'FontWeight', 'bold');

%% Save Figure

savefig(fig, strcat(figPath, '.fig'));
print(fig, strcat(figPath, '.emf'), '-dmeta');
print(fig, strcat(figPath, '.svg'), '-dsvg');
print(fig, strcat(figPath, '.tif'), '-dtiff', '-r300');

fprintf('Confusion matrices figure saved:\n');
fprintf('  .fig: %s\n', strcat(figPath, '.fig'));
fprintf('  .emf: %s\n', strcat(figPath, '.emf'));
fprintf('  .svg: %s\n', strcat(figPath, '.svg'));
fprintf('  .tif: %s\n\n', strcat(figPath, '.tif'));

end