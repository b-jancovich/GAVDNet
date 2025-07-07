function plotRecallPrecisionCurve(sweepResults, figPath)
% PLOTRECALLPRECISIONCURVE Creates and saves recall-precision curve figure
%
% This function creates a figure showing how precision and recall vary as a
% function of activation threshold (AT) for the parametric sweep results.
% The figure is saved in multiple formats (.fig, .emf, .svg, .tif).
%
% INPUTS:
%   sweepResults - Structure containing AT_values, precision, recall, f1Score
%   figPath      - String path for saving figure (without extension)
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

    % Sort results by activation threshold for proper plotting
    [sortedAT, sortIdx] = sort(sweepResults.AT_values);
    sortedPrecision = sweepResults.precision(sortIdx);
    sortedRecall = sweepResults.recall(sortIdx);
    sortedF1 = sweepResults.f1Score(sortIdx);
    
    % Set font size
    fontSz = 12;
    
    % Create figure sized appropriately for A4 page
    % Using inches for better control - 6.5" width fits within A4 margins
    fig = figure('Units', 'inches', 'Position', [2, 2, 6.5, 3.5]);
    
    % Set font for all text
    set(fig, 'DefaultAxesFontName', 'Aptos', 'DefaultAxesFontSize', fontSz);
    set(fig, 'DefaultTextFontName', 'Aptos', 'DefaultTextFontSize', fontSz);
    
    % Create tiled layout with tighter spacing
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Subplot 1: Precision vs Recall scatter plot colored by Activation Threshold
    nexttile;
    scatter(sortedRecall, sortedPrecision, 50, sortedAT, 'filled', 'o');
    xlabel('Recall');
    ylabel('Precision');
    colormap(parula);
    c = colorbar('Location', 'eastoutside', 'Limits', [0, 1]);
    c.Label.String = 'Activation Threshold';
    c.FontSize = fontSz;
    c.Ticks = 0:0.2:1;
    grid on;
    xlim([0, 1]);
    ylim([0, 1]);
    set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);
    axis square;
    
    % Subplot 2: F1-Score vs Activation Threshold
    nexttile;
    plot(sortedAT, sortedF1, '*-');
    xlabel('Activation Threshold');
    ylabel('F1-Score');
    grid on;
    xlim([0, 1]);
    ylim([0, 1]);
    set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);  
    axis square;
    
    % Find and highlight optimal F1-score
    [maxF1, maxF1Idx] = max(sortedF1);
    hold on;
    plot(sortedAT(maxF1Idx), maxF1, 'ro', 'MarkerSize', 15, 'LineWidth', 1);
    
    % Save figure in multiple formats
    savefig(fig, strcat(figPath, '.fig'));
    print(fig, strcat(figPath, '.emf'), '-dmeta');
    print(fig, strcat(figPath, '.svg'), '-dsvg');
    print(fig, strcat(figPath, '.tif'), '-dtiff', '-r300');
    
    fprintf('Figure saved in multiple formats:\n');
    fprintf('  .fig: %s\n', strcat(figPath, '.fig'));
    fprintf('  .emf: %s\n', strcat(figPath, '.emf'));
    fprintf('  .svg: %s\n', strcat(figPath, '.svg'));
    fprintf('  .tif: %s\n', strcat(figPath, '.tif'));
    
    close(fig);
end