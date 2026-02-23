function plotAdjudicationOutcomeBreakdown(disagreements, figPath)
% PLOTADJUDICATIONOUTCOMEBREAKDOWN Stacked bar chart of analyst decisions
%
% Creates a stacked bar chart showing the distribution of analyst decisions
% for false positives and false negatives. Each bar is segmented into the
% four possible analyst decisions: DiscreteCallsPresent, ChorusPresent,
% DiscreteCallsChorusPresent, and CallChorusAbsent.
%
% This figure is independent of decision logic since it shows the raw
% analyst classifications before any logic is applied.
%
% Inputs:
%   disagreements - Struct with fields:
%                     .falsePositives(i).analystDecision
%                     .falseNegatives(i).analystDecision
%   figPath       - String path for saving figure (without extension)
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%% Count decisions

decisionCategories = {'DiscreteCallsPresent', 'ChorusPresent', ...
                      'DiscreteCallsChorusPresent', 'CallChorusAbsent'};
nCategories = length(decisionCategories);

nFP = length(disagreements.falsePositives);
nFN = length(disagreements.falseNegatives);

fpCounts = zeros(1, nCategories);
fnCounts = zeros(1, nCategories);

for i = 1:nFP
    for k = 1:nCategories
        if strcmp(disagreements.falsePositives(i).analystDecision, decisionCategories{k})
            fpCounts(k) = fpCounts(k) + 1;
        end
    end
end

for i = 1:nFN
    for k = 1:nCategories
        if strcmp(disagreements.falseNegatives(i).analystDecision, decisionCategories{k})
            fnCounts(k) = fnCounts(k) + 1;
        end
    end
end

% Rows = groups (FP, FN), columns = categories
barData = [fpCounts; fnCounts];

%% Create figure

fontSz = 12;
fig = figure('Units', 'inches', 'Position', [2, 2, 6.5, 3.5]);
set(fig, 'DefaultAxesFontName', 'Aptos', 'DefaultAxesFontSize', fontSz);
set(fig, 'DefaultTextFontName', 'Aptos', 'DefaultTextFontSize', fontSz);

%% Plot stacked bar chart

b = bar(barData, 'stacked');

% Color scheme: 3 greens/teals for vocal activity, red for absent
colours = [0.20, 0.65, 0.50;   % teal       - Discrete
           0.40, 0.80, 0.60;   % light green - Chorus
           0.15, 0.50, 0.55;   % dark teal   - Discrete+Chorus
           0.80, 0.35, 0.35];  % muted red   - Absent

for k = 1:nCategories
    b(k).FaceColor = colours(k, :);
end

set(gca, 'XTickLabel', {'False Positives', 'False Negatives'});
ylabel('Count');
title('Adjudication Outcome Breakdown');

legend({'Discrete Calls', 'Chorus', 'Discrete + Chorus', 'Absent'}, ...
    'Location', 'best');
grid on;

%% Add count annotations on each segment

for iGroup = 1:2
    cumHeight = 0;
    for k = 1:nCategories
        segVal = barData(iGroup, k);
        if segVal > 0
            yPos = cumHeight + segVal / 2;
            text(iGroup, yPos, sprintf('%d', segVal), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', fontSz - 1, ...
                'FontWeight', 'bold', ...
                'Color', 'w');
        end
        cumHeight = cumHeight + segVal;
    end
end

%% Save figure

savefig(fig, strcat(figPath, '.fig'));
print(fig, strcat(figPath, '.emf'), '-dmeta');
print(fig, strcat(figPath, '.svg'), '-dsvg');
print(fig, strcat(figPath, '.tif'), '-dtiff', '-r300');

fprintf('Adjudication outcome breakdown figure saved:\n');
fprintf('  .fig: %s\n', strcat(figPath, '.fig'));
fprintf('  .emf: %s\n', strcat(figPath, '.emf'));
fprintf('  .svg: %s\n', strcat(figPath, '.svg'));
fprintf('  .tif: %s\n\n', strcat(figPath, '.tif'));

end
