% Analyse confidence score distributions
function [confPercentiles, percentiles] = analyseConfidenceDistribution(inferenceResults, plotting)
%
% Ben Jancovich, 2025  
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Extract confidence scores
confidenceScores = [inferenceResults.confidence];

if plotting == true
    % Plot confidence distributions for TP, FP, FN
    figure;
    histogram(confidenceScores, 'Normalization', 'percentage');
    xlabel('Confidence Score');
    ylabel('Percentage of Detections');
    title('Distribution of Confidence Scores for Post-Processed Detections');
end

% Calculate percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99];
confPercentiles = prctile(confidenceScores, percentiles);
end
