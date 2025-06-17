function results = analyzeTrainingInfo(trainInfo, options)
% analyzeTrainingInfo - Analyze training metrics for common problems
%
% This function evaluates training information returned by trainnet to
% identify potential issues such as overfitting, underfitting, gradient
% problems, and other training anomalies.
%
% Inputs:
%   trainInfo - Training information structure returned by trainnet
%
% Name-Value Arguments:
%   Verbose - Print detailed analysis (default: true)
%   GradientThreshold - Threshold for gradient explosion/vanishing detection
%                      as [vanishing_threshold, explosion_threshold]
%                      (default: [1e-6, 10])
%   ConvergenceWindow - Window size for convergence analysis (default: 10)
%   OverfitThreshold - Threshold for overfitting detection (default: 0.1)
%   OscillationThreshold - Threshold for detecting loss oscillations (default: 0.3)
%
% Outputs:
%   results - Structure containing analysis results
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia

% Set default values.
if nargin == 1
    options.Verbose = true;
    options.GradientThreshold = [1e-6, 10];
    options.ConvergenceWindow = 10;
    options.OverfitThreshold = 0.1;
    options.OscillationThreshold = 0.3;
end

% Initialize results structure
results = struct();
results.MetricsFound = {};
results.Issues = {};
results.Recommendations = {};

% Get available fields in trainInfo
availableFields = fieldnames(trainInfo);
results.MetricsFound = availableFields;

if options.Verbose
    fprintf('\n=== TRAINING ANALYSIS REPORT ===\n');
    fprintf('Tracked metrics: %s\n', strjoin(availableFields, ', '));
    fprintf('Training iterations: %d\n', height(trainInfo.(availableFields{1})));
end

% Determine which metrics are available
hasTrainingLoss = isFieldWithData(trainInfo, 'TrainingLoss');
hasValidationLoss = isFieldWithData(trainInfo, 'ValidationLoss');
hasGradientNorm = isFieldWithData(trainInfo, 'GradientNorm');
hasLearnRate = isFieldWithData(trainInfo, 'BaseLearnRate');

% Check for other possible field names
if ~hasTrainingLoss
    hasTrainingLoss = isFieldWithData(trainInfo, 'Loss') || isFieldWithData(trainInfo, 'TrainingAccuracy');
end

% 1. Training Loss Analysis
if hasTrainingLoss
    if isfield(trainInfo, 'TrainingLoss')
        trainLoss = trainInfo.TrainingLoss;
        lossName = 'TrainingLoss';
    elseif isfield(trainInfo, 'Loss')
        trainLoss = trainInfo.Loss;
        lossName = 'Loss';
    else
        % Find first numeric field that could be a loss
        for i = 1:length(availableFields)
            if isnumeric(trainInfo.(availableFields{i})) && length(trainInfo.(availableFields{i})) > 1
                trainLoss = trainInfo.(availableFields{i});
                lossName = availableFields{i};
                break;
            end
        end
    end
    
    results.TrainingLoss = analyzeLoss(trainLoss, lossName, options);
    
    % 2. Validation Loss Analysis and Overfitting Detection
    if hasValidationLoss
        valLoss = trainInfo.ValidationLoss;
        results.ValidationLoss = analyzeLoss(valLoss, 'ValidationLoss', options);
        results.OverfittingAnalysis = analyzeOverfitting(trainLoss, valLoss, options);
    else
        results.Issues{end+1} = 'No validation loss available - cannot detect overfitting';
        if options.Verbose
            fprintf('\nWARNING: No validation data used - overfitting detection not possible\n');
        end
    end
    
    % 5. Overall Convergence Analysis
    results.ConvergenceAnalysis = analyzeConvergence(trainLoss, options);
end

% 3. Gradient Analysis
if hasGradientNorm
    gradNorm = trainInfo.GradientNorm;
    results.GradientAnalysis = analyzeGradients(gradNorm, options);
end

% 4. Learning Rate Analysis
if hasLearnRate
    learnRate = trainInfo.BaseLearnRate;
    if hasTrainingLoss
        results.LearningRateAnalysis = analyzeLearningRate(learnRate, trainLoss, options);
    else
        results.LearningRateAnalysis = analyzeLearningRate(learnRate, [], options);
    end
end

% 6. Analyze other metrics
otherMetrics = setdiff(availableFields, {'TrainingLoss', 'ValidationLoss', 'GradientNorm', 'BaseLearnRate'});
for i = 1:length(otherMetrics)
    metric = otherMetrics{i};
    if isnumeric(trainInfo.(metric)) && length(trainInfo.(metric)) > 1
        results.OtherMetrics.(metric) = analyzeGenericMetric(trainInfo.(metric), metric, options);
    end
end

% Print summary and recommendations
if options.Verbose
    printSummary(results);
end
end

function hasData = isFieldWithData(s, fieldName)
% Check if field exists and contains numeric data
hasData = isfield(s, fieldName) && isnumeric(s.(fieldName)) && length(s.(fieldName)) > 1;
end

function lossResults = analyzeLoss(loss, lossType, options)
% Analyze individual loss curves

lossResults = struct();
lossResults.Type = lossType;
lossResults.FinalValue = loss(end);
lossResults.InitialValue = loss(1);

if lossResults.InitialValue ~= 0
    lossResults.ReductionPercent = (lossResults.InitialValue - lossResults.FinalValue) / lossResults.InitialValue * 100;
else
    lossResults.ReductionPercent = 0;
end

% Check for convergence in recent iterations
window = min(options.ConvergenceWindow, length(loss));
if length(loss) >= window
    recentValues = loss(end-window+1:end);
    lossResults.RecentMean = mean(recentValues);
    lossResults.RecentStd = std(recentValues);
    
    % Coefficient of variation for convergence assessment
    if lossResults.RecentMean > 0
        cv = lossResults.RecentStd / lossResults.RecentMean;
        lossResults.IsConverged = cv < 0.01; % 1% coefficient of variation
        lossResults.CoefficientOfVariation = cv;
    else
        lossResults.IsConverged = true;
        lossResults.CoefficientOfVariation = 0;
    end
else
    lossResults.IsConverged = false;
    lossResults.CoefficientOfVariation = inf;
end

% Check for oscillations and instability
if length(loss) > 5
    diff1 = diff(loss);
    signChanges = sum(abs(diff(sign(diff1))) > 0);
    lossResults.OscillationRatio = signChanges / length(diff1);
    lossResults.HighOscillations = lossResults.OscillationRatio > options.OscillationThreshold;
    
    % Check for increasing loss (very bad sign)
    lossResults.IsIncreasing = loss(end) > loss(1);
    
    % Standard deviation of loss changes
    lossResults.Stability = std(diff1) / mean(abs(diff1));
else
    lossResults.OscillationRatio = 0;
    lossResults.HighOscillations = false;
    lossResults.IsIncreasing = false;
    lossResults.Stability = 0;
end

if options.Verbose
    fprintf('\n--- %s Analysis ---\n', lossType);
    fprintf('Initial: %.6f, Final: %.6f (%.2f%% reduction)\n', ...
        lossResults.InitialValue, lossResults.FinalValue, lossResults.ReductionPercent);
    fprintf('Converged: %s (CV: %.4f)\n', mat2str(lossResults.IsConverged), lossResults.CoefficientOfVariation);
    
    if lossResults.HighOscillations
        fprintf('WARNING: High oscillations detected (%.1f%% direction changes)\n', lossResults.OscillationRatio*100);
    end
    if lossResults.IsIncreasing
        fprintf('WARNING: Loss is increasing overall!\n');
    end
    if lossResults.Stability > 2
        fprintf('WARNING: Unstable loss (high variability)\n');
    end
end

end

function overfitResults = analyzeOverfitting(trainLoss, valLoss, options)
% Analyze overfitting patterns between training and validation loss

overfitResults = struct();

% Ensure same length for comparison
minLen = min(length(trainLoss), length(valLoss));
if minLen < 5
    overfitResults.InsufficientData = true;
    overfitResults.IsOverfitting = false;
    return;
end

trainLoss = trainLoss(1:minLen);
valLoss = valLoss(1:minLen);

% Calculate gap between training and validation loss
gap = valLoss - trainLoss;
overfitResults.FinalGap = gap(end);
overfitResults.MaxGap = max(gap);
overfitResults.MeanGap = mean(gap);
overfitResults.GapTrend = gap(end) - gap(1);

% Analyze trends in latter half of training
halfPoint = max(1, ceil(minLen/2));
if minLen - halfPoint >= 3
    laterIndices = halfPoint:minLen;
    
    % Fit linear trends
    trainTrend = polyfit(laterIndices, trainLoss(laterIndices), 1);
    valTrend = polyfit(laterIndices, valLoss(laterIndices), 1);
    
    overfitResults.TrainingTrendSlope = trainTrend(1);
    overfitResults.ValidationTrendSlope = valTrend(1);
    overfitResults.TrendDivergence = valTrend(1) - trainTrend(1);
    
    % Overfitting detection criteria
    divergingTrends = overfitResults.TrendDivergence > options.OverfitThreshold;
    significantGap = abs(overfitResults.FinalGap) > options.OverfitThreshold;
    valIncreasing = valTrend(1) > 0;
    trainDecreasing = trainTrend(1) < 0;
    
    overfitResults.IsOverfitting = divergingTrends && significantGap && (valIncreasing || trainDecreasing);
    
    % Assess severity
    if overfitResults.IsOverfitting
        if abs(overfitResults.FinalGap) > 3 * options.OverfitThreshold
            overfitResults.Severity = 'Severe';
        elseif abs(overfitResults.FinalGap) > 2 * options.OverfitThreshold
            overfitResults.Severity = 'Moderate';
        else
            overfitResults.Severity = 'Mild';
        end
    else
        overfitResults.Severity = 'None';
    end
else
    overfitResults.IsOverfitting = false;
    overfitResults.Severity = 'Unknown';
end

if options.Verbose
    fprintf('\n--- Overfitting Analysis ---\n');
    fprintf('Final gap (val - train): %.6f\n', overfitResults.FinalGap);
    fprintf('Maximum gap: %.6f\n', overfitResults.MaxGap);
    
    if overfitResults.IsOverfitting
        fprintf('*** OVERFITTING DETECTED (%s) ***\n', overfitResults.Severity);
        fprintf('  Training loss slope: %.8f\n', overfitResults.TrainingTrendSlope);
        fprintf('  Validation loss slope: %.8f\n', overfitResults.ValidationTrendSlope);
        fprintf('  Trend divergence: %.8f\n', overfitResults.TrendDivergence);
    else
        fprintf('No significant overfitting detected\n');
    end
end

end

function gradResults = analyzeGradients(gradNorm, options)
% Analyze gradient norm behavior

gradResults = struct();
gradResults.Mean = mean(gradNorm);
gradResults.Std = std(gradNorm);
gradResults.Median = median(gradNorm);
gradResults.Min = min(gradNorm);
gradResults.Max = max(gradNorm);
gradResults.Final = gradNorm(end);
gradResults.Initial = gradNorm(1);

% Gradient health assessment
lowThresh = options.GradientThreshold(1);
highThresh = options.GradientThreshold(2);

gradResults.VanishingGradients = gradResults.Median < lowThresh;
gradResults.ExplodingGradients = gradResults.Max > highThresh;

% Check for gradient instability (high coefficient of variation)
if gradResults.Mean > 0
    cv = gradResults.Std / gradResults.Mean;
    gradResults.CoefficientOfVariation = cv;
    gradResults.Unstable = cv > 2.0;
else
    gradResults.CoefficientOfVariation = inf;
    gradResults.Unstable = true;
end

% Trend analysis
if length(gradNorm) > 5
    trend = polyfit(1:length(gradNorm), gradNorm, 1);
    gradResults.Trend = trend(1);
    gradResults.TrendDirection = sign(trend(1));
else
    gradResults.Trend = 0;
    gradResults.TrendDirection = 0;
end

if options.Verbose
    fprintf('\n--- Gradient Analysis ---\n');
    fprintf('Statistics: Mean=%.2e, Median=%.2e, Std=%.2e\n', ...
        gradResults.Mean, gradResults.Median, gradResults.Std);
    fprintf('Range: [%.2e, %.2e], Final=%.2e\n', ...
        gradResults.Min, gradResults.Max, gradResults.Final);
    
    if gradResults.VanishingGradients
        fprintf('*** WARNING: Vanishing gradients detected (median: %.2e) ***\n', gradResults.Median);
    end
    if gradResults.ExplodingGradients
        fprintf('*** WARNING: Exploding gradients detected (max: %.2e) ***\n', gradResults.Max);
    end
    if gradResults.Unstable
        fprintf('WARNING: Unstable gradients (CV: %.2f)\n', gradResults.CoefficientOfVariation);
    end
    
    if gradResults.TrendDirection > 0
        fprintf('Gradient norms are increasing over time\n');
    elseif gradResults.TrendDirection < 0
        fprintf('Gradient norms are decreasing over time\n');
    end
end

end

function lrResults = analyzeLearningRate(learnRate, trainLoss, options)
% Analyze learning rate behavior and appropriateness

lrResults = struct();

if isscalar(learnRate) || all(learnRate == learnRate(1))
    lrResults.Type = 'Fixed';
    lrResults.Value = learnRate(1);
else
    lrResults.Type = 'Adaptive';
    lrResults.Initial = learnRate(1);
    lrResults.Final = learnRate(end);
    lrResults.Min = min(learnRate);
    lrResults.Max = max(learnRate);
    lrResults.ReductionRatio = (learnRate(1) - learnRate(end)) / learnRate(1);
    
    % Analyze adaptation pattern
    if length(learnRate) > 5
        lrChanges = diff(learnRate);
        lrResults.Reductions = sum(lrChanges < 0);
        lrResults.Increases = sum(lrChanges > 0);
        lrResults.AdaptationFrequency = (lrResults.Reductions + lrResults.Increases) / length(lrChanges);
    end
end

% If training loss is available, assess learning rate appropriateness
if ~isempty(trainLoss) && length(trainLoss) == length(learnRate)
    % Simple heuristic: learning rate should adapt with loss behavior
    lossChanges = diff(trainLoss);
    badUpdates = sum(lossChanges > 0); % iterations where loss increased
    lrResults.BadUpdateRatio = badUpdates / length(lossChanges);
    lrResults.PossiblyTooHigh = lrResults.BadUpdateRatio > 0.2; % More than 20% bad updates
end

if options.Verbose
    fprintf('\n--- Learning Rate Analysis ---\n');
    if strcmp(lrResults.Type, 'Fixed')
        fprintf('Fixed learning rate: %.2e\n', lrResults.Value);
    else
        fprintf('Adaptive learning rate: %.2e -> %.2e (%.1f%% reduction)\n', ...
            lrResults.Initial, lrResults.Final, lrResults.ReductionRatio*100);
        if isfield(lrResults, 'AdaptationFrequency')
            fprintf('Adaptation frequency: %.1f%% of iterations\n', lrResults.AdaptationFrequency*100);
        end
    end
    
    if isfield(lrResults, 'PossiblyTooHigh') && lrResults.PossiblyTooHigh
        fprintf('WARNING: Learning rate may be too high (%.1f%% bad updates)\n', lrResults.BadUpdateRatio*100);
    end
end

end

function convResults = analyzeConvergence(loss, options)
% Analyze overall convergence behavior and patterns

convResults = struct();

if loss(1) ~= 0
    convResults.TotalReductionPercent = (loss(1) - loss(end)) / loss(1) * 100;
else
    convResults.TotalReductionPercent = 0;
end

% Estimate convergence rate (exponential decay assumption)
if loss(end) > 0 && loss(1) > 0
    convResults.ConvergenceRate = -log(loss(end)/loss(1)) / length(loss);
else
    convResults.ConvergenceRate = 0;
end

% Find when significant improvement stopped
if length(loss) > 10
    smoothLoss = movmean(loss, min(5, ceil(length(loss)/10)));
    relativeImprovements = -diff(smoothLoss) ./ smoothLoss(1:end-1);
    
    % Find first point where improvement drops below 0.1% per iteration
    improvementThreshold = 0.001;
    plateauStart = find(relativeImprovements < improvementThreshold, 1);
    
    if ~isempty(plateauStart)
        convResults.PlateauStartIteration = plateauStart;
        convResults.PlateauStartPercent = plateauStart / length(loss) * 100;
        
        % Check if this represents premature convergence
        remainingIterations = length(loss) - plateauStart;
        convResults.PrematureConvergence = (remainingIterations > 0.3 * length(loss)) && ...
            (loss(plateauStart) > 0.01);
    else
        convResults.PlateauStartIteration = length(loss);
        convResults.PlateauStartPercent = 100;
        convResults.PrematureConvergence = false;
    end
else
    convResults.PrematureConvergence = false;
    convResults.PlateauStartPercent = 100;
end

% Assess final convergence quality
finalWindow = min(options.ConvergenceWindow, length(loss));
if length(loss) >= finalWindow
    finalValues = loss(end-finalWindow+1:end);
    convResults.FinalStability = std(finalValues) / mean(finalValues);
    convResults.WellConverged = convResults.FinalStability < 0.01;
else
    convResults.WellConverged = false;
    convResults.FinalStability = inf;
end

if options.Verbose
    fprintf('\n--- Convergence Analysis ---\n');
    fprintf('Total loss reduction: %.2f%%\n', convResults.TotalReductionPercent);
    fprintf('Convergence rate: %.6f per iteration\n', convResults.ConvergenceRate);
    
    if isfield(convResults, 'PlateauStartPercent')
        fprintf('Improvement plateau started at: %.1f%% of training\n', convResults.PlateauStartPercent);
    end
    
    if convResults.PrematureConvergence
        fprintf('WARNING: Possible premature convergence detected\n');
    end
    
    if convResults.WellConverged
        fprintf('Well converged (final stability: %.4f)\n', convResults.FinalStability);
    else
        fprintf('Convergence incomplete (final stability: %.4f)\n', convResults.FinalStability);
    end
end

end

function metricResults = analyzeGenericMetric(metricData, metricName, options)
% Analyze any other numeric metric

metricResults = struct();
metricResults.Name = metricName;
metricResults.Initial = metricData(1);
metricResults.Final = metricData(end);
metricResults.Mean = mean(metricData);
metricResults.Std = std(metricData);
metricResults.Min = min(metricData);
metricResults.Max = max(metricData);

% Determine if this looks like an accuracy metric (should increase)
% or a loss metric (should decrease)
if contains(lower(metricName), {'accuracy', 'acc', 'precision', 'recall', 'f1'})
    metricResults.ShouldIncrease = true;
    metricResults.Improvement = metricResults.Final - metricResults.Initial;
    metricResults.IsImproving = metricResults.Improvement > 0;
else
    metricResults.ShouldIncrease = false;
    metricResults.Improvement = metricResults.Initial - metricResults.Final;
    metricResults.IsImproving = metricResults.Improvement > 0;
end

if options.Verbose && ~strcmp(metricName, 'Iteration')
    fprintf('\n--- %s Analysis ---\n', metricName);
    fprintf('Initial: %.6f, Final: %.6f\n', metricResults.Initial, metricResults.Final);
    if metricResults.IsImproving
        fprintf('Improving (%+.6f)\n', metricResults.Improvement);
    else
        fprintf('Not improving (%+.6f)\n', metricResults.Improvement);
    end
end

end

function printSummary(results)

% Print overall summary and actionable recommendations

fprintf('\n=== TRAINING HEALTH SUMMARY ===\n');

% Collect all issues and recommendations
allIssues = {};
allRecommendations = {};

% Check training loss issues
if isfield(results, 'TrainingLoss')
    tl = results.TrainingLoss;
    if tl.IsIncreasing
        allIssues{end+1} = 'Training loss is increasing';
        allRecommendations{end+1} = 'URGENT: Check data preprocessing, reduce learning rate, or revise model architecture';
    elseif tl.HighOscillations
        allIssues{end+1} = 'High training loss oscillations';
        allRecommendations{end+1} = 'Reduce learning rate, use learning rate scheduler, or add batch normalization';
    elseif ~tl.IsConverged
        allIssues{end+1} = 'Training loss not converged';
        allRecommendations{end+1} = 'Train longer, adjust learning rate, or check for gradient issues';
    end
end

% Check overfitting
if isfield(results, 'OverfittingAnalysis') && results.OverfittingAnalysis.IsOverfitting
    severity = results.OverfittingAnalysis.Severity;
    allIssues{end+1} = sprintf('%s overfitting detected', severity);
    if strcmp(severity, 'Severe')
        allRecommendations{end+1} = 'URGENT: Add strong regularization, reduce model complexity, increase dataset size';
    else
        allRecommendations{end+1} = 'Add regularization (dropout, weight decay), early stopping, or data augmentation';
    end
end

% Check gradient issues
if isfield(results, 'GradientAnalysis')
    ga = results.GradientAnalysis;
    if ga.VanishingGradients
        allIssues{end+1} = 'Vanishing gradients';
        allRecommendations{end+1} = 'Use better initialization (Xavier/He), add residual connections, or gradient clipping';
    end
    if ga.ExplodingGradients
        allIssues{end+1} = 'Exploding gradients';
        allRecommendations{end+1} = 'Apply gradient clipping, reduce learning rate, or add batch normalization';
    end
    if ga.Unstable
        allIssues{end+1} = 'Unstable gradients';
        allRecommendations{end+1} = 'Add batch normalization, reduce learning rate, or use gradient clipping';
    end
end

% Check learning rate issues
if isfield(results, 'LearningRateAnalysis') && isfield(results.LearningRateAnalysis, 'PossiblyTooHigh')
    if results.LearningRateAnalysis.PossiblyTooHigh
        allIssues{end+1} = 'Learning rate may be too high';
        allRecommendations{end+1} = 'Reduce initial learning rate or use learning rate scheduler';
    end
end

% Check convergence issues
if isfield(results, 'ConvergenceAnalysis')
    ca = results.ConvergenceAnalysis;
    if ca.PrematureConvergence
        allIssues{end+1} = 'Premature convergence';
        allRecommendations{end+1} = 'Use learning rate scheduler, increase model capacity, or train longer';
    elseif ~ca.WellConverged
        allIssues{end+1} = 'Poor convergence quality';
        allRecommendations{end+1} = 'Train longer, adjust learning rate schedule, or improve optimization settings';
    end
end

% Print results
if isempty(allIssues)
    fprintf('✓ No major issues detected - Training appears healthy!\n');
    fprintf('✓ Model training completed successfully\n');
    uniqueRecs = [];
else
    fprintf('Issues Detected (%d):\n', length(allIssues));
    for i = 1:length(allIssues)
        fprintf('  ⚠ %s\n', allIssues{i});
    end
    
    fprintf('\nRecommended Actions:\n');
    uniqueRecs = unique(allRecommendations, 'stable');
    for i = 1:length(uniqueRecs)
        fprintf('  → %s\n', uniqueRecs{i});
    end
end

% Store in results
results.Issues = allIssues;
results.Recommendations = uniqueRecs;

end
