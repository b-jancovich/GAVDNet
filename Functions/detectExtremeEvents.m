function [isExtreme, extremeScore, diagnostics] = detectExtremeEvents(audioEnv)
    % detectExtremeEvents - Detect extreme acoustic events in hydrophone recordings
    %
    % This function implements a novel composite scoring algorithm to identify
    % extreme acoustic events (earthquakes, iceberg calving, etc.) in marine
    % hydrophone recordings based on statistical analysis of envelope amplitude
    % distributions. The method combines multiple statistical discriminators
    % with empirically-derived weights to distinguish extreme events from
    % high dynamic range biological calls.
    %
    % INPUTS:
    %   audioEnv - Audio envelope (linear amplitude, positive values)
    %              To be computed using envelope() function with RMS method
    %
    % OUTPUTS:
    %   isExtreme    - Logical flag indicating extreme event detection
    %   extremeScore - Composite numerical score (higher = more extreme)
    %                  Threshold-independent measure for sensitivity tuning
    %   diagnostics  - Structure containing detailed statistical measures
    %                  and component scores for analysis and validation
    %
    % ALGORITHM:
    %   The detection algorithm uses a weighted composite of seven statistical
    %   measures, with weights empirically derived from analysis of labeled
    %   hydrophone data containing known extreme events:
    %   
    %   1. Variance (35% weight) - Primary discriminator for extreme events
    %   2. Standard deviation (25% weight) - Secondary amplitude spread measure  
    %   3. Dynamic range (15% weight) - Total min-max spread, less weighted
    %      due to overlap with high-amplitude biological calls
    %   4. Skewness (10% weight) - Measures distribution asymmetry
    %   5. Kurtosis (10% weight) - Measures outlier proneness
    %   6. Spread ratio (3% weight) - Ratio of outer to inner percentile ranges
    %   7. Mean-median difference (2% weight) - Additional asymmetry measure
    %
    % THRESHOLDING:
    %   Default threshold = 1.0 (conservative, tune based on validation)
    %   Lower values increase sensitivity (more false positives, fewer false negatives)
    %   Higher values decrease sensitivity (fewer false positives, more false negatives)
    %
    % Ben Jancovich, 2025
    % Centre for Marine Science and Innovation
    % School of Biological, Earth and Environmental Sciences
    % University of New South Wales, Sydney, Australia
    %

    % Convert envelope to dB scale for statistical analysis
    % eps prevents log(0) errors for zero-amplitude samples
    audioEnvdB = 20 * log10(audioEnv + eps);
    
    % BASIC STATISTICAL MEASURES
    % Central tendency and spread measures
    envdBMin = min(audioEnvdB);
    envdBMax = max(audioEnvdB);
    envdBMean = mean(audioEnvdB);
    envdBMedian = median(audioEnvdB);
    envdBStd = std(audioEnvdB);
    envVariance = var(audioEnvdB);
    
    % Distribution shape measures
    envSkewness = skewness(audioEnvdB);  % Asymmetry: >0 indicates right tail
    envKurtosis = kurtosis(audioEnvdB);  % Outlier proneness: >3 indicates heavy tails
    
    % PERCENTILE-BASED ANALYSIS
    % Robust measures less sensitive to individual outliers
    envP01 = prctile(audioEnvdB, 0.1);   % Near-minimum, robust to noise floor
    envP25 = prctile(audioEnvdB, 25);    % First quartile
    envP75 = prctile(audioEnvdB, 75);    % Third quartile  
    envP999 = prctile(audioEnvdB, 99.9); % Near-maximum, robust to single peaks
    
    % DISCRIMINATING FEATURE EXTRACTION
    dynamicRange = abs(envdBMax - envdBMin);     % Total amplitude span
    IQR = envP75 - envP25;                       % Interquartile range (robust spread)
    outerRange = envP999 - envP01;               % Near-total range (outlier-robust)
    spreadRatio = outerRange / IQR;              % Heavy-tail indicator (>5 suggests extremes)
    meanMedianDiff = abs(envdBMean - envdBMedian); % Asymmetry measure
    
    % COMPOSITE SCORING
    % Transform raw statistics into normalized scores [0-5+ range]
    % Scoring functions derived from empirical analysis of labeled data
    scores = struct();
    
    % VARIANCE SCORE (Primary discriminator, 35% weight)
    % Empirical observations: Normal files <12 dB², extreme events >40 dB²
    % Linear scaling with cap to prevent single measure domination
    scores.variance = min(envVariance / 15, 5); % Saturates at 75 dB² variance
    
    % STANDARD DEVIATION SCORE (Secondary discriminator, 25% weight)  
    % Empirical observations: Normal files <4 dB, extreme events >6 dB
    % Zero score below 3 dB, linear increase above threshold
    scores.stdDev = max(0, (envdBStd - 3) / 2); % 1 point per 2 dB above 3 dB
    
    % DYNAMIC RANGE SCORE (Tertiary importance, 15% weight)
    % Less weighted due to overlap between extreme events and high-amplitude calls
    % Conservative threshold to avoid false positives from whale calls
    scores.dynRange = max(0, (dynamicRange - 25) / 15); % 1 point per 15 dB above 25 dB
    
    % DISTRIBUTION SHAPE SCORES (10% weight each)
    % Extreme events typically create asymmetric, heavy-tailed distributions
    scores.skewness = max(0, (envSkewness - 1) / 3); % Positive skew threshold
    scores.kurtosis = max(0, (envKurtosis - 4) / 5); % Excess kurtosis above normal (3)
    scores.spreadRatio = max(0, (spreadRatio - 4) / 3); % Heavy-tail detection
    
    % ASYMMETRY SCORE (Minimal weight, 2%)
    % Additional measure for distribution asymmetry
    scores.asymmetry = meanMedianDiff / 2; % Direct scaling
    
    % WEIGHTED COMPOSITE SCORING
    % Weights derived from discriminative power analysis of labeled data
    % Higher weights assigned to measures with best separation between
    % extreme events and normal/high-dynamic-range biological calls
    weights = struct();
    weights.variance = 0.35;      % Primary discriminator (strongest separation)
    weights.stdDev = 0.25;        % Secondary discriminator  
    weights.dynRange = 0.15;      % Tertiary (overlap with biological calls)
    weights.skewness = 0.10;      % Distribution shape measures
    weights.kurtosis = 0.10;
    weights.spreadRatio = 0.03;   % Minor contributions
    weights.asymmetry = 0.02;
    
    % Compute weighted composite score
    extremeScore = weights.variance * scores.variance + ...
                   weights.stdDev * scores.stdDev + ...
                   weights.dynRange * scores.dynRange + ...
                   weights.skewness * scores.skewness + ...
                   weights.kurtosis * scores.kurtosis + ...
                   weights.spreadRatio * scores.spreadRatio + ...
                   weights.asymmetry * scores.asymmetry;
    
    % THRESHOLDING
    % Tune based on validation data and operational requirements:
    %   - Lower threshold (0.7-0.8): More sensitive, increased false positives
    %   - Higher threshold (1.2-1.5): More specific, increased false negatives
    extremeThreshold = 0.75; % Conservative starting point for validation
    
    isExtreme = extremeScore > extremeThreshold;
    
    % DIAGNOSTIC OUTPUT
    % Store comprehensive results for analysis, validation, and threshold tuning
    diagnostics = struct();
    diagnostics.variance = envVariance;
    diagnostics.stdDev = envdBStd;
    diagnostics.dynamicRange = dynamicRange;
    diagnostics.skewness = envSkewness;
    diagnostics.kurtosis = envKurtosis;
    diagnostics.spreadRatio = spreadRatio;
    diagnostics.meanMedianDiff = meanMedianDiff;
    diagnostics.scores = scores;     % Individual component scores
    diagnostics.weights = weights;   % Weighting coefficients
    
    % % Display detailed breakdown for real-time analysis and debugging
    % fprintf('\tVariance: %.2f dB² (score: %.2f)\n', envVariance, scores.variance)
    % fprintf('\tStandard Deviation: %.2f dB (score: %.2f)\n', envdBStd, scores.stdDev)
    % fprintf('\tDynamic Range: %.2f dB (score: %.2f)\n', dynamicRange, scores.dynRange)
    % fprintf('\tSkewness: %.2f (score: %.2f)\n', envSkewness, scores.skewness)
    % fprintf('\tKurtosis: %.2f (score: %.2f)\n', envKurtosis, scores.kurtosis)
    % fprintf('\tSpread Ratio: %.2f (score: %.2f)\n', spreadRatio, scores.spreadRatio)
    % fprintf('\tMean-Median Diff: %.2f dB (score: %.2f)\n', meanMedianDiff, scores.asymmetry)
    % fprintf('\t>>> EXTREME SCORE: %.3f (threshold: %.1f) -> %s\n', ...
    %         extremeScore, extremeThreshold, iif(isExtreme, 'EXTREME EVENT DETECTED', 'Normal'))
end

function result = iif(condition, trueValue, falseValue)
    % Inline conditional function (MATLAB doesn't have ternary operator)
    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end