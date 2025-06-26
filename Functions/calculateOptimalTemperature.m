function [optimalTemperature, calibratedConfidences] = calculateOptimalTemperature(confidenceScores, trueLabels)
% Implements temperature scaling calibration method from Guo et al. (2017)
% to find the optimal temperature parameter that minimizes negative 
% log-likelihood on validation data.
%
% Temperature scaling applies a single scalar parameter T to the logits:
% calibrated_prob = sigmoid(logit/T) for binary classification
% where logit = log(p/(1-p)) and p is the original confidence score.
%
% Inputs:
%   confidenceScores - Vector of confidence scores from detector (0 to 1)
%   trueLabels       - Vector of true binary labels (0 or 1)
%
% Outputs:
%   optimalTemperature    - Scalar temperature parameter T > 0
%   calibratedConfidences - Vector of calibrated confidence scores
%
% Reference:
%   Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). 
%   On Calibration of Modern Neural Networks (arXiv:1706.04599). 
%   arXiv. https://doi.org/10.48550/arXiv.1706.04599
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

%% Input validation
if length(confidenceScores) ~= length(trueLabels)
    error('confidenceScores and trueLabels must have the same length');
end

if any(confidenceScores < 0) || any(confidenceScores > 1)
    error('confidenceScores must be between 0 and 1');
end

if any(trueLabels ~= 0 & trueLabels ~= 1)
    error('trueLabels must be binary (0 or 1)');
end

% Convert to column vectors
confidenceScores = confidenceScores(:);
trueLabels = trueLabels(:);

%% Handle edge cases in confidence scores
% Clip extreme values to avoid numerical issues when converting to logits
epsilon = 1e-7;
confidenceScores = max(epsilon, min(1-epsilon, confidenceScores));

%% Convert confidence scores to logits
% logit = log(p/(1-p)) where p is the confidence score
logits = log(confidenceScores ./ (1 - confidenceScores));

%% Define negative log-likelihood objective function
    function nll = negativeLogLikelihood(T)
        % Apply temperature scaling: calibrated_prob = sigmoid(logit/T)
        calibratedProbs = 1 ./ (1 + exp(-logits ./ T));
        
        % Compute negative log-likelihood
        % NLL = -sum(y*log(p) + (1-y)*log(1-p))
        logProbs = log(calibratedProbs);
        logOneMinusProbs = log(1 - calibratedProbs);
        
        nll = -sum(trueLabels .* logProbs + (1 - trueLabels) .* logOneMinusProbs);
        
        % Handle potential numerical issues
        if ~isfinite(nll)
            nll = inf;
        end
    end

%% Optimize temperature parameter
% Use golden section search for robust optimization
% Temperature must be positive, typically in range [0.1, 10]
options = optimset('Display', 'off', 'TolX', 1e-6);
[optimalTemperature, ~] = fminbnd(@negativeLogLikelihood, 0.01, 10, options);

%% Calculate calibrated confidences using optimal temperature
calibratedConfidences = 1 ./ (1 + exp(-logits ./ optimalTemperature));

%% Display results
fprintf('Temperature scaling calibration completed:\n');
fprintf('  Optimal temperature: %.4f\n', optimalTemperature);
fprintf('  Original confidence range: [%.4f, %.4f]\n', ...
    min(confidenceScores), max(confidenceScores));
fprintf('  Calibrated confidence range: [%.4f, %.4f]\n', ...
    min(calibratedConfidences), max(calibratedConfidences));

end