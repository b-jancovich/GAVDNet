function calibratedProbs = temperatureScaleProbabilities(originalProbs, temperature)
% Apply post-hoc probability correction at inference time based on
% calibrated temperature value.
%
% Based on [1].
% 
% References:
%   [1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). 
%       On Calibration of Modern Neural Networks (arXiv:1706.04599). 
%       arXiv. https://doi.org/10.48550/arXiv.1706.04599
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Convert probabilities to logits
logits = log(originalProbs ./ (1 - originalProbs + eps));

% Apply temperature scaling
calibratedLogits = logits / temperature;

% Convert back to probabilities
calibratedProbs = 1 ./ (1 + exp(-calibratedLogits));
end