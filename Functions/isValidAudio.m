function isValid = isValidAudio(audioVector)
% ISVALIDAUDIO Checks if a vector contains valid audio data
%   isValid = ISVALIDAUDIO(audioVector) returns true if audioVector contains
%   valid audio data, false otherwise.
%
%   Invalid audio is defined as:
%   - Vector contains any NaN or Inf values
%   - Vector is empty
%   - Vector contains only zeros
%   - Vector contains only values smaller than realmin(single) in magnitude
%   - Vector contains only a single constant value
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Check if vector is empty
if isempty(audioVector)
    isValid = false;
    return;
end

% Check if vector contains any NaN or Inf values
if any(isnan(audioVector)) || any(isinf(audioVector))
    isValid = false;
    return;
end

% Check if vector contains only zeros
if all(audioVector == 0)
    isValid = false;
    return;
end

% Check if vector contains only values smaller than realmin(single) in magnitude
if all(abs(audioVector) < realmin('single'))
    isValid = false;
    return;
end

% Check if vector contains only a single constant value
if all(audioVector == audioVector(1))
    isValid = false;
    return;
end

% If we've passed all checks, the audio is valid
isValid = true;
end