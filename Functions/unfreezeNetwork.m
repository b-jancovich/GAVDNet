function net = unfreezeNetwork(net, learnRateFactor)
% Unfreeze all learnable parameters in a dlnetwork
%
%   Unfreezes all learnable parameters in the network by setting their 
%   learn rate factors to 1, or optionally, some other value.
%
%   Inputs:
%       net - dlnetwork object
%       learnRateFactor - (optional) Learn rate factor to apply to all
%                         parameters. Default is 1.
%
%   Output:
%       net - dlnetwork object with all parameters unfrozen
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Set default learn rate factor if not provided
if nargin < 2
    learnRateFactor = 1;
end

% Validate inputs
if ~isa(net, 'dlnetwork')
    error('unfreezeNetwork:InvalidInput', ...
        'Input must be a dlnetwork object');
end

if ~isscalar(learnRateFactor) || ~isnumeric(learnRateFactor) || learnRateFactor < 0
    error('unfreezeNetwork:InvalidLearnRateFactor', ...
        'Learn rate factor must be a non-negative scalar');
end

% Get the learnables table
learnables = net.Learnables;

% Get the number of learnable parameters
numLearnables = size(learnables, 1);

% Loop through all learnable parameters and set their learn rate factor
for i = 1:numLearnables
    layerName = learnables.Layer(i);
    parameterName = learnables.Parameter(i);
    
    % Set the learn rate factor for this parameter
    net = setLearnRateFactor(net, layerName, parameterName, learnRateFactor);
end

end