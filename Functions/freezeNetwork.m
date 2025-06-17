function net = freezeNetwork(net)
% Unfreeze all learnable parameters in a dlnetwork
%
%   Freezes all learnable parameters in a dlnetwork by setting their learn
%   rate factors to zero.
%
%   Input:
%       net - dlnetwork object
%
%   Output:
%       net - dlnetwork object with all parameters frozen
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Validate input
if ~isa(net, 'dlnetwork')
    error('freezeNetwork:InvalidInput', ...
        'Input must be a dlnetwork object');
end

% Get the learnables table
learnables = net.Learnables;

% Get the number of learnable parameters
numLearnables = size(learnables, 1);

% Loop through all learnable parameters and set their learn rate factor to 0
for i = 1:numLearnables
    layerName = learnables.Layer(i);
    parameterName = learnables.Parameter(i);
    
    % Set the learn rate factor to 0 for this parameter
    net = setLearnRateFactor(net, layerName, parameterName, 0);
end

end