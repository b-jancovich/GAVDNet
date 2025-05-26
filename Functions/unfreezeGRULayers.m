function net = unfreezeGRULayers(net, unfreezeOption, newLearnRateFactor)
    % Function to unfreeze GRU layers in a pretrained VADNet network
    %
    % Inputs:
    %   net - dlnetwork object to modify
    %   unfreezeOption - Optional string specifying which layers to unfreeze:
    %                   'gru1' - Unfreeze only GRU1 layers (default)
    %                   'all'  - Unfreeze both GRU1 and GRU2 layers
    %   newLearnRateFactor - Optional learn rate factor to apply (default: 1)
    %
    % Output:
    %   net - Modified dlnetwork object with unfrozen GRU layers
    %
    % The function sets a non-zero learn rate factor for the previously frozen
    % GRU layers to enable fine-tuning.
    %
    % Ben Jancovich, 2024
    % Centre for Marine Science and Innovation
    % School of Biological, Earth and Environmental Sciences
    % University of New South Wales, Sydney, Australia
    %

    % Validate input
    if ~isa(net, 'dlnetwork')
        error('Input must be a dlnetwork object.');
    end
    
    % Handle default arguments
    if nargin < 2 || isempty(unfreezeOption)
        unfreezeOption = 'gru1';
    end
    
    if nargin < 3 || isempty(newLearnRateFactor)
        newLearnRateFactor = 1;
    end
    
    if newLearnRateFactor <= 0
        error('Learn rate factor must be positive.');
    end
    
    % Get learnable parameters table
    learnables = net.Learnables;
    
    % Define the layers to unfreeze based on the option
    if strcmpi(unfreezeOption, 'gru1')
        layersToUnfreeze = {'gru1.forward', 'gru1.reverse'};
        fprintf('Unfreezing GRU1 layers (forward and reverse)...\n');
    elseif strcmpi(unfreezeOption, 'all')
        layersToUnfreeze = {'gru1.forward', 'gru1.reverse', 'gru2.forward', 'gru2.reverse'};
        fprintf('Unfreezing all GRU layers (GRU1 and GRU2, forward and reverse)...\n');
    else
        error('Invalid unfreezeOption. Use ''gru1'' or ''all''.');
    end
    
    % Get unique layer names in the network
    uniqueLayerNames = unique(learnables.Layer);
    
    % Check if the specified frozen layers exist in the network
    foundLayers = false(size(layersToUnfreeze));
    for i = 1:length(layersToUnfreeze)
        foundLayers(i) = any(strcmp(uniqueLayerNames, layersToUnfreeze{i}));
        if ~foundLayers(i)
            warning('Layer ''%s'' not found in network''s Learnables table.', layersToUnfreeze{i});
        end
    end
    
    if ~any(foundLayers)
        error('None of the specified layers were found in the network''s Learnables table.');
    end
    
    % Only process layers that were found
    layersToUnfreeze = layersToUnfreeze(foundLayers);
    
    % Count the total number of parameters unfrozen
    unfrozenParamCount = 0;
    
    % Loop through the learnables table and unfreeze parameters
    for i = 1:height(learnables)
        layerName = learnables.Layer{i};
        
        % Check if this layer is one we want to unfreeze
        if ismember(layerName, layersToUnfreeze)
            paramName = learnables.Parameter{i};
            
            % Get current learn rate factor
            currentFactor = getLearnRateFactor(net, layerName, paramName);
            
            % Only update if currently frozen (factor = 0)
            if currentFactor == 0
                % Set learn rate factor for this parameter to non-zero value
                net = setLearnRateFactor(net, layerName, paramName, newLearnRateFactor);
                unfrozenParamCount = unfrozenParamCount + numel(learnables.Value{i});
                fprintf('  Unfroze %s.%s with learn rate factor %.2f\n', layerName, paramName, newLearnRateFactor);
            else
                fprintf('  Skipped %s.%s (already unfrozen with factor %.2f)\n', layerName, paramName, currentFactor);
            end
        end
    end
    
    fprintf('Total parameters unfrozen: %d\n', unfrozenParamCount);
    
    % Verify that the layers are now unfrozen by analyzing the network again
    fprintf('\nVerifying unfreeze operation...\n');
    stillFrozen = checkFrozenLayers(net, layersToUnfreeze);
    
    if isempty(stillFrozen)
        fprintf('All specified layers are now unfrozen.\n\n');
    else
        warning('Some layers remain frozen: %s', strjoin(stillFrozen, ', '));
    end
end

function stillFrozen = checkFrozenLayers(net, layersToCheck)
    % Helper function to check if any of the specified layers are still frozen
    
    % Get learnable parameters table
    learnables = net.Learnables;
    
    % Initialize result
    stillFrozen = {};
    
    % Check each layer
    for i = 1:length(layersToCheck)
        layerName = layersToCheck{i};
        
        % Find all parameters for this layer
        layerParams = learnables(strcmp(learnables.Layer, layerName), :);
        
        % Check if all parameters have zero learn rate factor
        allFrozen = true;
        for j = 1:height(layerParams)
            paramName = layerParams.Parameter{j};
            if getLearnRateFactor(net, layerName, paramName) > 0
                allFrozen = false;
                break;
            end
        end
        
        if allFrozen && ~isempty(layerParams)
            stillFrozen{end+1} = layerName;
        end
    end
end