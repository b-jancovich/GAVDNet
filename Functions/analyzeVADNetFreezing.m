function analyzeVADNetFreezing(net)
    % Function to analyze freezing status of layers in a pretrained VADNet network
    %
    % Ben Jancovich, 2024
    % Centre for Marine Science and Innovation
    % School of Biological, Earth and Environmental Sciences
    % University of New South Wales, Sydney, Australia
    %
    
    % Get learnable parameters table
    learnables = net.Learnables;
    
    % Initialize counters and storage
    frozenLayers = {};
    unfrozenLayers = {};
    partiallyFrozenLayers = {};
    frozenParamCount = 0;
    totalParamCount = 0;
    
    % Use containers.Map which allows any string as key (including those with periods)
    layerStatus = containers.Map();
    
    % First identify all unique layer names
    uniqueLayers = unique(learnables.Layer);
    for i = 1:length(uniqueLayers)
        layerName = uniqueLayers{i};
        layerStatus(layerName) = struct('frozen', {{}}, 'unfrozen', {{}});
    end
    
    % Check each learnable parameter
    for i = 1:height(learnables)
        layerName = learnables.Layer{i};
        paramName = learnables.Parameter{i};
        paramValue = learnables.Value{i};
        
        % Count parameters
        numParams = numel(paramValue);
        totalParamCount = totalParamCount + numParams;
        
        % Get learning rate factor
        learnRate = getLearnRateFactor(net, layerName, paramName);
        isFrozen = (learnRate == 0);
        
        if isFrozen
            frozenParamCount = frozenParamCount + numParams;
            status = layerStatus(layerName);
            status.frozen{end+1} = paramName;
            layerStatus(layerName) = status;
        else
            status = layerStatus(layerName);
            status.unfrozen{end+1} = paramName;
            layerStatus(layerName) = status;
        end
    end
    
    % Determine overall status for each layer
    for i = 1:numel(uniqueLayers)
        layerName = uniqueLayers{i};
        status = layerStatus(layerName);
        
        if isempty(status.unfrozen)
            frozenLayers{end+1} = layerName;
        elseif isempty(status.frozen)
            unfrozenLayers{end+1} = layerName;
        else
            partiallyFrozenLayers{end+1} = layerName;
        end
    end
    
    % Generate report
    fprintf('VADNet Learnable Parameter Analysis\n');
    fprintf('=======================\n\n');
    
    fprintf('Summary:\n');
    fprintf('- Total layers with learnable parameters: %d\n', numel(uniqueLayers));
    fprintf('- Frozen layers: %d\n', numel(frozenLayers));
    fprintf('- Unfrozen layers: %d\n', numel(unfrozenLayers));
    fprintf('- Partially frozen layers: %d\n', numel(partiallyFrozenLayers));
    fprintf('- Total parameters: %d\n', totalParamCount);
    fprintf('- Frozen parameters: %d (%.2f%%)\n', frozenParamCount, 100 * frozenParamCount / totalParamCount);
    fprintf('- Unfrozen parameters: %d (%.2f%%)\n\n', ...
        totalParamCount - frozenParamCount, 100 * (totalParamCount - frozenParamCount) / totalParamCount);
    
    % Display unfrozen layers
    if ~isempty(unfrozenLayers)
        fprintf('Completely unfrozen layers:\n');
        for i = 1:numel(unfrozenLayers)
            fprintf('- %s\n', unfrozenLayers{i});
        end
        fprintf('\n');
    else
        fprintf('No completely unfrozen layers found.\n\n');
    end
    
    % Display partially frozen layers
    if ~isempty(partiallyFrozenLayers)
        fprintf('Partially frozen layers:\n');
        for i = 1:numel(partiallyFrozenLayers)
            layerName = partiallyFrozenLayers{i};
            status = layerStatus(layerName);
            
            fprintf('- %s\n', layerName);
            fprintf('  * Frozen parameters: %s\n', strjoin(status.frozen, ', '));
            fprintf('  * Unfrozen parameters: %s\n', strjoin(status.unfrozen, ', '));
        end
        fprintf('\n');
    end
    
    % Display completely frozen layers (often numerous, so just count)
    if ~isempty(frozenLayers)
        fprintf('Completely frozen layers: %d layers\n', numel(frozenLayers));
        if numel(frozenLayers) <= 10
            fprintf('Frozen layer names: %s\n', strjoin(frozenLayers, ', '));
        else
            fprintf('First 10 frozen layers: %s, ...\n', strjoin(frozenLayers(1:10), ', '));
        end
    end
end