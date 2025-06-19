function varargout = gpuConfig()
% GPUCONFIG Configures system resources for machine learning tasks
%   [useGPU, deviceID, numWorkers] = gpuConfig() optimizes system resources
%   for machine learning by:
%   1. Detecting and selecting the optimal GPU with maximum available memory
%   2. Configuring CPU parallelism via parallel computing toolbox
%   3. Setting memory management options
%   4. Optimizing numerical precision settings
%
%   Outputs:
%   - useGPU: Boolean flag indicating whether a GPU is available for use
%   - deviceID: ID of the selected GPU device with the most available memory
%   - numWorkers: Number of parallel workers configured
%
%   For optimal performance in deep learning and machine learning applications,
%   this function should be called at the beginning of training/inference scripts.
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Part 1: GPU Configuration
numGPUs = gpuDeviceCount("available");
deviceID = 1; % Default value

% Check if GPU is available
if numGPUs > 0
    % GPU is available
    fprintf('Found %d GPU device(s)\n', numGPUs);
    % Initialize variables to store max memory and corresponding device ID
    maxMemory = 0;
    maxMemoryDeviceID = 1;
    % Loop through each GPU to find the one with maximum available memory
    for i = 1:numGPUs
        % Get current GPU info
        gpuInfo = gpuDevice(i);
        % Get available memory in bytes and convert to GB for display
        availableMemory = gpuInfo.AvailableMemory;
        availableMemoryGB = availableMemory / 1e9;
        fprintf('GPU %d: %s - Available Memory: %.2f GB\n', ...
            i, gpuInfo.Name, availableMemoryGB);
        % Check if this GPU has more available memory
        if availableMemory > maxMemory
            maxMemory = availableMemory;
            maxMemoryDeviceID = i;
        end
    end
    % Select the GPU with the most available memory
    fprintf('Selecting GPU %d with %.2f GB available memory\n', ...
        maxMemoryDeviceID, maxMemory / (1024^3));
    gpu = gpuDevice(maxMemoryDeviceID);

    % Reset GPU to clear memory
    reset(gpu);

    % Enable GPU caching for better performance
    if verLessThan('matlab', '9.8') % R2020a
        % Older versions use different settings
        setenv('CUDA_CACHE_MAXSIZE', '2147483648'); % 2GB cache
    else
        % For newer MATLAB versions, use environment variables if needed
        setenv('CUDA_CACHE_MAXSIZE', '4294967296'); % 4GB cache
    end

    % Set GPU buffer policy to grow as needed (R2019b and newer)
    if ~verLessThan('matlab', '9.7')
        % Use try-catch as this might not be available in all MATLAB versions
        try
            enableCUDABuffer(true);
        catch
            warning('enableCUDABuffer not available in this MATLAB version');
        end
    end

    useGPU = true;
    deviceID = maxMemoryDeviceID;
    disp('Datastores will output to GPU.');
else
    % No GPU available
    disp('No compatible GPUs found. Datastores will output to CPU.');
    useGPU = false;
end

% Part 2: CPU Parallelism Configuration
% Check if Parallel Computing Toolbox is available
numWorkers = 0;
if license('test', 'Distrib_Computing_Toolbox')
    % Get system information
    numCores = feature('numcores');
    % Reserve some cores for system operations
    reservedCores = 1; % Reserve at least 1 core for system operations

    % Calculate optimal number of workers based on system resources
    if useGPU
        % When using GPU, we need fewer CPU workers
        optimalWorkers = max(1, floor(numCores * 0.7) - reservedCores);
    else
        % When CPU-only, use more workers
        optimalWorkers = max(1, numCores - reservedCores);
    end

    % Get the current parallel cluster
    try
        c = parcluster('local');
        maxAllowedWorkers = c.NumWorkers;

        % Respect the maximum allowed workers configuration
        optimalWorkers = min(optimalWorkers, maxAllowedWorkers);

        % Check if parallel pool already exists
        poolObj = gcp('nocreate');
        if isempty(poolObj)
            % Create new parallel pool with optimal workers
            fprintf('Creating parallel pool with %d workers (system has %d cores)...\n', ...
                optimalWorkers, numCores);
            poolObj = parpool('local', optimalWorkers);
            fprintf('Created parallel pool with %d workers\n', optimalWorkers);
        else
            if poolObj.NumWorkers ~= optimalWorkers
                % Only recreate if significantly different to avoid overhead
                if abs(poolObj.NumWorkers - optimalWorkers) > 1
                    % Delete existing pool and create new one with optimal size
                    delete(poolObj);
                    poolObj = parpool('local', optimalWorkers);
                    fprintf('Reconfigured parallel pool with %d workers\n', optimalWorkers);
                else
                    fprintf('Using existing parallel pool with %d workers\n', poolObj.NumWorkers);
                end
            else
                fprintf('Using existing parallel pool with %d workers\n', poolObj.NumWorkers);
            end
        end
        numWorkers = poolObj.NumWorkers;
    catch ME
        warning('GPUCONFIG:ParallelPoolSetupFailed', 'Parallel pool setup encountered an issue: %s', ME.message);
        fprintf('Continuing with default parallel settings\n');

        % Try to get any existing pool
        try
            poolObj = gcp('nocreate');
            if ~isempty(poolObj)
                numWorkers = poolObj.NumWorkers;
                fprintf('Using existing parallel pool with %d workers\n', numWorkers);
            else
                fprintf('Running in single-threaded mode\n');
            end
        catch
            fprintf('Running in single-threaded mode\n');
        end
    end
else
    fprintf('Parallel Computing Toolbox not available. Running in single-threaded mode.\n');
end

% Part 3: Memory Management
% Get system memory information
memInfo = memory;
totalMem = memInfo.MemAvailableAllArrays / (1024^3); % Convert to GB
fprintf('Available system memory: %.2f GB\n', totalMem);

% Set memory options based on available resources
if totalMem > 32
    % Abundant memory - optimize for performance
    memoryOpt = 'performance';
elseif totalMem > 16
    % Moderate memory - balanced
    memoryOpt = 'balanced';
else
    % Limited memory - optimize for memory usage
    memoryOpt = 'memory';
end

% Apply memory management settings
switch memoryOpt
    case 'performance'
        % Performance focused - use more memory for caching
        maxNumCompThreads('automatic');
        if useGPU
            % Set large workspace arrays to go directly to GPU when possible
            arrayLocation = 'gpu';
        else
            arrayLocation = 'cpu';
        end
        fprintf('Memory configuration: Performance mode\n');

    case 'balanced'
        % Balanced approach
        maxNumCompThreads('automatic');
        arrayLocation = 'cpu'; % Safer to use CPU for initial arrays
        fprintf('Memory configuration: Balanced mode\n');

    case 'memory'
        % Memory conservative approach
        maxNumCompThreads(max(1, floor(numCores/2))); % Use fewer threads
        arrayLocation = 'cpu';
        fprintf('Memory configuration: Memory-saving mode\n');
end

% Set tall array preferences for memory efficiency
try
    % Try to set tall preferences if available
    tall.setPreferences('ChunkSize', '64MB', 'DataLocation', arrayLocation);
catch
    % Continue if not available
end

% Part 4: Set numerical precision options
% For ML workloads, half precision can be faster on supported GPUs
if useGPU
    % Check if GPU supports half precision
    try
        halfPrecisionSupported = false;
        gpuInfo = gpuDevice;
        if contains(lower(gpuInfo.Name), {'tesla', 'volta', 'turing', 'ampere', 'hopper', 'ada lovelace', 'rtx'})
            % Modern NVIDIA architectures supporting FP16
            halfPrecisionSupported = true;
        end

        if halfPrecisionSupported
            % Enable fast math mode for machine learning (available in newer MATLAB)
            if ~isMATLABReleaseOlderThan("R2020a") % R2020b or newer
                try
                    setenv('CUDA_FAST_MATH', '1');
                    fprintf('Enabled fast math mode for GPU computations\n');
                catch
                    % Continue if not available
                end
            end
            fprintf('GPU supports half precision (FP16) operations\n');
        else
            fprintf('GPU does not support half precision operations\n');
        end
    catch
        warning('Could not determine GPU precision capabilities');
    end

    % Check and use cudnn if available (for deep learning)
    if exist('cudnnGetVersion', 'file')
        try
            cudnnVer = cudnnGetVersion();
            fprintf('Using cuDNN version: %d\n', cudnnVer);
        catch
            % Continue if not available
        end
    end
end

% Part 5: Validate GPU operation
if useGPU
    try
        % Run a simple test to verify GPU operation
        testResult = validateGPUOperation(deviceID);
        if testResult
            fprintf('GPU validation successful\n');
        else
            warning('GPU validation failed - performance might be affected');
        end
    catch ME
        warning('GPUCONFIG:ValidationFailed', 'GPU validation error: %s - continuing with caution', ME.message);
    end
end

% Display final configuration summary
fprintf('\nSystem Configuration Summary:\n');
fprintf('---------------------------\n');
fprintf('GPU Enabled: %s\n', mat2str(useGPU));
if useGPU
    fprintf('Selected GPU ID: %d\n', deviceID);
end
fprintf('Parallel Workers: %d\n', numWorkers);
fprintf('Memory Mode: %s\n', memoryOpt);
fprintf('---------------------------\n\n');

if useGPU == true
    gpuInfo = gpuDevice(deviceID);
    bytesAvailable = gpuInfo.AvailableMemory;
else
    bytesAvailable = memInfo.MemAvailableAllArrays;
end

if nargout == 2
    varargout{1} = useGPU;
    varargout{2} = deviceID;
elseif nargout == 3
    varargout{1} = useGPU;
    varargout{2} = deviceID;
    varargout{3} = numWorkers;
elseif nargout == 4
    varargout{1} = useGPU;
    varargout{2} = deviceID;
    varargout{3} = numWorkers;
    varargout{4} = bytesAvailable;
else
    error('Invalid output arguments. This function returns 2, 3 or 4 output arguments.')
end
end

% Helper function to enable CUDA buffer (only for newer MATLAB versions)
function enableCUDABuffer(enable)
% This is a helper to enable CUDA buffer growth policy
% This may not be available in all MATLAB versions
try
    if enable
        setenv('CUDA_MALLOC_POLICY', 'growth');
    else
        setenv('CUDA_MALLOC_POLICY', 'static');
    end
catch
    warning('Could not set CUDA buffer policy');
end
end

% Helper function to validate GPU operation
function result = validateGPUOperation(deviceID)
% Perform a simple test to validate GPU is working properly
try
    % Select the device
    gpuDevice(deviceID);

    % Test 1: Simple matrix operation
    A = gpuArray(rand(1000));
    B = gpuArray(rand(1000));
    C = A * B;
    wait(gpuDevice); % Ensure computation completes

    % Test 2: Memory transfer
    D = gather(C);

    % Test 3: Check for NaN/Inf
    if any(isnan(D(:))) || any(isinf(D(:)))
        warning('GPU operation produced NaN or Inf values');
        result = false;
        return;
    end

    result = true;
catch ME
    warning('GPUCONFIG:ValidationFailed', 'GPU validation error: %s', ME.message);
    result = false;
end
end