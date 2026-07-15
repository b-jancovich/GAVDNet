function [useGPU, gpuDeviceID, bytesAvailable, newChainIdx, switchedTo] = ...
        stepGpuFallback(currentChainIdx, gpuChain, cpuMemoryBytes)
% STEPGPUFALLBACK Advance one step down the GPU fallback chain after the
% current device is judged to be failing (e.g. consecGpuFailures >= threshold,
% or the year-start health check returned 'failed').
%
% currentChainIdx convention:
%   1..N  - index into gpuChain (entries sorted by TotalMemory desc, so 1
%           is the primary / most-capable GPU)
%   0     - currently on CPU; no further fallback is possible
%
% Transitions:
%   1..N-1  -> next GPU in chain (try to reset and activate it; if that
%              also throws, skip straight to CPU)
%   N       -> CPU (chain exhausted)
%   0       -> remains CPU
%
% Each dual-GPU worker passes a SINGLE-device chain (just its own GPU), so a
% worker never falls back onto the other worker's GPU - it drops straight to
% CPU, honouring the "one CUDA context per GPU" eGPU constraint.
%
% Returns the new (useGPU, gpuDeviceID, bytesAvailable, newChainIdx) plus a
% human-readable description of where we ended up, for logging.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

    if currentChainIdx == 0
        useGPU = false;
        gpuDeviceID = 0;
        bytesAvailable = cpuMemoryBytes;
        newChainIdx = 0;
        switchedTo = 'CPU (already on CPU; no further fallback)';
        return
    end

    % Best-effort release of the failing device's context. The device may
    % already be unresponsive, which is exactly the situation that triggered
    % the fallback, so swallow any error here.
    try
        wait(gpuDevice(gpuChain(currentChainIdx).deviceID));
        reset(gpuDevice(gpuChain(currentChainIdx).deviceID));
    catch
        % Ignore - device unresponsive
    end

    if currentChainIdx < numel(gpuChain)
        % Step to next GPU in chain
        newChainIdx = currentChainIdx + 1;
        candidateID = gpuChain(newChainIdx).deviceID;
        try
            g = gpuDevice(candidateID);
            reset(g);
            useGPU = true;
            gpuDeviceID = candidateID;
            bytesAvailable = g.AvailableMemory;
            switchedTo = sprintf('GPU %d ("%s", %.1f GB free)', ...
                gpuDeviceID, char(g.Name), bytesAvailable / 1e9);
        catch ME
            warning(['Could not activate fallback GPU %d: %s. ' ...
                'Skipping to CPU.'], candidateID, ME.message)
            useGPU = false;
            gpuDeviceID = 0;
            bytesAvailable = cpuMemoryBytes;
            newChainIdx = 0;
            switchedTo = 'CPU (fallback GPU also unavailable)';
        end
    else
        % Chain exhausted - drop to CPU
        useGPU = false;
        gpuDeviceID = 0;
        bytesAvailable = cpuMemoryBytes;
        newChainIdx = 0;
        switchedTo = sprintf('CPU (%.1f GB available)', cpuMemoryBytes / 1e9);
    end
end
