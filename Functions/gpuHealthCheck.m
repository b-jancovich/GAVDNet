function [status, metrics] = gpuHealthCheck(deviceID, throughputMinGBps)
%GPUHEALTHCHECK Sentinel-computation and throughput probe on a GPU device.
%
% [STATUS, METRICS] = gpuHealthCheck(DEVICEID, THROUGHPUTMINGBPS) runs two
% short checks against the GPU at DEVICEID and returns a summary status
% plus a metrics struct. Intended for periodic invocation during long
% inference runs on an external GPU, where the Thunderbolt link can
% degrade or drop under sustained load.
%
% The probe has two parts:
%   1. Sentinel computation - a 512x512 single-precision matrix multiply on
%      the GPU, verifying the device is responsive and the result is
%      numerically sane (no NaN/Inf).
%   2. Throughput test - 3 iterations of 200 MB host<->device transfers,
%      reporting mean / std / min bandwidth in GB/s for each direction.
%
% Inputs:
%   deviceID          - GPU device index (1..gpuDeviceCount).
%   throughputMinGBps - Minimum acceptable mean H<->D throughput in GB/s.
%                       Optional, default 1.5. Healthy TB3 baseline is
%                       ~2.5 GB/s; below ~1.5 the link is likely in
%                       error-recovery on a degraded physical layer.
%
% Outputs:
%   status  - One of:
%               'healthy'  : sentinel OK and min(H<->D mean) >= threshold
%               'marginal' : sentinel OK but min(H<->D mean) < threshold
%               'failed'   : sentinel errored, returned NaN/Inf, or the
%                            device is unreachable
%   metrics - struct with fields:
%               sentinelOK            (logical)
%               sentinelErr           (char, '' if sentinelOK)
%               h2dMean, h2dStd, h2dMin   (GB/s, NaN on failure)
%               d2hMean, d2hStd, d2hMin   (GB/s, NaN on failure)
%               availableMemoryBytes  (double)
%               deviceName            (char)
%
% Reference: Utilitiy Scripts/test_egpu_throughput.m uses the same
% transfer-and-time pattern with 20 iterations of 800 MB; this is a
% cheaper variant designed for per-batch invocation.
%
% This function is novel - written to harden GAVDNet inference runs on an
% unreliable external GPU (Thunderbolt eGPU).
%
% Ben Jancovich, 2026
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

arguments
    deviceID (1,1) double {mustBePositive, mustBeInteger}
    throughputMinGBps (1,1) double {mustBeNonnegative} = 1.5
end

% Failure-state defaults (overwritten as the probe progresses)
metrics = struct( ...
    'sentinelOK', false, ...
    'sentinelErr', '', ...
    'h2dMean', NaN, 'h2dStd', NaN, 'h2dMin', NaN, ...
    'd2hMean', NaN, 'd2hStd', NaN, 'd2hMin', NaN, ...
    'availableMemoryBytes', NaN, ...
    'deviceName', '');

% Select the device and snapshot identity / free memory. If gpuDevice()
% itself throws the device is unreachable; return 'failed' immediately.
try
    g = gpuDevice(deviceID);
    metrics.deviceName = char(g.Name);
    metrics.availableMemoryBytes = g.AvailableMemory;
catch ME
    metrics.sentinelErr = sprintf('gpuDevice(%d) failed: %s', deviceID, ME.message);
    status = 'failed';
    return
end

% Sentinel matrix multiply
try
    A = gpuArray.rand(512, 'single');
    B = gpuArray.rand(512, 'single');
    C = A * B;
    wait(g);
    Ccpu = gather(C);
    if any(isnan(Ccpu(:))) || any(isinf(Ccpu(:)))
        metrics.sentinelErr = 'Sentinel matrix multiply produced NaN/Inf.';
        status = 'failed';
        return
    end
    metrics.sentinelOK = true;
catch ME
    metrics.sentinelErr = sprintf('Sentinel compute failed: %s', ME.message);
    status = 'failed';
    return
end

% Throughput probe - 3 iterations of 200 MB H<->D transfers
try
    N = 5e7;          % 50M singles -> 200 MB
    nReps = 3;
    bytes = 4*N;

    h2d = zeros(nReps, 1);
    d2h = zeros(nReps, 1);
    data = rand(N, 1, 'single');

    for k = 1:nReps
        wait(g);
        t0 = tic;
        d = gpuArray(data);
        wait(g);
        h2d(k) = bytes / toc(t0) / 1e9;

        wait(g);
        t0 = tic;
        out = gather(d); %#ok<NASGU>
        wait(g);
        d2h(k) = bytes / toc(t0) / 1e9;
    end

    metrics.h2dMean = mean(h2d);
    metrics.h2dStd  = std(h2d);
    metrics.h2dMin  = min(h2d);
    metrics.d2hMean = mean(d2h);
    metrics.d2hStd  = std(d2h);
    metrics.d2hMin  = min(d2h);
catch ME
    % Throughput test errored after a successful sentinel. Inference
    % requires H<->D transfers, so this is effectively a failed device.
    metrics.sentinelErr = sprintf('Throughput test failed: %s', ME.message);
    status = 'failed';
    return
end

% Classify from the slower of the two transfer directions
meanThroughput = min(metrics.h2dMean, metrics.d2hMean);
if meanThroughput >= throughputMinGBps
    status = 'healthy';
else
    status = 'marginal';
end
end
