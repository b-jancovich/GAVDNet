function threshold = currentFailureThreshold(currentChainIdx, primaryThreshold)
% CURRENTFAILURETHRESHOLD Per-device consecutive-failure threshold before
% triggering a fallback down the GPU chain.
%
% The primary GPU (chain idx 1) gets the user-configurable threshold; any
% non-primary GPU gets a single shot before being dropped, per the eGPU
% policy "RTX 4090 -> T550 -> CPU, T550 fails on first file -> CPU". A
% single-device chain (as used by each dual-GPU worker) therefore falls to
% CPU on the first failure when its device is non-primary, and after
% primaryThreshold failures when it is the primary.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
    if currentChainIdx == 1
        threshold = primaryThreshold;
    else
        threshold = 1;
    end
end
