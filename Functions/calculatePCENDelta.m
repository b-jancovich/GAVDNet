function delta = calculatePCENDelta(bandwidth)
%CALCULATEDELTA Calculate frequency-dependent delta parameter for underwater PCEN
%   This function computes the PCEN bias parameter delta based on the
%   empirical underwater acoustic formula from Mohebbi-Kalkhoran et al. (2024).
%   The formula accounts for the exponential decay of ocean ambient noise
%   with frequency.
%
%   Input:
%   bandwidth = [fmin, fmax] frequency range of interest (Hz)
%
%   Output:
%   delta     = PCEN bias parameter (scalar)
%
%   Formula: δ = 2 + 5*exp(-a*f) + 5*exp(-b*f)
%   where a = 0.003, b = 0.0001, f = center frequency
%
%   Reference:
%   Mohebbi-Kalkhoran, H., Makris, N. C., & Ratilal, P. (2024). 
%   Real-time detection, bearing estimation, and whale species vocalization 
%   classification from passive underwater acoustic array data. 
%   IEEE Sensors Journal, 24(22), 37432-37444.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
% Validate bandwidth ordering
assert(bandwidth(2) > bandwidth(1), 'bandwidth(2) must be greater than bandwidth(1)');

% Empirically determined constants for underwater acoustics
% From Mohebbi-Kalkhoran et al. (2024) Gulf of Maine experiment
a = 0.003;
b = 0.0001;

% Calculate center frequency of bandwidth
f_center = mean(bandwidth);

% Apply frequency-dependent delta formula
% Higher delta values at lower frequencies account for higher ambient noise
delta = 2 + 5*exp(-a*f_center) + 5*exp(-b*f_center);

end