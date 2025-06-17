function pcenSpect = spectPCEN(spect, hopLen, fsTarget, options)
%spectPCEN Apply Per-Channel Energy Normalization to mel spectrogram
%   This function applies PCEN (Per-Channel Energy Normalization) to a
%   linear magnitude mel spectrogram. PCEN combines temporal integration,
%   adaptive gain control, and dynamic range compression to enhance
%   transient events while suppressing stationary background noise.
%
%   NOTE: Default values are replicated from the librosa implementation,
%   which is intended for human speech processing. Librosa documentaion
%   states these values are only valid if the raw audio values are in the 
%   interval [-2^31; 2^31-1] and not to [-1, 1]. The default values do not
%   appear to work well for deep sea hydrophone data.
%
%   Inputs:
%   spect     = Linear magnitude mel spectrogram (N_bands x N_frames)
%   hopLen    = Hop length in samples for discretization time step
%   fsTarget  = Target sampling rate (Hz)
%   T         = Time constant for temporal integration (seconds) 
%               aka. time_constant (optional, default = 0.4)
%   alpha     = Exponent for adaptive gain control (0 < alpha < 1) 
%               aka. gain (optional, default = 0.98)
%   epsilon   = Soft threshold for AGC (small positive value) 
%               aka. eps (optional, default = 1e-6)
%   delta     = Bias for dynamic range compression (delta > 1) 
%               aka. bias (optional, default = 2)
%   r         = Exponent for dynamic range compression (0 < r < 1) 
%               aka. power (optional, default = 0.5)
%
%   Output:
%   pcenSpect = PCEN-processed spectrogram (same size as input)
%
%   NOTE: The delta parameter can be calculated based on target signal 
%       bandwidth using the function 'delta = calculatePCENDelta(bandwidth)'.
%
%   Parameter Tuning Guide:
%   Larger values of T: 
%       Longer time constant → more temporal smoothing → slower adaptation to changes
%       Background estimate changes slowly
%       Better at suppressing steady-state noise
%       May miss rapid transients or fail to adapt to quick background changes
%   Smaller values of T: 
%       Shorter time constant → less temporal smoothing → faster adaptation
%       Background estimate tracks changes quickly
%       Better at preserving rapid transients
%       May be more sensitive to short-term noise fluctuations
%   Larger values of alpha: 
%       Stronger gain control
%       More aggressive suppression of stationary background components
%       Greater emphasis on transient events relative to background
%       Risk of over-suppressing signals that are only slightly above background
%   Smaller values of alpha: 
%       Weaker gain control
%       Less suppression of background components
%       More conservative enhancement of transients
%       May not sufficiently emphasize weak transients
%   Larger values of r: 
%       Weaker compression → preserves more dynamic range
%       Maintains larger differences between loud and quiet components
%       More linear response
%       At r = 1, no compression occurs
%   Smaller values of r: 
%       Stronger compression → reduces dynamic range more
%       Compresses differences between loud and quiet components
%       More logarithmic-like response
%       At r → 0, approaches logarithmic compression
%       Better for emphasizing weak signals
%   Larger values of delta:
%       Shifts the entire compression characteristic
%       The delta^r subtraction term ensures PCEN = 0 when G = 0
%       Higher values can affect the shape of the compression curve
%       In underwater acoustics, higher values at lower frequencies 
%       account for higher ambient noise levels
%   Smaller values of delta: 
%       Different compression curve shape
%       Must be > 1 for proper PCEN behavior
%       Lower values may be appropriate for cleaner acoustic environments
%   Larger values of epsilon: 
%       More numerical stability but potential impact on very quiet signals
%       Provides a "floor" below which gain control becomes less effective
%       May prevent proper processing of very weak transients
%   Smaller values of epsilon: 
%       Less impact on quiet signals but potential numerical issues
%       Allows processing of weaker signals
%       Risk of numerical instability with very small background estimates
%
%   Recommended starting values:
%   - Low-Frequency Marine Mammal Applications (Jancovich, 2025):
%       T=60s, alpha=0.95, r=0.25, epsilon=1e-10, delta=calculatePCENDelta(bandwidth)
%
%   - Birdsong detection applications (Lostanlen et al., 2018)
%       T=60ms, alpha=0.8, r=0.25, epsilon=1e-6, delta=10
%
%   - Indoor, human Speech applications (Wang et al., 2017):
%       T=400ms, alpha=0.98, r=0.5, epsilon=1e−6, delta=2;
%
% Summary:
%   - T determines how the background is estimated over time
%   - alpha and epsilon control how strongly signals are normalized against this background
%   - delta and r apply compression to the normalized result
%   - For enhancing weak transients in noisy environments: use longer T, higher alpha, lower r, and frequency-appropriate delta values.
%   - For preserving signal fidelity with less processing: use shorter T, lower alpha, higher r, and smaller delta values.

%   References:
%   Lostanlen, V., Salamon, J., Cartwright, M., McFee, B., Farnsworth, A., 
%   Kelling, S., & Bello, J. P. (2018). Per-channel energy normalization: 
%   Why and how. IEEE Signal Processing Letters, 26(1), 39-43.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

arguments
    spect {validateattributes(spect,{'single','double'},{'nonempty','2d','real','finite'},'spectPCEN','spect')}
    hopLen {validateattributes(hopLen,{'single','double'},{'nonempty','scalar','real','finite','positive'},'spectPCEN','hopLen')}
    fsTarget {validateattributes(fsTarget,{'single','double'},{'nonempty','scalar','real','finite','positive'},'spectPCEN','fsTarget')}
    options.T {validateattributes(options.T,{'single','double'},{'nonempty','scalar','real','finite','positive'},'spectPCEN','T')} = 0.4
    options.alpha {validateattributes(options.alpha,{'single','double'},{'nonempty','scalar','real','finite','>',0,'<',1},'spectPCEN','alpha')} = 0.98
    options.epsilon {validateattributes(options.epsilon,{'single','double'},{'nonempty','scalar','real','finite','positive'},'spectPCEN','epsilon')} = 1e-6
    options.delta {validateattributes(options.delta,{'single','double'},{'nonempty','scalar','real','finite','>',1},'spectPCEN','delta')} = 2
    options.r {validateattributes(options.r,{'single','double'},{'nonempty','scalar','real','finite','>',0,'<',1},'spectPCEN','r')} = 0.5
end

% Unpack options
T = options.T;
alpha = options.alpha;
epsilon = options.epsilon;
delta = options.delta;
r = options.r;

% Calculate IIR filter coefficient using LibROSA method
% Reference: librosa.pcen documentation
T_normalized = T * fsTarget / hopLen;
b = (sqrt(1 + 4*T_normalized^2) - 1) / (2*T_normalized^2);

% Validate b is in valid range
assert(b > 0 && b < 1, 'Computed filter coefficient b is out of valid range [0,1]');

% Temporal Integration using first-order IIR filter
% M(t,f) = (1-b)*M(t-1,f) + b*E(t,f)
M = temporalIntegration(spect, b);

% Adaptive Gain Control
% G(t,f) = E(t,f) / (epsilon + M(t,f))^alpha
G = spect ./ ((epsilon + M).^alpha);

% Dynamic Range Compression
% PCEN(t,f) = (G(t,f) + delta)^r - delta^r
pcenSpect = (G + delta).^r - delta^r;

end

%% Helper Function 

function M = temporalIntegration(E, b)
%TEMPORALINTEGRATION Apply first-order IIR temporal integration filter
%   Implements the temporal smoothing filter for PCEN using LibROSA convention:
%   M(t,f) = (1-b)*M(t-1,f) + b*E(t,f)

[numBands, numFrames] = size(E);

% Initialize output array
M = zeros(numBands, numFrames, like=E);

% Initialize first frame
M(:, 1) = E(:, 1);

% Apply IIR filter across time dimension for each frequency band
for t = 2:numFrames
    M(:, t) = (1 - b) * M(:, t - 1) + b * E(:, t);
end
end
