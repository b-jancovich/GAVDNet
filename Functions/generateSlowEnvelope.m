function env = generateSlowEnvelope(N, fs, period_s, depth_dB)
% generateSlowEnvelope Smooth random low-frequency amplitude envelope.
%
% Returns an N-sample envelope normalised to the interval
% [10^(-depth_dB/20), 1]. The waveform is produced by Gaussian-smoothing
% white Gaussian noise so the dominant temporal scale of the result is
% approximately period_s seconds.
%
% Note: an earlier draft used cumsum(randn) (a random walk) before
% smoothing. That produced a 1/f^2 process whose energy is concentrated at
% DC, giving effective periods of hundreds of seconds rather than period_s.
% Plain smoothed white noise is the correct construction; see
% Chorus Prototyping/proto1_slow_envelope.m for the validation that flagged
% the original bug.
%
% This is a bespoke alternative to customAudioAugmenter/simulateRandomTransLoss,
% whose density/sigmoid-gating math is built for occasional dropouts rather
% than continuous gentle modulation of a chorus track.
%
% Inputs:
%   N         - number of samples in the output (samples)
%   fs        - sample rate (Hz)
%   period_s  - approximate fundamental period of the modulation (seconds)
%   depth_dB  - peak-to-trough modulation depth (dB, positive value)
%
% Outputs:
%   env       - N-by-1 column vector in [10^(-depth_dB/20), 1]
%
% Example:
%   env = generateSlowEnvelope(1800*250, 250, 10, 6);
%
% Ben Jancovich, 2026
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

    arguments
        N        (1,1) double {mustBePositive, mustBeInteger}
        fs       (1,1) double {mustBePositive}
        period_s (1,1) double {mustBePositive}
        depth_dB (1,1) double {mustBeNonnegative}
    end

    % Gaussian smoothing window in samples. Choose width = fs*period_s so
    % the dominant temporal scale of the smoothed noise is ~period_s.
    win = max(3, round(fs * period_s));

    % Smooth white Gaussian noise. smoothdata with 'gaussian' applies a
    % Gaussian-weighted moving average of full width "win". Output PSD is
    % the original (flat) noise spectrum multiplied by |G(f)|^2, where G is
    % the smoothing kernel's Fourier transform, giving low-pass content
    % with effective cutoff ~1/period_s.
    e = smoothdata(randn(N, 1), 'gaussian', win);

    % Normalise to [0, 1]. eps guards the degenerate case max==min.
    e = (e - min(e)) / max(eps, max(e) - min(e));

    % Map to [floorLin, 1]: floorLin = 10^(-depth_dB/20).
    floorLin = 10 ^ (-depth_dB / 20);
    env = floorLin + (1 - floorLin) * e;
end
