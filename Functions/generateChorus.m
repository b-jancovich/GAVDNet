function chorus = generateChorus(cleanFilesList, fs, sequenceDuration_s, chorusParams)
% generateChorus Build one synthetic chorus track for one training sequence.
%
% Assembles a chorus by (1) sampling N already-augmented exemplar calls from
% the same library used to populate discrete-call placements, (2) amplitude
% jittering them in dB, (3) overlap-summing them into a "chorus base",
% (4) looping that base with ramp-overlap (loops stride by baseLen-mean(lens)
% so the natural energy ramps at each end of the base sum to approximately
% constant), (5) applying short-decay reverb to add a sense of propagation
% depth and to smear sample-level repetition across loops, and (6)
% multiplying by a slow time-varying amplitude envelope. The result is
% RMS-normalised to 1.0; the caller is responsible for scaling to a
% target SNR vs noise.
%
% The mask convention in constructMultiCallNoisySequences is not touched by
% this function: chorus contributions are always negative in the training
% target because the mask is set only inside the call-placement loop. This
% keeps the integration footprint to a single block in the sequence builder.
%
% Inputs:
%   cleanFilesList     - cell array of absolute paths to clean exemplar
%                        .wav files (e.g. ads_cleanSignals.Files)
%   fs                 - sample rate of the call exemplars and target output
%                        (Hz). All exemplars must share this rate.
%   sequenceDuration_s - target chorus duration in seconds
%   chorusParams       - struct with fields:
%       .num_calls_in_chorus       (scalar int)       number of exemplars summed into base
%       .chorus_calls_level_range  ([min, max] dB)    amplitude jitter per call
%       .chorus_call_overlap_range ([min, max] frac)  fractional overlap of consecutive calls
%       .chorus_modulation_period_s (scalar s)        approx period of slow envelope
%       .chorus_sequence_level_range (scalar dB)      depth of slow envelope (positive)
%
% Outputs:
%   chorus - column vector, length round(sequenceDuration_s*fs), RMS = 1.0
%
% Notes:
%   - Each exemplar already carries the full customAudioAugmenter pipeline
%     (Doppler, reverb, transmission loss, frequency shift, distortion,
%     time stretch, end trim). Summing N of them yields a heterogeneous
%     chorus reminiscent of real distant-conspecific aggregates.
%   - randperm guarantees N distinct exemplars when numel(cleanFilesList)
%     >= N. Falls back to sampling with replacement (and warns once) only
%     in the unrealistic case N > library size.
%   - The chorus base is built by sequential layered overlap, not random
%     placement, so the spacing is deterministic given the sampled
%     overlap fractions.
%
% Ben Jancovich, 2026
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%


arguments
    cleanFilesList     (:,1) cell
    fs                 (1,1) double {mustBePositive}
    sequenceDuration_s (1,1) double {mustBePositive}
    chorusParams       (1,1) struct
end

persistent warnedOnReplacement reverb lastFs

    %% 1. Sample N exemplar paths (distinct when possible)
    N = chorusParams.num_calls_in_chorus;
    nFiles = numel(cleanFilesList);
    if N <= nFiles
        idx = randperm(nFiles, N);
    else
        if isempty(warnedOnReplacement)
            warning('generateChorus:sampleWithReplacement', ...
                'num_calls_in_chorus (%d) exceeds library size (%d); sampling with replacement.', N, nFiles);
            warnedOnReplacement = true;
        end
        idx = randi(nFiles, N, 1);
    end

    %% 2. Read each exemplar, validate fs, force mono column
    calls = cell(N, 1);
    lens  = zeros(N, 1);
    for k = 1:N
        [a, fk] = audioread(cleanFilesList{idx(k)});
        if fk ~= fs
            error('generateChorus:fsMismatch', ...
                'Exemplar %s has fs=%d but expected %d.', cleanFilesList{idx(k)}, fk, fs);
        end
        if size(a, 2) > 1
            a = mean(a, 2);
        end
        calls{k} = a(:);
        lens(k)  = numel(a);
    end

    %% 3. Amplitude jitter (dB -> linear)
    lvl = chorusParams.chorus_calls_level_range;
    dB  = lvl(1) + (lvl(2) - lvl(1)) * rand(N, 1);
    g   = 10 .^ (dB / 20);
    for k = 1:N
        calls{k} = calls{k} * g(k);
    end

    %% 4. Layered overlap-sum into a chorus base
    o_rng = chorus_call_overlap_range_clamped(chorusParams.chorus_call_overlap_range);
    starts = zeros(N, 1);
    for k = 2:N
        ov = o_rng(1) + (o_rng(2) - o_rng(1)) * rand;
        % Step = lens(k-1)*(1-ov) samples; never below 1 to ensure progress.
        step = max(1, round(lens(k-1) * (1 - ov)));
        starts(k) = starts(k-1) + step;
    end
    % Base must span the latest end-of-call, which is not necessarily the
    % last call's end if an earlier call is longer.
    baseLen = max(starts + lens);
    base = zeros(baseLen, 1);
    for k = 1:N
        i0 = starts(k) + 1;
        i1 = i0 + lens(k) - 1;
        base(i0:i1) = base(i0:i1) + calls{k};
    end

    %% 5. Loop with ramp-overlap to target length
    % The base has natural energy ramps spanning ~mean(lens) at each end
    % (the active-call count grows linearly from zero over one
    % mean-call duration, then plateaus, then decays back to zero over
    % the same duration). We stride by baseLen - mean(lens) so the
    % ramp-down of loop N overlaps with the ramp-up of loop N+1, and
    % their roughly mirror-image energy profiles sum to approximately
    % constant. No equal-power fade weights are needed; the ramps ARE
    % the fades. A "warmup" loop is prepended and then discarded so the
    % returned output begins at plateau energy rather than at loop 1's
    % ramp-up.
    M = round(sequenceDuration_s * fs);
    chorus = loop_with_ramp_overlap(base, M, mean(lens));

    %% 6. Reverb
    % Cache the reverberator System object across calls (creation is
    % expensive). Recreate only if fs changes. HighCutFrequency is set
    % just below Nyquist; exact-Nyquist is on the edge of where
    % reverberator's internal filter is well-defined.
    if isempty(reverb) || ~isequal(fs, lastFs)
        reverb = reverberator("PreDelay", 0, ...
            "HighCutFrequency", 0.95 * fs / 2, ...
            DecayFactor = 0.01, ...
            Diffusion = 0.5, ...
            WetDryMix = 0.25, ...
            SampleRate = fs);
        lastFs = fs;
    end
    % reverberator retains its impulse-response state between calls.
    % Reset before each chorus so the previous sequence's tail does not
    % bleed into the first samples of this one.
    reset(reverb);
    chorus_verby_stereo = reverb(chorus);
    chorus = chorus_verby_stereo(:, 1);   % wet output is stereo; keep left channel only

    %% 7. Slow amplitude envelope
    env = generateSlowEnvelope(M, fs, ...
        chorusParams.chorus_modulation_period_s, ...
        chorusParams.chorus_sequence_level_range);
    chorus = chorus .* env;

    %% 8. RMS normalise to 1.0 (caller applies SNR-derived gain)
    r = rms(chorus);
    if r > eps
        chorus = chorus / r;
    end
end

% -------------------------------------------------------------------------
function out = loop_with_ramp_overlap(base, M, rampLen)
% Tile "base" to length M by additively overlapping each loop with the
% next by rampLen samples. The chorus base has natural energy ramps
% spanning rampLen samples at each end (active-call count rises
% linearly from zero and falls back to zero). Aligning ramp-down of one
% loop with ramp-up of the next causes the two roughly mirror-image
% energy profiles to sum to approximately constant - no equal-power
% fade weighting required; the natural ramps act as the fades.
%
% A warmup loop is prepended and then discarded so the returned output
% begins at plateau energy. Without the warmup, samples 1..rampLen
% would contain only loop 1's own ramp-up and be quiet.
    baseLen = numel(base);
    if baseLen >= M
        out = base(1:M);
        return
    end

    % rampLen must be strictly less than baseLen for stride >= 1, and
    % much less than baseLen for the overlap to be useful (typically
    % rampLen ~ baseLen/3 to baseLen/2 across the supported overlap
    % range). Clamp defensively.
    rampLen = max(1, min(round(rampLen), baseLen - 1));
    stride  = baseLen - rampLen;

    % Allocate an extended buffer covering [-baseLen, M+baseLen]; output
    % index = position + baseLen + 1 so position 0 -> index baseLen+1.
    bufLen = M + 2 * baseLen;
    buf    = zeros(bufLen, 1);

    % Iterate loops starting from position -baseLen (warmup), advancing
    % by stride, until we cover the output region.
    pos = -baseLen;
    while pos < M
        i0 = pos + baseLen + 1;
        i1 = i0 + baseLen - 1;
        if i0 >= 1 && i1 <= bufLen
            buf(i0:i1) = buf(i0:i1) + base;
        end
        pos = pos + stride;
    end

    % Discard the warmup baseLen and the trailing extra baseLen.
    out = buf(baseLen + 1 : baseLen + M);
end

% -------------------------------------------------------------------------
function r = chorus_call_overlap_range_clamped(r_in)
% Defensive clamp: overlap fractions must lie in (0, 1). Hard upper bound
% at 0.999 to keep the step size finite.
    r = r_in;
    r(1) = max(1e-3, min(0.999, r_in(1)));
    r(2) = max(1e-3, min(0.999, r_in(2)));
    if r(2) < r(1)
        r = [r(2), r(1)];
    end
end
