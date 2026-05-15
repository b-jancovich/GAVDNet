function [audioData, sampleRate, errMsg] = audioReadWithRetry(filePath, retryWaits, finalCooldown)
%AUDIOREADWITHRETRY Read an audio file with progressive retry on transient I/O.
%
% [AUDIODATA, SAMPLERATE, ERRMSG] = audioReadWithRetry(FILEPATH, RETRYWAITS, FINALCOOLDOWN)
% reads an audio file with retry-on-failure to tolerate transient I/O errors
% (external drive blips, OneDrive sync races, antivirus locks, etc.).
%
% On failure the function pauses for RETRYWAITS(k) seconds before retry k.
% If every retry fails it pauses for FINALCOOLDOWN seconds before returning
% the failure result, giving the underlying I/O time to recover before the
% caller attempts the next file.
%
% Total attempts = 1 (initial) + numel(RETRYWAITS) (retries). With the
% defaults this is 4 attempts in total (1, 3, 5 min apart) plus a 5 min
% cooldown on permanent failure, ~14 min worst case per file.
%
% Inputs:
%   filePath       - Full path to the audio file (char or string).
%   retryWaits     - Vector of wait times in seconds between failed
%                    attempts. Length sets the number of retries.
%                    Default [60, 180, 300] (three retries, 1/3/5 min).
%   finalCooldown  - Seconds to pause after all retries fail, before
%                    returning. Default 300 (5 min).
%
% Outputs:
%   audioData   - Audio samples as 'single'. Empty single([]) on failure.
%   sampleRate  - Sample rate (Hz). NaN on failure.
%   errMsg      - '' on success, last error message string on failure.
%
% This function is novel - written specifically to harden GAVDNet
% inference against transient I/O failures observed in long production
% runs over external drive storage.
%
% Ben Jancovich, 2026
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

arguments
    filePath {mustBeTextScalar}
    retryWaits (1,:) double {mustBeNonnegative} = [60, 180, 300]
    finalCooldown (1,1) double {mustBeNonnegative} = 300
end

% Failure-state defaults (overwritten on success)
audioData = single([]);
sampleRate = NaN;
errMsg = '';

numRetries = numel(retryWaits);
totalAttempts = 1 + numRetries;

for attempt = 1:totalAttempts
    try
        [audioData, sampleRate] = audioread(filePath);
        audioData = single(audioData);
        errMsg = '';
        return
    catch ME
        errMsg = ME.message;
        if attempt < totalAttempts
            % Still have retries left
            waitDur = retryWaits(attempt);
            warning('audioReadWithRetry:transientFailure', ...
                'Read attempt %d/%d failed for %s (%s). Waiting %g s before retry...', ...
                attempt, totalAttempts, char(filePath), ME.message, waitDur);
            pause(waitDur)
        else
            % All retries exhausted - cool down before returning failure
            warning('audioReadWithRetry:allRetriesFailed', ...
                'All %d attempts failed for %s (%s). Cooling down %g s before returning.', ...
                totalAttempts, char(filePath), ME.message, finalCooldown);
            pause(finalCooldown)
        end
    end
end
% audioData / sampleRate retain their failure-state initial values.
end
