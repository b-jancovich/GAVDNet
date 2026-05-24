% View randomly sampled detections from the new GAVDNet detector.
%
% For each year in NEW_YEARS, picks N_PER_YEAR random detections from the
% postprocessed results file and displays a spectrogram of the audio
% segment around each detection. Click on the figure to advance to the
% next; press Q or Esc (or close the figure) to quit.
%
% Spectrogram settings (window, overlap, FFT length) and context padding
% are configurable at the top of the script.
%
% Ben Jancovich, 2026
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

clear; clc;

%% ===== USER CONFIG =====
% Spectrogram parameters
SPEC_WINDOW_SAMPLES = 100;        % window length, samples
SPEC_OVERLAP_PCT = 95;         % overlap, percent of window length
SPEC_NFFT = 2048;       % FFT length

% Resampling
fs_target = 250; % Resample to this sampling frequency, if not there already (Hz)

% Sampling and display
N_PER_YEAR        = 5;         % random detections per year
NEW_YEARS         = 2019:2025;  % years to sample from
PAD_SECONDS       = 20;         % context padding before/after detection
FREQ_DISPLAY_LIM  = [10 60];     % y-axis limits in Hz ([] for full)
COLORMAP_NAME     = 'default';    % spectrogram colormap
SPECTROGRAM_DYNAMIC_RANGE = 60; % dB

% Paths
newResultsDir = ['C:\Users\z5439673\OneDrive - UNSW\H0419778\' ...
    'GAVDNet_DGS_Detections_2000_to_2025\-10 to 10 single exemplar exclude chorus'];
audioRoot     = 'E:\Diego Garcia South 3Ch';   % per-year subfolders inside

%% ===== DO NOT MODIFY BELOW =====

RNG_SEED = 42;
if ~isempty(RNG_SEED)
    rng(RNG_SEED);
end

overlapSamples = round(SPEC_WINDOW_SAMPLES * SPEC_OVERLAP_PCT / 100);

%% Build Filter

order = 4;  % Filter order
Rp = 0.1; % Max ripple in passband (dB)
Rs = 60; % Stopband Attenuation (dB)
Wn(1) = FREQ_DISPLAY_LIM(1)/(fs_target/2);
Wn(2) = FREQ_DISPLAY_LIM(2)/(fs_target/2);
[B, A] = ellip(order, Rp, Rs, Wn, 'band', 'ctf');

%% Build the queue of detections to view
queue = struct('year', {}, 'fileName', {}, 'fileFs', {}, ...
    'eventSampleStart', {}, 'eventSampleEnd', {}, ...
    'eventStartTime', {}, 'confidence', {});

for k = 1:numel(NEW_YEARS)
    yr = NEW_YEARS(k);
    matPath = fullfile(newResultsDir, ...
        sprintf('detector_results_postprocessed_%d.mat', yr));
    if ~isfile(matPath)
        warning('Missing results file for %d: %s', yr, matPath);
        continue
    end

    finfo = dir(matPath);
    fprintf('Year %d: loading %.1f MB ...\n', yr, finfo.bytes / 1e6);
    tLoad = tic;
    S = load(matPath, 'results');
    fprintf('Year %d: load took %.1f s\n', yr, toc(tLoad));

    if ~isfield(S, 'results') || isempty(S.results)
        warning('Year %d: no detections', yr);
        continue
    end

    nAvail = numel(S.results);
    nPick  = min(N_PER_YEAR, nAvail);
    pickIdx = randperm(nAvail, nPick);

    % Vectorised field extraction first, then index - avoids the per-element
    % O(N^2) struct-array indexing that bit us before.
    fileNames        = {S.results.fileName};
    fileFsAll        = [S.results.fileFs];
    sampleStartsAll  = [S.results.eventSampleStart];
    sampleEndsAll    = [S.results.eventSampleEnd];
    startTimesAll    = [S.results.eventStartTime];
    confidencesAll   = [S.results.confidence];

    for j = 1:numel(pickIdx)
        idx = pickIdx(j);
        e = numel(queue) + 1;
        queue(e).year             = yr;
        queue(e).fileName         = fileNames{idx};
        queue(e).fileFs           = fileFsAll(idx);
        queue(e).eventSampleStart = sampleStartsAll(idx);
        queue(e).eventSampleEnd   = sampleEndsAll(idx);
        queue(e).eventStartTime   = startTimesAll(idx);
        queue(e).confidence       = confidencesAll(idx);
    end

    fprintf('Year %d: queued %d/%d detections\n\n', yr, nPick, nAvail);
    clear S fileNames fileFsAll sampleStartsAll sampleEndsAll ...
        startTimesAll confidencesAll
end

if isempty(queue)
    error('No detections queued - check paths and config.');
end

fprintf('\nLoaded %d detections total. Click figure to advance, Q/Esc to quit.\n\n', ...
    numel(queue));

%% Display loop
fig = figure('Name', 'GAVDNet detection viewer', 'Color', 'w', ...
    'NumberTitle', 'off', 'Position', [100 100 1200 600], ...
    'WindowKeyPressFcn', @(src, evt) setappdata(src, 'lastKey', evt.Key), ...
    'CloseRequestFcn', @(src, ~) setappdata(src, 'userQuit', true));
setappdata(fig, 'lastKey', '');
setappdata(fig, 'userQuit', false);

for i = 1:numel(queue)
    if ~isvalid(fig) || getappdata(fig, 'userQuit')
        fprintf('User closed figure, exiting.\n');
        if isvalid(fig); delete(fig); end
        return
    end

    d = queue(i);
    audioPath = fullfile(audioRoot, num2str(d.year), d.fileName);
    if ~isfile(audioPath)
        warning('[%d/%d] Missing audio: %s - skipping', i, numel(queue), audioPath);
        continue
    end

    try
        info = audioinfo(audioPath);
    catch ME
        warning('[%d/%d] audioinfo failed for %s: %s', i, numel(queue), ...
            audioPath, ME.message);
        continue
    end
    nTotal = info.TotalSamples;
    fs     = info.SampleRate;
    if fs ~= d.fileFs
        warning('fs mismatch: stored=%g, file=%g (using file)', d.fileFs, fs);
    end

    padSamp = round(PAD_SECONDS * fs);
    s0 = max(1, d.eventSampleStart - padSamp);
    s1 = min(nTotal, d.eventSampleEnd + padSamp);

    try
        x = audioread(audioPath, [s0 s1]);
    catch ME
        warning('[%d/%d] audioread failed: %s', i, numel(queue), ME.message);
        continue
    end

    % If fs of input differs from target Fs, resample audio
    if fs ~= fs_target
        [p, q] = rat(fs_target/fs, 1e-9);
        x = cast(resample(double(x(:)), p, q), like=x);
    else
        x = x(:);
    end

    % Sum stereo to mono
    if size(x, 2) > 1
        x = mean(x, 2);
    end

    % Normalize, DC Filter
    x = x - mean(x, 'omitnan');
    peakVal = max(abs(x));
    if peakVal > 0
        x = x / peakVal;
    end

    % High Pass Filter
    x = ctffilt(B, A, x);

    % Normalize, DC Filter
    x = x - mean(x, 'omitnan');
    peakVal = max(abs(x));
    if peakVal > 0
        x = x / peakVal;
    end

    % Spectrogram
    [SS, F, T] = spectrogram(x, SPEC_WINDOW_SAMPLES, overlapSamples, SPEC_NFFT, fs);
    Sdb = 20 * log10(abs(SS) + eps);
    cMax = max(Sdb(:));
    cMin = cMax- SPECTROGRAM_DYNAMIC_RANGE;
    
    clf(fig);
    imagesc(T, F, Sdb);
    set(gca, 'YDir', 'normal');
    xlabel('Time (s, relative to segment start)');
    ylabel('Frequency (Hz)');
    cb = colorbar; cb.Label.String = 'Power (dB)';
    colormap(fig, COLORMAP_NAME);
    if ~isempty(FREQ_DISPLAY_LIM)
        ylim(FREQ_DISPLAY_LIM);
    end
    clim([cMin, cMax])

    % Mark detection boundaries
    evtStartSec = (d.eventSampleStart - s0) / fs;
    evtEndSec   = (d.eventSampleEnd   - s0) / fs;
    hold on
    xline(evtStartSec, 'w--', 'LineWidth', 1.5);
    xline(evtEndSec,   'w--', 'LineWidth', 1.5);
    hold off

    title(sprintf(['[%d/%d]  Year %d  |  %s  |  %s  |  conf = %.3f\n' ...
                   'win = %d samp, overlap = %d%%, NFFT = %d, fs = %g Hz, pad = %g s'], ...
        i, numel(queue), d.year, d.fileName, ...
        string(d.eventStartTime, 'yyyy-MM-dd HH:mm:ss'), d.confidence, ...
        SPEC_WINDOW_SAMPLES, SPEC_OVERLAP_PCT, SPEC_NFFT, fs, PAD_SECONDS), ...
        'Interpreter', 'none', 'FontSize', 10);

    drawnow

    fprintf('[%d/%d] Click figure to advance (Q/Esc to quit) ...\n', i, numel(queue));

    setappdata(fig, 'lastKey', '');
    waitforbuttonpress;
    if ~isvalid(fig)
        fprintf('Figure closed, exiting.\n');
        return
    end
    key = getappdata(fig, 'lastKey');
    if any(strcmpi(key, {'q', 'escape'}))
        fprintf('User quit.\n');
        delete(fig);
        return
    end
end

fprintf('\nAll %d detections shown.\n', numel(queue));
if isvalid(fig)
    delete(fig);
end
