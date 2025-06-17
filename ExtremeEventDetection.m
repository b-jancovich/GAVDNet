% Compute statistics to identify any regions of the signal that are
% extremely high amplitude, relative to the rest of the signal.
clear  
close all
clc

% File which contains calls + extremely high-power transient noise
audioPath{1} = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_150605-120000_calls+extremelyHighPowerNoise.wav";
comment{1} = 'This file contains both target calls and relatively short duration extreme events.';
extremeGT{1} = true;

% File which contains high dynamic range calls, but no extreme events 
audioPath{2} = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-160000_HighDynamicRangeCalls.wav";
comment{2} = 'This file contains high dynamic range target calls, but no extreme events.';
extremeGT{2} = false;

% An unremarkable file, with no extreme events
audioPath{3} = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-200000_TypicalLotsOfCalls.wav";
comment{3} = 'This file is unremarkable. It contains target calls, and no extreme events.';
extremeGT{3} = false;

% File with an extreme, very long duration event (earthquake or iceberg calving) - also contains calls
audioPath{4} = "C:\Users\z5439673\OneDrive - UNSW\H0419778\GAVDNet_Testing\Chagos_DGS\H08S1_071102-000000_EarthquakeDynamicRangeTest.wav";
comment{4} = 'This file contains target calls, and variety of short and long duration extreme events.';
extremeGT{4} = true;

% Filter Parameters
n = 8;
Rp = 0.1;
Rs = 90;

% Analyse files
for i = 1:length(audioPath)
    % Read audio
    [audioRaw, fs] = audioread(audioPath{i});
    
    % Remove DC Offset
    audioMeasure = audioRaw - mean(audioRaw);

    % Design high pass filter
    nyq = fs/2;
    Wp = 5 / nyq;
    [b,a] = ellip(n, Rp, Rs, Wp, "high", "ctf");
    
    % Filter audio
    audioMeasure = ctffilt(b, a, audioMeasure);
    
    % Normalize to max=1
    audioMeasure = audioMeasure ./ max(abs(audioMeasure));
    
    % Compute RMS envelope of audio with window sizes of 1s and 60s
    [audioEnvShort, ~] = envelope(audioMeasure, 1*fs, 'rms');
    [audioEnvLong, ~] = envelope(audioMeasure, 60*fs, 'rms');
    
    [~, name, ~] = fileparts(audioPath{i});
    fprintf('File Name: %s\n', name)
    fprintf('File Comment: %s\n', comment{i})
    
    % Compute extreme event indicators 
    fprintf('\nLong-window envelope Analysis:\n')
    [isExtremeLong, scoreLong, diagLong] = detectExtremeEvents(audioEnvShort);
    
    % Compute extreme event indicators 
    fprintf('\nShort-window envelope Analysis:\n')
    [isExtremeShort, scoreShort, diagShort] = detectExtremeEvents(audioEnvLong);

    % Store results for validation and threshold tuning
    results(i).filename = name;
    results(i).hasExtremeEvents = extremeGT{i};
    results(i).longWindow.isExtreme = isExtremeLong;
    results(i).longWindow.score = scoreLong;
    results(i).longWindow.diagnostics = diagLong;
    results(i).shortWindow.isExtreme = isExtremeShort;
    results(i).shortWindow.score = scoreShort;
    results(i).shortWindow.diagnostics = diagShort;
    
    fprintf('\n--- SUMMARY for %s ---\n', name)
    fprintf('Expected extreme events: %s\n', iif(results(i).hasExtremeEvents, 'YES', 'NO'))
    fprintf('Long window detection: %s (score: %.3f)\n', iif(isExtremeLong, 'EXTREME', 'Normal'), scoreLong)
    fprintf('Short window detection: %s (score: %.3f)\n', iif(isExtremeShort, 'EXTREME', 'Normal'), scoreShort)
    fprintf('Overall assessment: %s\n', iif(isExtremeLong || isExtremeShort, 'CONTAINS EXTREME EVENTS', 'Normal file'))
end

% Performance summary
fprintf('\n\n=== DETECTION PERFORMANCE SUMMARY ===\n')
for i = 1:length(results)
    detected = results(i).longWindow.isExtreme || results(i).shortWindow.isExtreme;
    expected = results(i).hasExtremeEvents;
    if detected && expected
        status = 'CORRECT (True Positive)';
    elseif detected && ~expected
        status = 'FALSE POSITIVE';
    elseif ~detected && expected
        status = 'FALSE NEGATIVE';
    else
        status = 'CORRECT (True Negative)';
    end
    fprintf('%s: %s\n', results(i).filename, status)
end

%% Helper Functions


function result = iif(condition, trueValue, falseValue)
    % Inline conditional function (MATLAB doesn't have ternary operator)
    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end
