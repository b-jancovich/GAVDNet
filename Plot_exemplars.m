% Make exemplars figure for adjudicator app.

% Set path to exemplars
exemplarsPath = 'D:\DGS_Chagos_Exemplars\U1 & U2';
% exemplarsPath = 'D:\SORP_BmAntZ_exemplars';

% Regex pattern to extract date from filename 
datePattern = '\d{2}-[A-Za-z]{3}-\d{4}'; % (CTBT data)
% datePattern = '\d{4}_\d{1,2}_\d{1,2}'; % (SORP data)

% Freq axis zoom range (Hz)
fRange = [10, 60];

% Window size (seconds)
win = 0.8;

% Window overlap (percent)
ovlp = 95;

% FFT Length
fftSize = 4096;

% Dynamic Range (dB)
dynRange = 80;

% Get file list
fileList = dir(fullfile(exemplarsPath, '*.wav'));

% Initialize Figure
figure(1)
tiledlayout('flow')

% Run audio processing loop
for i = 1:length(fileList)
    % Load audio
    [audio, Fs] = audioread(fullfile(exemplarsPath, fileList(i).name));

    % Convert spectrogram params to samples
    winSamps = round(win * Fs);
    ovlpSamps = round(winSamps * (ovlp/100));

    % Compute Spectrogram
    [s, f, t] = spectrogram(audio, winSamps, ovlpSamps, fftSize, Fs, 'yaxis');

    % Convert complex spectra to power, dB
    s = 10*log10(abs(s).^2 + eps);

    % Normalize spectrogram
    s = s - max(s, [], 'all');

    % Set dynamic range
    cMax = max(s, [], 'all');
    cMin = cMax - dynRange;

    % Draw axes
    nexttile
    imagesc(t, f, s)
    set(gca, "YDir", "normal")
    ylabel('Frequency (Hz)')
    xlabel('Time (s)')
    ylim(fRange)
    grid on
    c = colorbar;
    c.Label.String = 'Power (dB)';
    clim([cMin, cMax])
    dateStr = regexp(fileList(i).name, datePattern, 'match', 'once');
    title(dateStr, Interpreter="none")
end
sgtitle('Chagos pygmy blue whale Song Exemplars')

