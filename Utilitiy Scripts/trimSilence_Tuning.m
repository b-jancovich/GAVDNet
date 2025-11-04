clear
clc

% Path to one of your target call exemplars:
audioPath = "D:\DGS_Chagos_Exemplars\U1 & U2\Denoised\detectionAudio_150846_26-Oct-2017_17_50_52_24.627544_RXDENOISED.wav";
[audio, fs] = audioread(audioPath);

% Make the last n seconds of the signal really quiet:
quietSeconds = 7;
AmpScalingFactor = 0.1;
audio(end-quietSeconds*fs:end) = audio(end-quietSeconds*fs:end) .* ampScalingFactor;

% Trim params - We are tuning these:
trim_threshold_ratio = 0.2;
trim_window_size = 25;

% Spectrogram params - Set these to the same you plan to use for training
windowDur = 0.85;
hopDur = 0.05;
fsTarget = 250;
bandwidth = [10, 50];
FFTLen = 4096;

% Init STFT parameters (ensuring even values)
windowLen = 2 * round((windowDur * fsTarget) / 2);
hopLen = 2 * round((hopDur * fsTarget) / 2);
nOverlap = windowLen - hopLen;

% Resample file
audio = audio ./ max(abs(audio));
[p, q] = rat(fsTarget/fs, 1e-9);
audioUntrimmed = resample(audio(:), p, q);

% Trim silence
audioTrimmed = trimSilence(audioUntrimmed, trim_threshold_ratio, trim_window_size);

% Build time vectors for plotting original and trimmed audio
dt = 1/fsTarget;
duration = length(audio) / fsTarget;
durationTrimmed = length(audioTrimmed) / fsTarget;
audioT = 0:dt:duration-dt;
audioTrimmedT = 0:dt:durationTrimmed-dt;

% Build the mel filterbank
[filterBank, Fc, ~] = designAuditoryFilterBank(fsTarget, ...
    FFTLength = FFTLen, ...
    Normalization = "none", ...
    OneSided = true, ...
    FrequencyRange = bandwidth, ...
    FilterBankDesignDomain = "warped", ...
    FrequencyScale = "mel", ...
    NumBands = 40);

% Compute the spectrograms
[s, f, t] = spectrogram(audioUntrimmed, windowLen, nOverlap, FFTLen, fsTarget, 'yaxis');
[sTrimmed, fTrimmed, tTrimmed] = spectrogram(audioTrimmed, windowLen, nOverlap, FFTLen, fsTarget, 'yaxis');

% Convert to power
s = abs(s).^2;
sTrimmed = abs(sTrimmed).^2;

% Apply Mel filterbank
s = filterBank * s;
sTrimmed = filterBank * sTrimmed;

% Convert to dB
s = 10*log10(max(s, 1e-10));
sTrimmed = 10*log10(max(sTrimmed, 1e-10));

figure(3)
tiledlayout(2,2)

nexttile
plot(audioT, audio)
ylabel('Amplitude')
xlabel('Time (s)')
xlim([0, 20])

nexttile
imagesc(t, Fc, s)
set(gca, 'YDir', 'normal')
ylabel('Frequency (Hz)')
xlabel('Bin Center Time Index (s)')
xlim([0, 20])
colorbar

nexttile
plot(audioTrimmedT, audioTrimmed)
ylabel('Amplitude')
xlabel('Time (s)')
xlim([0, 20])

nexttile
imagesc(tTrimmed, Fc, sTrimmed)
set(gca, 'YDir', 'normal')
ylabel('Frequency (Hz)')
xlabel('Bin Center Time Index (s)')
xlim([0, 20])
colorbar