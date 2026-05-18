clear
clc

sequenceDuration_s = 300;
fs = 250;
samples = sequenceDuration_s * fs;
dt = 1/fs;
t = 0:dt:sequenceDuration_s-dt;
period_s = 45;
depth_dB = 20;

chorusParams.num_calls_in_chorus = 100;
chorusParams.chorus_calls_level_range = [-3, 3];
chorusParams.chorus_call_overlap_range = [0.85, 0.99];

chorusParams.chorus_modulation_period_s = period_s;
chorusParams.chorus_sequence_level_range = depth_dB;
cleanFilesStruct = dir('E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10\clean_signals\*.wav');

cleanFilesList = cell(length(cleanFilesStruct), 1);
for i = 1:length(cleanFilesStruct)
    cleanFilesList{i} = fullfile(cleanFilesStruct(i).folder, cleanFilesStruct(i).name); 
end

env = generateSlowEnvelope(samples, fs, period_s, depth_dB);

chorus = generateChorus(cleanFilesList, fs, sequenceDuration_s, chorusParams);

[S_chorus, F_chorus, T_chorus] = spectrogram(chorus, 250, 240, 2048, fs);

S_chorus = mag2db(abs(S_chorus));

dyn_range = 60;

figure(1)
tiledlayout(1, 3)
nexttile
plot(t', 20*log10(env))
xlabel('Time (s)')
ylabel('Amplitude (dB)')
title('Amp Modulation Envelope')

nexttile
plot(t', chorus)
xlabel('Time (s)')
ylabel('Amplitude')
title('Modulated Chorus Signal')

nexttile
imagesc(T_chorus, F_chorus, S_chorus)
xlabel('Time (s)')
ylabel('Frequency (Hz)')
title('Chorus Spectrogram')
clim([max(S_chorus(:))-dyn_range, max(S_chorus(:))])
set(gca, "YDir", "normal")

