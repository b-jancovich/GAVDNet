function [audioNoisy, SNRs] = addNoiseToSequences(audioIn, adsNoise, snrRange, sequenceDuration, adsFs)
%ADDNOISETOSEQUENCES Adds noise to each training sequence with random SNR
%
% Inputs:
% audioIn - Cell array of clean training audio sequences
% ads_noise - AudioDatastore containing noise samples
% snrRange - Two-element vector specifying min and max SNR in dB
% sequenceDuration - Duration of each sequence in seconds
% ads_fs - Sample rate of the training data
%
% Outputs:
% audioTrainNoisy - Cell array of noisy training sequences
% snrUsed - Vector of SNR values used for each sequence
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Read all filenames from the AudioDatastore
noiseFileList = adsNoise.Files;
numNoiseFiles = length(noiseFileList);

if numNoiseFiles == 0
    error('No noise files available in the AudioDatastore');
end

% Define output arrays
audioNoisy = cell(size(audioIn));
SNRs = zeros(size(audioIn));
sequenceLength = round(sequenceDuration * adsFs);

% Use parallel for loop
for i = 1:length(audioIn)
    % Read noise from file using worker-specific index
    % This ensures each worker gets different noise
    fileIdx = mod(i-1, numNoiseFiles) + 1;
    localNoiseDS = audioDatastore(noiseFileList{fileIdx});
    [noise, noiseInfo] = readValidNoise(localNoiseDS, fileIdx, numNoiseFiles, noiseFileList);

    % Resample noise if needed
    if adsFs ~= noiseInfo.SampleRate
        noise = resample(double(noise), adsFs, double(noiseInfo.SampleRate));
    end

    % Extend noise with additional files if it's too short
    fileCounter = i + numNoiseFiles;
    while length(noise) < sequenceLength
        nextFileIdx = mod(fileCounter-1, numNoiseFiles) + 1;
        fileCounter = fileCounter + 1;
        localNoiseDS = audioDatastore(noiseFileList{nextFileIdx});
        [more_noise, noiseInfo] = readValidNoise(localNoiseDS, nextFileIdx, numNoiseFiles, noiseFileList);

        % Resample if needed
        if adsFs ~= noiseInfo.SampleRate
            more_noise = resample(double(more_noise), adsFs, double(noiseInfo.SampleRate));
        end
        noise = [noise; more_noise];

        % If we've gone through many files and still don't have enough, recycle
        if fileCounter > i + 2*numNoiseFiles
            numRepeats = ceil(sequenceLength/length(noise));
            noiseExtended = [];
            originalNoise = noise;
            for j = 1:numRepeats
                if mod(j, 2) == 1
                    % Odd repetitions: keep original direction
                    noiseExtended = [noiseExtended; originalNoise];
                else
                    % Even repetitions: reverse direction
                    noiseExtended = [noiseExtended; flipud(originalNoise)];
                end
            end
            noise = noiseExtended;
            break
        end
    end

    % Handle excess noise length
    if length(noise) >= sequenceLength
        trimmedNoise = noise(1:sequenceLength);
    else
        % In case we still don't have enough (shouldn't happen with the recycling)
        trimmedNoise = [noise; zeros(sequenceLength-length(noise), 1)];
    end

    % Generate random SNR
    SNRs(i) = snrRange(1) + (snrRange(2) - snrRange(1)) * rand();

    % If either the audioIn or the trimmed noise are empty/silent/NaN, 
    % don't return audio noisy and continue the loop - this shouldn't
    % happen...
    if isValidAudio(audioIn{i}) == false
        fprintf('Empty audioIn for iteration %g\n', i)
    end
    if isValidAudio(trimmedNoise) == false
        fprintf('Empty noise for iteration %g\n', i)
    end

    % Mix signal and noise
    audioNoisy{i} = mixSNR(audioIn{i}, trimmedNoise, SNRs(i));
end
end

function [validNoise, noiseInfo] = readValidNoise(localNoiseDS, currentIdx, numNoiseFiles, noiseFileList)
%READVALIDNOISE Helper function to read valid noise data
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
% Maximum attempts to find valid noise
maxAttempts = numNoiseFiles;
attemptCount = 0;
fileIdx = currentIdx;

while attemptCount < maxAttempts
    try
        % Try to read the noise file
        [noise, noiseInfo] = read(localNoiseDS);

        % Check if noise is valid (not empty, not all zeros, not all NaNs, not all Infs)
        if isValidAudio(noise) == true
            validNoise = noise;
            return;
        end
    catch
        % If reading fails, continue to the next file
    end

    % Increment attempt counter
    attemptCount = attemptCount + 1;

    % Move to next file in a circular fashion
    fileIdx = mod(fileIdx, numNoiseFiles) + 1;
    localNoiseDS = audioDatastore(noiseFileList{fileIdx});
end

% If we've tried all files and none are valid, throw an error
error('No valid noise files found after trying all available files.');
end

% Has no mechaism to handle failed audio reads from the noise datastore.
% function [audioNoisy, SNRs] = addNoiseToSequences(audioIn, adsNoise, snrRange, sequenceDuration, adsFs)
% %ADDNOISETOSEQUENCES Adds noise to each training sequence with random SNR
% %
% % Inputs:
% % audioIn - Cell array of clean training audio sequences
% % ads_noise - AudioDatastore containing noise samples
% % snrRange - Two-element vector specifying min and max SNR in dB
% % sequenceDuration - Duration of each sequence in seconds
% % ads_fs - Sample rate of the training data
% %
% % Outputs:
% % audioTrainNoisy - Cell array of noisy training sequences
% % snrUsed - Vector of SNR values used for each sequence
% %
% % Ben Jancovich, 2024
% % Centre for Marine Science and Innovation
% % School of Biological, Earth and Environmental Sciences
% % University of New South Wales, Sydney, Australia
% %
%
% % Read all filenames from the AudioDatastore
% noiseFileList = adsNoise.Files;
% numNoiseFiles = length(noiseFileList);
%
% if numNoiseFiles == 0
%     error('No noise files available in the AudioDatastore');
% end
%
% % Define output arrays
% audioNoisy = cell(size(audioIn));
% SNRs = zeros(size(audioIn));
% sequenceLength = round(sequenceDuration * adsFs);
%
% % Use parallel for loop
% parfor i = 1:length(audioIn)
%     % Initialize noise variable for this worker
%     noise = [];
%
%     % Read noise from file using worker-specific index
%     % This ensures each worker gets different noise
%     fileIdx = mod(i-1, numNoiseFiles) + 1;
%     localNoiseDS = audioDatastore(noiseFileList{fileIdx});
%     [noise, noiseInfo] = read(localNoiseDS);
%
%     % Resample noise if needed
%     if adsFs ~= noiseInfo.SampleRate
%         noise = resample(double(noise), adsFs, double(noiseInfo.SampleRate));
%     end
%
%     % Extend noise with additional files if it's too short
%     fileCounter = i + numNoiseFiles;
%     while length(noise) < sequenceLength
%         nextFileIdx = mod(fileCounter-1, numNoiseFiles) + 1;
%         fileCounter = fileCounter + 1;
%
%         localNoiseDS = audioDatastore(noiseFileList{nextFileIdx});
%         [more_noise, noiseInfo] = read(localNoiseDS);
%
%         % Resample if needed
%         if adsFs ~= noiseInfo.SampleRate
%             more_noise = resample(double(more_noise), adsFs, double(noiseInfo.SampleRate));
%         end
%
%         noise = [noise; more_noise];
%
%         % If we've gone through many files and still don't have enough, recycle
%         if fileCounter > i + 2*numNoiseFiles
%             numRepeats = ceil(sequenceLength/length(noise));
%             noiseExtended = [];
%             originalNoise = noise;
%             for j = 1:numRepeats
%                 if mod(j, 2) == 1
%                     % Odd repetitions: keep original direction
%                     noiseExtended = [noiseExtended; originalNoise];
%                 else
%                     % Even repetitions: reverse direction
%                     noiseExtended = [noiseExtended; flipud(originalNoise)];
%                 end
%             end
%             noise = noiseExtended;
%             break
%         end
%     end
%
%     % Handle excess noise length
%     if length(noise) >= sequenceLength
%         trimmedNoise = noise(1:sequenceLength);
%     else
%         % In case we still don't have enough (shouldn't happen with the recycling)
%         trimmedNoise = [noise; zeros(sequenceLength-length(noise), 1)];
%     end
%
%     % Generate random SNR
%     SNRs(i) = snrRange(1) + (snrRange(2) - snrRange(1)) * rand();
%
%     % Mix signal and noise
%     audioNoisy{i} = mixSNR(audioIn{i}, trimmedNoise, SNRs(i));
% end
% end