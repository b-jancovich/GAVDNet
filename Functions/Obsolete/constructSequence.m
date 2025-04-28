function [audio, mask] = constructSequence(ads, fs, duration, numSequences, minSilenceDuration)
% constructSequence constructs sequences of audio signals by concatenating samples
% from an audioDatastore with random silence durations between.
%
% [audio, mask] = constructSignal(ads, fs, duration, numSequences, minSilenceDuration)
%
% Inputs:
% ads - audioDatastore containing single word samples
% fs - sample rate of data in datastore (Hz)
% duration - duration of each output audio sequence (s)
% numSequences - number of sequences to generate
% minSilenceDuration - minimum duration of silence between samples (s)
%
% Outputs:
% audio - cell array of audio sequences (preserves GPU arrays if input is on GPU)
% mask - cell array of binary masks indicating speech regions
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
% Calculate samples per sequence
N = duration * fs;

% Determine samples per sequence
totalFiles = numel(ads.Files);
samplesPerSequence = floor(totalFiles / numSequences);

% Initialize cell arrays for output
audio = cell(numSequences, 1);
mask = cell(numSequences, 1);

% Determine if we're working with GPU arrays
if strcmp(ads.OutputEnvironment, 'gpu')
    useGPU = true;
else
    useGPU = false;
end

% Determine data is single or double precision
dataType = ads.OutputDataType;

% Sequence index counter
seqIdx = 1;

% Continue until we have numSequences valid sequences
while seqIdx <= numSequences 
   
    % Initialize this sequence
    if useGPU
        seqAudio = gpuArray(zeros(N, 1, dataType));
        seqMask = gpuArray(zeros(N, 1, dataType));
    else
        seqAudio = zeros(N, 1, dataType);
        seqMask = zeros(N, 1, dataType);
    end

    % Estimate total amount of audio samples
    sampleBatch = min(samplesPerSequence, totalFiles - (seqIdx-1)*samplesPerSequence);

    % If we have no samples to process
    if sampleBatch <= 0
        warning('Insufficient call samples.')
        break
    end

    % Collect sample lengths for planning
    sampleLengths = zeros(sampleBatch, 1);
    sampleData = cell(sampleBatch, 1);
    for i = 1:sampleBatch
        data = read(ads);
        data = data ./ max(abs(data)); % Scale amplitude
        sampleData{i} = data;
        sampleLengths(i) = length(data);
    end

    % Calculate total audio content length
    totalAudioLength = sum(sampleLengths(1:sampleBatch));

    % Check if we can fit all samples with minimum silence
    minSilenceSamples = minSilenceDuration * fs;
    minTotalSilence = (sampleBatch - 1) * minSilenceSamples;

    % Check if we can fit all samples
    if totalAudioLength + minTotalSilence > N
        % Not enough space, reduce samples until we can fit
        while totalAudioLength + minTotalSilence > N && sampleBatch > 1
            sampleBatch = sampleBatch - 1;
            totalAudioLength = sum(sampleLengths(1:sampleBatch));
            minTotalSilence = (sampleBatch - 1) * minSilenceSamples;
        end
    end

    if sampleBatch == 0
        continue;  % Skip this sequence if we can't fit any samples
    end

    % Approach: Divide the sequence into equal segments and place one sample in each segment
    % with random positioning within the segment

    % Calculate segment size
    segmentSize = floor(N / sampleBatch);

    % Initialize array for actual positions
    positions = zeros(sampleBatch, 1);

    % For each sample, place it randomly within its segment
    for i = 1:sampleBatch
        % Define segment bounds
        segmentStart = (i-1) * segmentSize + 1;
        segmentEnd = min(i * segmentSize, N - sampleLengths(i) + 1);

        % Ensure we have enough space in this segment
        if segmentEnd <= segmentStart
            segmentEnd = segmentStart;
        end

        % Place the sample randomly within this segment
        positions(i) = randi([segmentStart, segmentEnd], 1, 1);
    end

    % Sort positions to ensure proper ordering
    [positions, order] = sort(positions);
    sampleLengths = sampleLengths(order);
    sampleData = sampleData(order);

    % Verify all samples have at least minimum silence between them
    for i = 2:sampleBatch
        % Check if we need to adjust position to maintain minimum silence
        minPos = positions(i-1) + sampleLengths(i-1) + minSilenceSamples;
        if positions(i) < minPos
            % Need to adjust position
            positions(i) = minPos;
        end
    end

    % Check if the last sample exceeds sequence length, adjust if needed
    if positions(end) + sampleLengths(end) - 1 > N
        % Apply sequential backward adjustment
        for i = sampleBatch:-1:2
            adjustment = (positions(i) + sampleLengths(i) - 1) - N;
            if adjustment > 0
                positions(i) = positions(i) - adjustment;

                % Ensure minimum silence is maintained
                if positions(i) < positions(i-1) + sampleLengths(i-1) + minSilenceSamples
                    positions(i) = positions(i-1) + sampleLengths(i-1) + minSilenceSamples;
                end
            else
                break; % No more adjustment needed
            end
        end
    end

    % Place the samples at the calculated positions
    for i = 1:sampleBatch
        data = sampleData{i};
        pos = positions(i);

        % Make sure we don't exceed array bounds
        endPos = min(N, pos + length(data) - 1);
        validLength = endPos - pos + 1;

        if validLength < length(data)
            data = data(1:validLength);
        end

        seqAudio(pos:endPos) = data;
        seqMask(pos:endPos) = 1;  % Changed from true to 1 for consistency
    end
    
    % If we got valid audio...
    if isValidAudio(seqAudio) == true
        % Store this sequence
        audio{seqIdx} = seqAudio;
        mask{seqIdx} = seqMask;
        seqIdx = seqIdx + 1;
    end
    % If not valid, we'll continue the while loop without incrementing seqIdx
end

% If we couldn't generate all sequences, trim the output arrays
if seqIdx <= numSequences
    audio = audio(1:seqIdx-1);
    mask = mask(1:seqIdx-1);
end
end

% Sometimes returns empty sequences:
% function [audio, mask] = constructSequence(ads, fs, duration, numSequences, minSilenceDuration)
% % constructSequence constructs sequences of audio signals by concatenating samples
% % from an audioDatastore with random silence durations between.
% %
% % [audio, mask] = constructSignal(ads, fs, duration, numSequences, minSilenceDuration)
% %
% % Inputs:
% % ads - audioDatastore containing single word samples
% % fs - sample rate of data in datastore (Hz)
% % duration - duration of each output audio sequence (s)
% % numSequences - number of sequences to generate
% % minSilenceDuration - minimum duration of silence between samples (s)
% %
% % Outputs:
% % audio - cell array of audio sequences (preserves GPU arrays if input is on GPU)
% % mask - cell array of binary masks indicating speech regions
% %
% % Ben Jancovich, 2024
% % Centre for Marine Science and Innovation
% % School of Biological, Earth and Environmental Sciences
% % University of New South Wales, Sydney, Australia
% %
% % Calculate samples per sequence
% N = duration * fs;
% 
% % Determine samples per sequence
% totalFiles = numel(ads.Files);
% samplesPerSequence = ceil(totalFiles / numSequences);
% 
% % Initialize cell arrays for output
% audio = cell(numSequences, 1);
% mask = cell(numSequences, 1);
% 
% % Reset datastore
% reset(ads);
% 
% % Determine if we're working with GPU arrays
% if strcmp(ads.OutputEnvironment, 'gpu')
%     useGPU = true;
% else
%     useGPU = false;
% end
% 
% % Determine data is single or double precision
% dataType = ads.OutputDataType;
% 
% % For each sequence
% for seqIdx = 1:numSequences
% 
%     % Initialize this sequence
%     if useGPU
%         seqAudio = gpuArray(zeros(N, 1, dataType));
%         seqMask = gpuArray(zeros(N, 1, dataType));
%     else
%         seqAudio = zeros(N, 1, dataType);
%         seqMask = zeros(N, 1, dataType);
%     end
% 
%     % Estimate total amount of audio samples
%     sampleBatch = min(samplesPerSequence, totalFiles - (seqIdx-1)*samplesPerSequence);
% 
%     % If we have no samples to process, break
%     if sampleBatch <= 0
%         break;
%     end
% 
%     % Collect sample lengths for planning
%     sampleLengths = zeros(sampleBatch, 1);
%     sampleData = cell(sampleBatch, 1);
%     for i = 1:sampleBatch
%         if ~hasdata(ads)
%             sampleBatch = i - 1;
%             break;
%         end
%         data = read(ads);
%         data = data ./ max(abs(data)); % Scale amplitude
%         sampleData{i} = data;
%         sampleLengths(i) = length(data);
%     end
% 
%     % Calculate total audio content length
%     totalAudioLength = sum(sampleLengths(1:sampleBatch));
% 
%     % Check if we can fit all samples with minimum silence
%     minSilenceSamples = minSilenceDuration * fs;
%     minTotalSilence = (sampleBatch - 1) * minSilenceSamples;
% 
%     % Check if we can fit all samples
%     if totalAudioLength + minTotalSilence > N
%         % Not enough space, reduce samples until we can fit
%         while totalAudioLength + minTotalSilence > N && sampleBatch > 1
%             sampleBatch = sampleBatch - 1;
%             totalAudioLength = sum(sampleLengths(1:sampleBatch));
%             minTotalSilence = (sampleBatch - 1) * minSilenceSamples;
%         end
%     end
% 
%     if sampleBatch == 0
%         continue;  % Skip this sequence if we can't fit any samples
%     end
% 
%     % Approach: Divide the sequence into equal segments and place one sample in each segment
%     % with random positioning within the segment
% 
%     % Calculate segment size
%     segmentSize = floor(N / sampleBatch);
% 
%     % Initialize array for actual positions
%     positions = zeros(sampleBatch, 1);
% 
%     % For each sample, place it randomly within its segment
%     for i = 1:sampleBatch
%         % Define segment bounds
%         segmentStart = (i-1) * segmentSize + 1;
%         segmentEnd = min(i * segmentSize, N - sampleLengths(i) + 1);
% 
%         % Ensure we have enough space in this segment
%         if segmentEnd <= segmentStart
%             segmentEnd = segmentStart;
%         end
% 
%         % Place the sample randomly within this segment
%         positions(i) = randi([segmentStart, segmentEnd], 1, 1);
%     end
% 
%     % Sort positions to ensure proper ordering
%     [positions, order] = sort(positions);
%     sampleLengths = sampleLengths(order);
%     sampleData = sampleData(order);
% 
%     % Verify all samples have at least minimum silence between them
%     for i = 2:sampleBatch
%         % Check if we need to adjust position to maintain minimum silence
%         minPos = positions(i-1) + sampleLengths(i-1) + minSilenceSamples;
%         if positions(i) < minPos
%             % Need to adjust position
%             positions(i) = minPos;
%         end
%     end
% 
%     % Check if the last sample exceeds sequence length, adjust if needed
%     if positions(end) + sampleLengths(end) - 1 > N
%         % Apply sequential backward adjustment
%         for i = sampleBatch:-1:2
%             adjustment = (positions(i) + sampleLengths(i) - 1) - N;
%             if adjustment > 0
%                 positions(i) = positions(i) - adjustment;
% 
%                 % Ensure minimum silence is maintained
%                 if positions(i) < positions(i-1) + sampleLengths(i-1) + minSilenceSamples
%                     positions(i) = positions(i-1) + sampleLengths(i-1) + minSilenceSamples;
%                 end
%             else
%                 break; % No more adjustment needed
%             end
%         end
%     end
% 
%     % Place the samples at the calculated positions
%     for i = 1:sampleBatch
%         data = sampleData{i};
%         pos = positions(i);
% 
%         % Make sure we don't exceed array bounds
%         endPos = min(N, pos + length(data) - 1);
%         validLength = endPos - pos + 1;
% 
%         if validLength < length(data)
%             data = data(1:validLength);
%         end
% 
%         seqAudio(pos:endPos) = data;
%         seqMask(pos:endPos) = true;
%     end
% 
%     % Store this sequence
%     audio{seqIdx} = seqAudio;
%     mask{seqIdx} = seqMask;
% end
% end