function [probabilities, features, execTime, silenceMask, numAudioSegments] = gavdNetInference(audio, fs, ...
    model, memoryAvailable, featureFraming, frameStandardization, minSilenceDuration, plotting, audioNormalize)
% This function extracts features (i.e, spectrograms) from audio files, then 
% runs them through a trained animal call detection model. The output is 
% a vector of probability scores, one for every spectrogram time bin.
%
% Inputs:
%   audio           - The audio file to run thorugh the detector
%   fs              - The sample rate of the audio file (Hz)
%   memoryAvailable - Memory available on the operating environment (bytes)
%   featureFraming  - different modes for splitting long inputs.
%           Options: 
%               'none'          - Computes the spectrogram for the whole 
%                                 audio file, and runs this through the 
%                                 network in one pass.
%               'simple'        - Computes the spectrogram for the whole 
%                                 audio file, and breaks it into frames of 
%                                 same size and overlap as the training 
%                                 data frames.
%               'event-splt'    - Uses signal statistics to find regions of
%                                 the audio file that have very high energy,
%                                 and splits the audio file based on changes 
%                                 in the mean of the signal envelope. If
%                                 the spectrogram of any segment is longer 
%                                 than the frameLength used for feature 
%                                 framing at training time, then it is
%                                 broken into frames of that size, with the
%                                 same overlap used at training.
%   audioNormalize  - Optional logical flag. If true (default when omitted),
%                     subtracts the DC component and divides by peak
%                     amplitude before inference. Required for calibrated
%                     archives whose absolute amplitude is far below the
%                     training-data range (e.g. files calibrated to
%                     physical pressure units), where the mel-spectrogram
%                     + saturation pipeline would otherwise collapse to a
%                     near-constant input and the network would return its
%                     no-call rest state. Pass false to disable.
%
% NOTE: For long audio files with large variance in amplitude, the choice of
% feature framing mode can have a dramatic effect on performance. For audio 
% with relatively small dynamic range, and no high amplitude events that
% could be considered outliers in terms of signal statistics, 'none' is
% probably best. If the audio has large amplitude variance, i.e, discrete, 
% high-amplitude events that are dramatically louder than the background 
% level, then 'simple' or 'event-split' are more appropriate.
%
% Outputs:
%   probabilities - The vector of probability scores in the range [0, 1],
%                   indicating the probability of target signal presence 
%                   for every spectrogram tim bin.
%   features      - The spectrogram of the entire signal. For event-split
%                   mode, this is stitched together from the spectrograms 
%                   of each audio segment.
%   execTime      - Execution time in seconds (excluding preprocessing)
%   numAudioSegments - The number of audio segments that the audio file was 
%                       divided into by the event splitter. For
%                       featureFraming = 'none' or 'simple', this is always 
%                       NaN.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%% Begin:

% Optional audio-amplitude normalisation. Default ON (opt-out by passing
% audioNormalize=false). Required for calibrated archives whose absolute
% amplitude is far below the training-data range, where the mel-spectrogram
% + saturation pipeline would otherwise collapse to a near-constant input
% and the network would return its no-call rest state (~0.13).
if nargin < 9 || isempty(audioNormalize)
    audioNormalize = true;
end
if audioNormalize
    audio = audio - mean(audio, 'omitnan');
    peakVal = max(abs(audio));
    if peakVal > 0
        audio = audio / peakVal;
    end
end

switch featureFraming
    case 'none'
        % Search for silence in the signal.
        silenceMask = detectSilentRegions(audio, fs, minSilenceDuration);
        if any(silenceMask)
            fprintf('Silent regions found in audio file. Probabilities will be set to zero for these regions.\n')
            % Transform silence mask from audio sample domain to the spectrogram time bin domain
            silenceMaskFeaturesDomain = maskToFeaturesDomain(silenceMask, ...
                fs, ...
                model.preprocParams.fsTarget, ...
                model.preprocParams.windowLen, ...
                model.preprocParams.hopLen);
        end

        % Run Preprocessing & Feature Extraction on audio
        fprintf('\tPreprocesing audio & extracting features...\n')
        features = gavdNetPreprocess(...
            audio, ...
            fs, ...
            model.preprocParams.fsTarget, ...
            model.preprocParams.bandwidth, ...
            model.preprocParams.windowLen,...
            model.preprocParams.hopLen,...
            model.preprocParams.saturationRange);

        % Estimate max minibatch size
        optimalMinibatchSize = estimateInferenceMinibatchSize(memoryAvailable, size(features, 2));

        % Run Model in minibatch mode
        fprintf('\tRunning features through model...\n')
        tic
        probabilities = minibatchpredictAdaptive(model.net, features, optimalMinibatchSize);
        execTime = toc;

        % Move reults back to the CPU if on GPU
        if isgpuarray(probabilities)
            probabilities = gather(probabilities);
        end

        % If there were silent parts of the audio signal, we should not
        % trust the probabilties for those samples.
        if any(silenceMask)
            probabilities(silenceMaskFeaturesDomain) = 0;
        end

        % Write number of audio segments to output
        numAudioSegments = NaN;

    case 'simple'
        % Search for silence in the signal.
        silenceMask = detectSilentRegions(audio, fs, minSilenceDuration);
        if any(silenceMask)
            fprintf('\tSilent regions found in audio file.\n')
            fprintf('\tProbabilities will be set to zero for these regions.\n')
            % Transform silence mask from audio sample domain to the spectrogram time bin domain
            silenceMaskFeaturesDomain = maskToFeaturesDomain(silenceMask, ...
                fs, ...
                model.preprocParams.fsTarget, ...
                model.preprocParams.windowLen, ...
                model.preprocParams.hopLen);
        end

        % Run Preprocessing & Feature Extraction on audio
        fprintf('\tPreprocesing audio & extracting features...\n')
        features = gavdNetPreprocess(...
            audio, ...
            fs, ...
            model.preprocParams.fsTarget, ...
            model.preprocParams.bandwidth, ...
            model.preprocParams.windowLen,...
            model.preprocParams.hopLen,...
            model.preprocParams.saturationRange);
       
        % Break features into frames
        featureFrames = featureBuffer(features, ...
            model.featureFraming.frameLength, ...
            model.featureFraming.frameOverlapPercent, ...
                    leftoverTimeBins='keep', ...
                    standardizeFrames=frameStandardization);
       
        % Estimate max minibatch size
        optimalMinibatchSize = estimateInferenceMinibatchSize(memoryAvailable, size(featureFrames{1}, 2));
    
        % Run Model in minibatch mode to save memory
        fprintf('\tRunning frames through model.\n')
        tic
        probabilitiesFrames = minibatchpredictAdaptive(model.net, featureFrames, optimalMinibatchSize);
                   
        % Move back to the CPU if on GPU
        if isgpuarray(probabilitiesFrames)
            probabilitiesFrames = gather(probabilitiesFrames);
        end
    
        % Convert 3D matrix to nested cell array to pack correctly for
        % concatenate function
        probabilitiesFrames = (squeeze(num2cell(probabilitiesFrames, [1 2])))';
        execTime = toc;
        
        % Stitch together probability vectors for each frame and take the 
        % average of overlapping elements
        numSpectrogramTimeBins = size(features, 2);
        probabilities = concatenateOverlappingProbs(probabilitiesFrames, ...
            numSpectrogramTimeBins, model.featureFraming.frameHopLength);

        % If there were silent parts of the audio signal, we should not
        % trust the probabilties for those samples.
        if any(silenceMask)
            probabilities(silenceMaskFeaturesDomain) = 0;
        end

        % Write number of audio segments to output
        numAudioSegments = NaN;

    case 'event-split'
        % Search for silence in the signal.
        silenceMask = detectSilentRegions(audio, fs, minSilenceDuration);
        if any(silenceMask)
            fprintf('\tSilent regions found in audio file.\n')
            fprintf('\tProbabilities will be set to zero for these regions.\n')
            % Transform silence mask from audio sample domain to the spectrogram time bin domain
            silenceMaskFeaturesDomain = maskToFeaturesDomain(silenceMask, ...
                fs, ...
                model.preprocParams.fsTarget, ...
                model.preprocParams.windowLen, ...
                model.preprocParams.hopLen);
        end

        % Split audio into segments based on signal statistics
        smoothingWindowDuration = model.dataSynthesisParams.maxTargetCallDuration * 4;
        eventOverlapDuration = model.dataSynthesisParams.maxTargetCallDuration * 2;
        [audioSegments, splitIndices, ~] = eventSplitter(audio, fs, ...
            smoothingWindowDuration, eventOverlapDuration);

        featuresSegments = cell(size(audioSegments));
        for i = 1:length(audioSegments)
            fprintf('\tPreprocesing audio & extracting features for segment %d of %d...\n',...
                i, length(audioSegments))
            
            % Run Preprocessing & Feature Extraction on audio
            featuresSegments{i, 1} = gavdNetPreprocess(...
                audioSegments{i, 1}, ...
                fs, ...
                model.preprocParams.fsTarget, ...
                model.preprocParams.bandwidth, ... 
                model.preprocParams.windowLen,...
                model.preprocParams.hopLen, ...
                model.preprocParams.saturationRange);
        end

        % If any of the featuresSegments are longer than the feature 
        % frameLength used at training time, we need to further break these 
        % into frames to make sure the GRU layers don't have to deal with
        % context longer than they are trained for:
        featureSegmentsFrames = cell(size(featuresSegments));
        for i = 1:length(featuresSegments)
            if size(featuresSegments{i, 1}, 2) > model.featureFraming.frameLength
                % Break features into frames
                featureSegmentsFrames{i, 1} = featureBuffer(featuresSegments{i, 1}, ...
                    model.featureFraming.frameLength, ...
                    model.featureFraming.frameOverlapPercent, ...
                    leftoverTimeBins='keep', ...
                    standardizeFrames=frameStandardization);
            else
                % Treat the whole segment as a single frame
                featureSegmentsFrames{i, 1} = {featuresSegments{i, 1}};
            end
        end
       
        % Run each frame through the model
        probabilitiesSegmentsFrames = cell(size(featureSegmentsFrames));
        probabilitiesSegments = cell(size(audioSegments));
        fprintf('\tRunning segments through model.')
        tic
        for i = 1:length(featureSegmentsFrames)
            % Estimate max minibatch size
            optimalMinibatchSize = estimateInferenceMinibatchSize(memoryAvailable, size(featureSegmentsFrames{i, 1}{1, 1}, 2));
        
            % Run Model in minibatch mode to save memory. OOM-safe adaptive
            % batching lets a small GPU (e.g. the 4 GB T550) process a long
            % segment that would not fit at the estimated minibatch size.
            probabilitiesSegmentsFrames{i, 1} = minibatchpredictAdaptive(model.net, ...
                featureSegmentsFrames{i, 1}, optimalMinibatchSize);
        
            % Move back to the CPU if on GPU
            if isgpuarray(probabilitiesSegmentsFrames{i, 1})
                probabilitiesSegmentsFrames{i, 1} = gather(probabilitiesSegmentsFrames{i, 1});
            end
        
            % Convert 3D matrix to nested cell array to pack correctly for
            % concatoate function
            probabilitiesSegmentsFrames{i, 1} = (squeeze(num2cell(probabilitiesSegmentsFrames{i, 1}, [1 2])))';

            % Stitch together probability vectors for each frame and 
            % take the average of overlapping elements
            numSpectrogramTimeBins = size(featuresSegments{i, 1}, 2);
            probabilitiesSegments{i, 1} = concatenateOverlappingProbs(probabilitiesSegmentsFrames{i, 1}, ...
                numSpectrogramTimeBins, model.featureFraming.frameHopLength);

            % Do some dots so the user knows we haven't hung 
            fprintf('.')
        end
        fprintf('\n')
        execTime = toc;

        % Transpose 
        probabilitiesSegments = cellfun(@transpose, probabilitiesSegments, UniformOutput=false);

         % Stitch together probability vectors for each segment
        if ~isscalar(probabilitiesSegments)
            [probabilities, features] = segmentStitcher(probabilitiesSegments, ...
                splitIndices, model.preprocParams, fs, featuresSegments);
        else
            probabilities = probabilitiesSegments{1};
            features = featuresSegments{1};
        end
        
        % If there were silent parts of the audio signal, we should not
        % trust the probabilties for those samples.
        if any(silenceMask)
            probabilities(silenceMaskFeaturesDomain) = 0;
        end
        
        % Write number of audio segments to output
        numAudioSegments = length(audioSegments);
                
    otherwise
        error('Invalid entry for argument "featureFraming".')
end

duration = length(audio)/fs;
dt = 1/fs;
tSamples = 0:dt:duration-dt;
if plotting == true
    figure(1)
    tiledlayout(2,1)
    nexttile
    plot(tSamples, audio)
    xlabel('Time (s)')
    ylabel('Amplitude')
    xlim([tSamples(1), tSamples(end)])
    title('Audio Signal')
    
    nexttile
    plot(probabilities)
    xlabel('Time (s)')
    ylabel('Probability')
    ylim([0, 1.1])
    xlim([1, length(probabilities)])
    title('Probability of Positive Detection')

end
end

%% Helper Funcitons

function probs = minibatchpredictAdaptive(net, data, minibatchSize)
% minibatchpredict with OOM-safe adaptive batching. The estimated minibatch
% size is tried first (so the normal, non-failing path is unchanged); on ANY
% failure - typically a GPU out-of-memory on a small device (e.g. the 4 GB
% T550) processing a very long segment's many frames - the MiniBatchSize is
% halved and the call retried, down to 1. A single-sequence input rethrows
% immediately, since reducing the batch cannot help it.
%
% Every frame is first right-zero-padded to the GLOBAL longest length.
% minibatchpredict's SequenceLength only supports "longest"/"shortest" (not a
% fixed number), and "longest" pads each MINIBATCH to its own longest
% sequence - so once MiniBatchSize < number-of-frames the sub-batches pad to
% different lengths and their numeric outputs cannot be assembled (at
% MiniBatchSize = 1 minibatchpredict errors outright). With all frames already
% the same length, "longest" adds no further padding and every minibatch is
% uniform. The manual padding is right-zero, matching minibatchpredict's own
% default sequence padding, so at the FULL batch the result is bit-identical
% to the pre-existing behaviour (verified empirically) - the normal path is
% unchanged. A retry (smaller batch) only happens when the full batch would
% otherwise FAIL; cuDNN's batch-size-dependent arithmetic then perturbs the
% result by a small amount (observed <= ~1e-3, below the postprocessing
% hysteresis band, no effect on detections) - the alternative for that file
% being no result at all.
    if iscell(data)
        nObs = numel(data);
        L = max(cellfun(@(f) size(f, 2), data));
        if any(cellfun(@(f) size(f, 2) < L, data))
            data = cellfun(@(f) padFrameRight(f, L), data, 'UniformOutput', false);
        end
    else
        nObs = 1;                    % a single [C x T] sequence
    end
    mbs = min(max(round(minibatchSize), 1), max(nObs, 1));
    while true
        try
            probs = minibatchpredict(net, data, ...
                SequenceLength="longest", MiniBatchSize=mbs);
            return
        catch ME
            if mbs <= 1 || nObs <= 1
                rethrow(ME)
            end
            newMbs = max(1, floor(mbs / 2));
            warning('gavdNetInference:minibatchRetry', ...
                'minibatchpredict failed at MiniBatchSize=%d: %s. Retrying at %d.', ...
                mbs, ME.message, newMbs);
            mbs = newMbs;
        end
    end
end

function f2 = padFrameRight(f, L)
% Right-pad a [C x l] feature frame with zeros to [C x L] (l <= L),
% preserving class / gpuArray-ness. Matches minibatchpredict's default
% sequence padding, so padded frames give the same predictions.
    l = size(f, 2);
    if l >= L
        f2 = f;
    else
        f2 = [f, zeros(size(f, 1), L - l, 'like', f)];
    end
end

function transformedMask = maskToFeaturesDomain(mask, fsIn, fsTarget, windowLen, hopLen)
%   This function transforms a binary mask from the audio sample domain to
%   the spectrogram time-bin domain, matching the processing steps used in
%   STFT computation.

% Resample mask if sample rate changes
if fsIn ~= fsTarget
    [p, q] = rat(fsTarget/fsIn, 1e-9);
    maskResamp = cast(resample(double(mask(:)), p, q), like=mask);
else
    maskResamp = mask(:);
end

% Zero pad to match the features (same padding as in STFT)
padLen = ceil(windowLen/2);
maskPadded = [zeros(padLen, 1, like=maskResamp); maskResamp; zeros(padLen, 1, like=maskResamp)];

% Buffer the mask to match the spectrogram time bins
overlapLen = windowLen - hopLen;
maskBuffered = buffer(maskPadded, windowLen, overlapLen, "nodelay");

% Take the mode of each frame to get the dominant mask value per time bin
transformedMask = mode(maskBuffered, 1);
end