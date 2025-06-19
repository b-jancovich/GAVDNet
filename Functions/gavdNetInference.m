function [probabilities, features, execTime, silenceMask, numAudioSegments] = gavdNetInference(audio, fs, ...
    model, memoryAvailable, featureFraming, minSilenceDuration)
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
%                                 in the mean of the signal envelope.
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
        probabilities = minibatchpredict(model.net, features, ...
            SequenceLength="longest", MiniBatchSize=optimalMinibatchSize);
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
            fprintf('Silent regions found in audio file. Probabilities will be set to zero for these regions.')
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
            model.featureFraming.frameOverlapPercent);
       
        % Run each frame through the model
        probabilitiesFrames = cell(size(featureFrames));
        fprintf('\tRunning frames through model.')
        tic
        for i = 1:length(featureFrames)
            % Estimate max minibatch size
            optimalMinibatchSize = estimateInferenceMinibatchSize(memoryAvailable, size(featureFrames{i}, 2));
    
            % Run Model in minibatch mode to save memory
            probabilitiesFrames{i} = minibatchpredict(model.net, featureFrames{i}, ...
                SequenceLength="longest", MiniBatchSize=optimalMinibatchSize);
                   
            % Move back to the CPU if on GPU
            if isgpuarray(probabilitiesFrames{i})
                probabilitiesFrames{i} = gather(probabilitiesFrames{i});
            end
            % Do some dots so the user knows we haven't hung 
            if mod(i, 100) == 0
                fprintf('.')
            end
        end
        execTime = toc;
        fprintf('\n')
        
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
            featuresSegments{i} = gavdNetPreprocess(...
                audioSegments{i}, ...
                fs, ...
                model.preprocParams.fsTarget, ...
                model.preprocParams.bandwidth, ... 
                model.preprocParams.windowLen,...
                model.preprocParams.hopLen, ...
                model.preprocParams.saturationRange);
        end

        % Run each frame through the model
        probabilitiesSegments = cell(size(featuresSegments));
        fprintf('\tRunning segments through model.')
        tic
        for i = 1:length(featuresSegments)
            % Estimate max minibatch size
            optimalMinibatchSize = estimateInferenceMinibatchSize(memoryAvailable, size(featuresSegments{i}, 2));
    
            % Run Model in minibatch mode to save memory
            probabilitiesSegments{i} = minibatchpredict(model.net, featuresSegments{i}, ...
                SequenceLength="longest", MiniBatchSize=optimalMinibatchSize);
            % Do some dots so the user knows we haven't hung 
            if mod(i, 100) == 0
                fprintf('.')
            end
        end
        fprintf('\n')
        execTime = toc;
        
        % Stitch together probability vectors for each segment
        if ~isscalar(probabilitiesSegments)
            [probabilities, features] = segmentStitcher(probabilitiesSegments, ...
            splitIndices, model.preprocParams.windowLen, ...
            model.preprocParams.hopLen, featuresSegments); 
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
end

%% Helper Funcitons

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