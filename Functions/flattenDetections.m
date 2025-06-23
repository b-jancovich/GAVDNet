function flatDetections = flattenDetections(detections, preprocParams)
% FLATTENDETECTIONS Converts hierarchical detections struct to flat format
%   Takes a detections struct with one row per audio file (containing multiple
%   detections) and returns a flattened struct with one row per detection.
%   Now includes probability subsequences for each detection.
%
% Inputs:
%   detections - Original struct with one entry per file
%   preprocParams - Preprocessing parameters struct containing fsTarget, 
%                   hopLen, windowLen for probability extraction
%
% Outputs:
%   flatDetections - Flattened struct with one entry per detection,
%                    including probability subsequences
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Handle case where preprocParams is not provided (backward compatibility)
if nargin < 2
    preprocParams = [];
end

% Unpack preprocessing parameters if available
if ~isempty(preprocParams)
    targetFs = preprocParams.fsTarget;
    hopLen = preprocParams.hopLen;
    windowLen = preprocParams.windowLen;
    padLen = ceil(windowLen/2);
    extractProbs = true;
else
    extractProbs = false;
end

% Initialize counter for flat detections
flatIdx = 1;

% Initialize the flat detections struct
flatDetections = struct();

% Loop through each file in the original detections struct
for fileIdx = 1:length(detections)
    % Get the current file entry
    currFile = detections(fileIdx);
    
    % Skip if no detections in this file
    if currFile.nDetections == 0 || ~isfield(currFile, 'eventSampleBoundaries') || isempty(currFile.eventSampleBoundaries)
        continue;
    end
    
    % Loop through each detection in the current file
    for detIdx = 1:currFile.nDetections
        % Copy file-level information
        flatDetections(flatIdx).fileName = currFile.fileName;
        flatDetections(flatIdx).fileFs = currFile.fileFs;
        flatDetections(flatIdx).fileSamps = currFile.fileSamps;
        flatDetections(flatIdx).fileDuration = currFile.fileDuration;
        flatDetections(flatIdx).fileStartDateTime = currFile.fileStartDateTime;
        
        % Copy detection-specific information
        flatDetections(flatIdx).eventSampleStart = currFile.eventSampleBoundaries(detIdx, 1);
        flatDetections(flatIdx).eventSampleEnd = currFile.eventSampleBoundaries(detIdx, 2);
        
        % Calculate duration in seconds
        flatDetections(flatIdx).eventDuration = (flatDetections(flatIdx).eventSampleEnd - ...
            flatDetections(flatIdx).eventSampleStart) / flatDetections(flatIdx).fileFs;
        
        % Copy confidence if available
        if isfield(currFile, 'confidence') && ~isempty(currFile.confidence)
            flatDetections(flatIdx).confidence = currFile.confidence(detIdx);
        end
        
        % Copy datetime information if available
        if isfield(currFile, 'eventTimesDT') && ~isempty(currFile.eventTimesDT)
            flatDetections(flatIdx).eventStartTime = currFile.eventTimesDT(detIdx, 1);
            flatDetections(flatIdx).eventEndTime = currFile.eventTimesDT(detIdx, 2);
        end
        
        % Extract probability subsequence for this detection
        if extractProbs && isfield(currFile, 'probabilities') && ~isempty(currFile.probabilities)
            % Convert sample boundaries to frame boundaries
            startFrame = sample2frame(flatDetections(flatIdx).eventSampleStart, ...
                currFile.fileFs, targetFs, hopLen, padLen);
            endFrame = sample2frame(flatDetections(flatIdx).eventSampleEnd, ...
                currFile.fileFs, targetFs, hopLen, padLen);
            
            % Ensure frame indices are within bounds
            startFrame = max(1, min(startFrame, length(currFile.probabilities)));
            endFrame = max(1, min(endFrame, length(currFile.probabilities)));
            
            % Extract probability subsequence
            if startFrame <= endFrame
                flatDetections(flatIdx).probabilities = currFile.probabilities(startFrame:endFrame);
            else
                % Handle edge case where start > end (very short detection)
                flatDetections(flatIdx).probabilities = currFile.probabilities(startFrame);
            end
            
            % Store frame boundaries for reference
            flatDetections(flatIdx).eventFrameStart = startFrame;
            flatDetections(flatIdx).eventFrameEnd = endFrame;
        end
        
        % Increment the flat index counter
        flatIdx = flatIdx + 1;
    end
end

end

function frameIdx = sample2frame(sampleIdx, fileFs, targetFs, hopLen, padLen)
% sample2frame Convert sample index to frame index
%
% This function converts from the audio sample domain to the spectrogram
% frame domain, accounting for resampling and padding used in preprocessing

% Convert sample index to time in original signal
timeInOriginal = (sampleIdx - 1) / fileFs;

% Convert to time in resampled signal (time is preserved)
timeInResampled = timeInOriginal;

% Account for padding offset added during preprocessing
timeInPaddedSignal = timeInResampled + padLen / targetFs;

% Convert to frame index (frames are numbered starting from 1)
frameIdx = round(timeInPaddedSignal * targetFs / hopLen) + 1;

end

% function flatDetections = flattenDetections(detections)
% % FLATTENDETECTIONS Converts hierarchical detections struct to flat format
% %   Takes a detections struct with one row per audio file (containing multiple
% %   detections) and returns a flattened struct with one row per detection.
% %
% % Inputs:
% %   detections - Original struct with one entry per file
% %
% % Outputs:
% %   flatDetections - Flattened struct with one entry per detection
% %
% % Ben Jancovich, 2024
% % Centre for Marine Science and Innovation
% % School of Biological, Earth and Environmental Sciences
% % University of New South Wales, Sydney, Australia
% %
% 
% % Initialize counter for flat detections
% flatIdx = 1;
% 
% % Initialize the flat detections struct
% flatDetections = struct();
% 
% % Loop through each file in the original detections struct
% for fileIdx = 1:length(detections)
%     % Get the current file entry
%     currFile = detections(fileIdx);
% 
%     % Skip if no detections in this file
%     if currFile.nDetections == 0 || ~isfield(currFile, 'eventSampleBoundaries') || isempty(currFile.eventSampleBoundaries)
%         continue;
%     end
% 
%     % Loop through each detection in the current file
%     for detIdx = 1:currFile.nDetections
%         % Copy file-level information
%         flatDetections(flatIdx).fileName = currFile.fileName;
%         flatDetections(flatIdx).fileFs = currFile.fileFs;
%         flatDetections(flatIdx).fileSamps = currFile.fileSamps;
%         flatDetections(flatIdx).fileDuration = currFile.fileDuration;
%         flatDetections(flatIdx).fileStartDateTime = currFile.fileStartDateTime;
% 
%         % Copy detection-specific information
%         flatDetections(flatIdx).eventSampleStart = currFile.eventSampleBoundaries(detIdx, 1);
%         flatDetections(flatIdx).eventSampleEnd = currFile.eventSampleBoundaries(detIdx, 2);
% 
%         % Calculate duration in seconds
%         flatDetections(flatIdx).eventDuration = (flatDetections(flatIdx).eventSampleEnd - ...
%             flatDetections(flatIdx).eventSampleStart) / flatDetections(flatIdx).fileFs;
% 
%         % Copy confidence if available
%         if isfield(currFile, 'confidence') && ~isempty(currFile.confidence)
%             flatDetections(flatIdx).confidence = currFile.confidence(detIdx);
%         end
% 
%         % Copy datetime information if available
%         if isfield(currFile, 'eventTimesDT') && ~isempty(currFile.eventTimesDT)
%             flatDetections(flatIdx).eventStartTime = currFile.eventTimesDT(detIdx, 1);
%             flatDetections(flatIdx).eventEndTime = currFile.eventTimesDT(detIdx, 2);
%         end
% 
%         % Increment the flat index counter
%         flatIdx = flatIdx + 1;
%     end
% end
% 
% end