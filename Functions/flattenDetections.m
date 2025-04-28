function flatDetections = flattenDetections(detections)
% FLATTENDETECTIONS Converts hierarchical detections struct to flat format
%   Takes a detections struct with one row per audio file (containing multiple
%   detections) and returns a flattened struct with one row per detection.
%
% Inputs:
%   detections - Original struct with one entry per file
%
% Outputs:
%   flatDetections - Flattened struct with one entry per detection
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

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
        
        % Increment the flat index counter
        flatIdx = flatIdx + 1;
    end
end

end