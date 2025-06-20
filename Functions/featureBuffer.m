function featureFrames = featureBuffer(features, timeBinsPerFrame, overlapPercent, options)
% featureFrames = featureBuffer(x,timeBinsPerFrame,overlapPercent,Name,Value) 
% buffers a spectrogram, x, into frames of length timeBinsPerFrame overlapped 
% by overlapPercent. The sequences output are returned in a cell array for
% consumption by trainnet.
%
% Name-Value Arguments:
%   leftoverTimeBins - "discard" (default) or "keep". Controls whether
%                      incomplete final frames are discarded or kept.
%   standardizeFrames - logical (default false). If true, each frame is
%                       standardized using standardizeMelSpect.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

arguments
    features
    timeBinsPerFrame
    overlapPercent
    options.leftoverTimeBins {mustBeMember(options.leftoverTimeBins, ["discard", "keep"])} = "discard"
    options.standardizeFrames logical = false
end

featureVectorOverlap = round((overlapPercent/100)*timeBinsPerFrame);
hopLength = timeBinsPerFrame - featureVectorOverlap;
N = floor((size(features,2) - timeBinsPerFrame)/hopLength) + 1;

% Check for leftover bins
lastFrameEnd = 1 + (N-1)*hopLength + timeBinsPerFrame - 1;
hasLeftovers = lastFrameEnd < size(features,2) && options.leftoverTimeBins == "keep";

if hasLeftovers
    featureFrames = cell(N+1,1);
else
    featureFrames = cell(N,1);
end

idx = 1;
for jj = 1:N
 featureFrames{jj} = features(:,idx:idx + timeBinsPerFrame - 1);
 if options.standardizeFrames
     featureFrames{jj} = standardizeMelSpect(featureFrames{jj});
 end
 idx = idx + hopLength;
end

% Handle leftover bins if keeping them
if hasLeftovers
    featureFrames{N+1} = features(:, lastFrameEnd+1:end);
    if options.standardizeFrames
        featureFrames{N+1} = standardizeMelSpect(featureFrames{N+1});
    end
end

end

function y = standardizeMelSpect(x)
%standardizeMelSpect Standardize audio to zero mean and unity standard deviation
amean = mean(x,2);
astd = max(std(x,[],2), 1e-10);
y = (x-amean)./astd;
end