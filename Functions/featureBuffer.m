function sequences = featureBuffer(features, timeBinsPerFrame, overlapPercent)
% y = featureBuffer(x,timeBinsPerFrame,overlapPercent) buffers a 
% spectrogram, x, into frames of length timeBinsPerFrame overlapped by
% overlapPercent. The sequences output are returned in a cell array for
% consumption by trainnet.

featureVectorOverlap = round((overlapPercent/100)*timeBinsPerFrame);
hopLength = timeBinsPerFrame - featureVectorOverlap;

N = floor((size(features,2) - timeBinsPerFrame)/hopLength) + 1;
sequences = cell(N,1);

idx = 1;
for jj = 1:N
    sequences{jj} = features(:,idx:idx + timeBinsPerFrame - 1);
    idx = idx + hopLength;
end

end