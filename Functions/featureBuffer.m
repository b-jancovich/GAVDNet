function sequences = featureBuffer(features, featureVectorsPerSequence, overlapPercent)
% y = featureBuffer(x,sequenceLength,overlapPercent) buffers a sequence of
% feature vectors, x, into sequences of length sequenceLength overlapped by
% overlapPercent. The sequences output are returned in a cell array for
% consumption by trainnet.

featureVectorOverlap = round(overlapPercent*featureVectorsPerSequence);
hopLength = featureVectorsPerSequence - featureVectorOverlap;

N = floor((size(features,2) - featureVectorsPerSequence)/hopLength) + 1;
sequences = cell(N,1);

idx = 1;
for jj = 1:N
    sequences{jj} = features(:,idx:idx + featureVectorsPerSequence - 1);
    idx = idx + hopLength;
end

end