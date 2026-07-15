function test_adaptiveBatching_postproc()
% Verify the postprocessing chain (concatenateOverlappingProbs /
% segmentStitcher -> gavdNetPostprocess -> sample boundaries -> timestamps)
% is unaffected by adaptive batching. For each file, run the FULL gavdNetInference
% at a full batch (large memoryAvailable) vs a FORCED-SMALL batch (tiny
% memoryAvailable, so minibatchpredictAdaptive reduces the minibatch), both on
% the 4090, and check:
%   - the probability vector has the SAME LENGTH (so bin->sample->time mapping
%     is unchanged);
%   - numAudioSegments matches (segment stitching unchanged);
%   - gavdNetPostprocess returns IDENTICAL eventSampleBoundaries (so detection
%     sample indices, and hence timestamps, are identical);
%   - confidences agree to a small tolerance.
%
% Ben Jancovich, 2025

configPath = "C:\Users\z5439673\Git\GAVDNet\GAVDNet_config_DGS_chagos_exclude_chorus.m";
gavdNetDataPath = "E:\GAVDNet\Chagos_DGS\Training & Models\-10 to 10 Single Exemplar Exclude Chorus";
audioFolder = "E:\Diego Garcia South 3Ch\2001";

userGavdNetDataPath = gavdNetDataPath;
run(configPath)
addpath(fullfile("C:\Users\z5439673\Git\GAVDNet", "Functions"))
gavdNetDataPath = userGavdNetDataPath;
modelList = dir(fullfile(gavdNetDataPath, 'GAVDNet_trained_*'));
load(fullfile(modelList(1).folder, modelList(1).name))  % model
ppo = postProcOptions;
ppo.LT = model.dataSynthesisParams.meanTargetCallDuration .* ppo.LT_scaler;
ppo.maxTargetCallDuration = model.dataSynthesisParams.maxTargetCallDuration * 1.2;
maxNumCompThreads('automatic');
gpuDevice(1);   % 4090 (both runs on the same GPU, so batch size is the only difference)

files = dir(fullfile(audioFolder, 'H08S1_*.wav'));
nTest = min(6, numel(files));
% A fixed reference datetime so we can compare derived event timestamps.
t0 = datetime(2001,10,12,6,42,10);

allPass = true;
fprintf('%-4s %-8s %-9s %-9s %-9s %-7s %-9s\n', 'file', 'nProbs', 'probDiff', 'nSegEq', 'boundsEq', 'nDet', 'tsEq');
for k = 1:nTest
    fp = fullfile(files(k).folder, files(k).name);
    [a, fs] = audioread(fp); a = single(a);
    % FULL batch and FORCED-SMALL batch (tiny memoryAvailable -> small minibatch)
    [Pf, ~, ~, ~, nsf] = gavdNetInference(a, fs, model, 24e9, featureFraming, frameStandardization, minSilenceDuration, false);
    [Pr, ~, ~, ~, nsr] = gavdNetInference(a, fs, model, 5e5,  featureFraming, frameStandardization, minSilenceDuration, false);
    Pf = Pf(:); Pr = Pr(:);

    lenEq = numel(Pf) == numel(Pr);
    if lenEq; probDiff = max(abs(double(Pf) - double(Pr))); else; probDiff = Inf; end
    segEq = isequaln(nsf, nsr);

    [bf, ~, cf] = gavdNetPostprocess(a, fs, Pf, model.preprocParams, ppo);
    [br, ~, cr] = gavdNetPostprocess(a, fs, Pr, model.preprocParams, ppo);
    boundsEq = isequal(bf, br);
    nDet = size(bf, 1);
    confDiff = 0;
    if isequal(size(cf), size(cr)) && ~isempty(cf)
        confDiff = max(abs(double(cf(:)) - double(cr(:))));
    end

    % Derived event timestamps from the two boundary sets must match if the
    % boundaries match (same fileStart + sampleIndex/fs formula as run_chagos).
    tsEq = true;
    for d = 1:nDet
        tf = t0 + seconds((bf(d,1)-1)/fs);
        tr = t0 + seconds((br(d,1)-1)/fs);
        if tf ~= tr; tsEq = false; end
    end

    fprintf('%-4d %-8d %-9.2e %-9d %-9d %-7d %-9d\n', k, numel(Pf), probDiff, segEq, boundsEq, nDet, tsEq);
    allPass = allPass && lenEq && segEq && boundsEq && tsEq && (probDiff < 1e-2) && (confDiff < 1e-2);
end

if allPass
    fprintf('\nPASS: probability length, segment count, detection sample boundaries and\n');
    fprintf('derived timestamps are IDENTICAL between full-batch and reduced-batch;\n');
    fprintf('postprocessing is compatible with adaptive batching.\n');
else
    error('FAIL: adaptive batching changed a length / boundary / timestamp - see table.');
end
end
