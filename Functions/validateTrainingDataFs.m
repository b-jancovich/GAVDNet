function varargout = validateTrainingDataFs(trainingDataPath)
% VALIDATETRAININGDATAFS Check training files have consistent sample rate
%
% trainingDataPath - Full path to training data directory with .wav files
%
% Files must have identical sample rate
% Errors if files have inconsistent sample rates
%
% Optional outputs:
% fs - Sample rate (Hz) of the audio files
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
% Get list of files
fileList = dir(fullfile(trainingDataPath, "*.wav"));

% Get audio file info
parfor i = 1:length(fileList)
    info(i) = audioinfo(fullfile(fileList(i).folder, fileList(i).name));
end

% Check all files have same Fs
allSameFs = all(diff([info.SampleRate]) == 0);

% Sample rate Test, set result & report
if allSameFs == true
    fprintf('All training samples have the same sample rate.\n')
else
    error('Training samples found with non-standard sample rate. All samples must have the same sample rate.\n')
end

% Get the consistent value for sample rate
fs = info(1).SampleRate;

% If output is requested, return the Fs of the data
if nargout == 1
    varargout{1} = fs;
end
end