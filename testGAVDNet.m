% Test VADNet

% Chagos VADNet Training data
vadnetTrainDataPath = "D:\VADNet_Training\Chagos_DGS";

% Load model
modelList = dir(fullfile(vadnetTrainDataPath, 'trainedVADNet_*'));
if isscalar(modelList)
    load(fullfile(modelList.folder, modelList.name))
else
    [file, location] = uigetfile(vadnetTrainDataPath, 'Select a model to load:');
    load(fullfile(location, file))
end
