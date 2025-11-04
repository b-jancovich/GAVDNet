%% extractNoiseLibraryFromDataset.m
%
% This script extracts noise-only segments from a dataset of passive acoustic
% monitoring recordings to build a library of background noise that can be
% used for data augmentation. The script steps through a dataset organized
% into site-year folders, containing .wav files, and:
%   1. Uses continuous wavelet transforms to identify segments containing
%      signals of interest
%   2. Extracts noise-only segments with configurable duration
%   3. Optionally allows manual inspection of extracted segments
%   4. Saves validated noise segments to disk
%
% Implementation Notes:
%   - Uses continuous wavelet transform (CWT) and 2D cross-correlation
%       for signal detection.
%   - Automatically handles mixed sample rates (assumes all wavs in a
%       single siteyear folder are the same sample rate)
%   - Includes manual inspection mode to be sure no nosie segments contain
%       the signal of interest.
%   - Randomly selects from sites, tracks number from each site to
%       ensure balanced site representation.
%   - Uses overlap-add with auto-sized crossfade to stitch together
%       short noise segments in order to make longer ones, but will not use
%       any segment with duration < 3 seconds.
%
% Expected Dataset Structure:
%   datasetPath/
%   ├── siteyear1/
%   │   └── wav/
%   │       └── (WAV files)
%   └── siteyear2/
%       └── wav/
%           └── (WAV files)
%
% Key Parameters:
%   Dataset Parameters:
%   - datasetPath: Root folder containing site-year folders
%   - savePath: Directory to save extracted noise segments
%   - signalOfInterestPath: Path to exemplar recording of target signal
%   - nNoiseOutputs: Target number of noise samples in library
%   - minPerSite: Minimum samples per site-year
%
%   Signal Detection Parameters:
%   - bandOfInterest: Frequency range of target signal [fMin fMax] (Hz)
%   - targetCallType: String specifying 'tonal' or non-tonal target
%   - corrThresh: Base correlation threshold for CWT detection [0:1]
%   - bufferSec: Time buffer around detections (s)
%
%   Noise Extraction Parameters:
%   - outputNoiseLen: Duration of saved noise segments (s)
%   - inspectSegments: Enable manual inspection of spectrograms
%
% Algorithm Overview:
%   1. Load and prepare exemplar signal
%   2. Pre-compute resamplers for different sample rates
%   3. Iterate through sites, prioritizing underrepresented ones:
%      a. Select random unprocessed file
%      b. Detect signals using CWT correlation
%      c. Extract noise segments
%      d. Optional manual validation
%      e. Save to disk
%   4. Continue until target number of segments reached
%
% Dependencies:
%   - Signal Processing Toolbox
%   - Wavelet Toolbox
%   - detectSignalPresenceCWT.m
%   - extractNoiseSegments.m
%
% Outputs:
%   Saves WAV files of noise segments to savePath. Filename format:
%   <siteyear>_<originalfile>_noiseSegment<n>.wav
%
% See also: detectSignalPresenceCWT.m, extractNoiseSegments.m
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%%
clear all
close all force
drawnow;
clc

%% User Inputs:

% Input/Output File Paths
datasetPath = "C:\Users\z5439673\OneDrive - UNSW\Documents\Detector Test Datasets\AAD_AcousticTrends_BlueFinLibrary\DATA";
savePath = "D:\SORP_BmAntZ_noise_library";
% savePath = "D:\noiseLibTest";

% Path to Exemplar recording of target signal
% signalOfInterestPath = "D:\SORP_BmAntZ_exemplars\Denoised\Bm_Ant_Z__casey2014_2014_4_21  1_20_27.695_rank1_19.3dB_RXDENOISE.wav";
signalOfInterestPath = "D:\SORP_BmAntZ_exemplars\Denoised\Bm_Ant_Z_averageExemplar.wav";

% Output Params
nNoiseOutputs = 10000;       % Target number of noise samples in library
minPerSite = 100;           % Minimum samples per site
 
% Detection CWT Params
bandOfInterest = [15, 30];  % Frequency band of interest [fMin fMax] (Hz)
targetCallType = 'tonal';   % String specifying 'tonal' or non-tonal target
corrThresh = 0.3;         % Base correlation threshold [scalar, 0:1]
bufferSec = 5;              % Time buffer around detections (s)

% Detection Energy Ratio Params ("Chorus Detector")
middleBandProportion = 0.87;      % The central proportion of the bandwidth 
%                                   of interest to consider as containing
%                                   chorus.
ratioThresh = 5;                    % Power ratio threshold - above this, 
%                                     chorus is present (dB)
minChorusDuration = 30;           % Minimum duration of power ratio over 
%                                   threshold to qualify as chorus (s)
extraEdgeBandwidth = 1;           % Additional bandwidth either side of the 
%                                   band of interest to evaluate as part of 
%                                   the in power ratio (Hz)
lowRatioTolerance = 6;            % duration of time that power ratio can 
%                                   drop below threshold, while not 
%                                   disqualifying an otherwise valid 
%                                   detection of chorus (s)

% Noise Segment Extractor Params
outputNoiseLen = 30;        % Length of noise segments saved to disk (s)
inspectSegments = true;     % Manual inspection of noise segment spectrograms

%% Add Paths

here = pwd;
detectorRoot = fileparts(here);
gitPath = here(1:strfind(here,'Git')+2);
addpath(fullfile(detectorRoot, 'Functions'));
addpath(fullfile(gitPath, 'Utilities'));
addpath(fullfile(gitPath, 'customAudioAugmenter'));

%% Begin checkpoint and File Retrieval

% Load signal of interest:
[signalOfInterestOriginal, fsSOI] = audioread(signalOfInterestPath);

% Get list of site-year folders
siteYearList = dir(datasetPath);
siteYearList = siteYearList(~ismember({siteYearList.name}, {'.', '..'}));
nSiteYears = length(siteYearList);

% Pre-create resamplers for unique sample rates
sampleRates = zeros(nSiteYears, 1);
resamplers = cell(nSiteYears, 1);
signalVersions = cell(nSiteYears, 1);

for i = 1:nSiteYears
    fileList = dir(fullfile(datasetPath, siteYearList(i).name, 'wav', '*.wav'));
    sampleRates(i) = audioinfo(fullfile(fileList(1).folder, fileList(1).name)).SampleRate;

    if sampleRates(i) ~= fsSOI

        % Design resamplers
        resamplers{i} = designArbitraryAudioResampler(fsSOI, sampleRates(i));

        % Run resamplers
        signalVersions{i} = customAudioResampler(signalOfInterestOriginal, ...
            resamplers{i});

    else
        signalVersions{i} = signalOfInterestOriginal;
    end
end

% Initialize or load checkpoint
checkpointFile = fullfile(savePath, 'noiseLibrary_extraction_checkpoint.mat');
if exist(checkpointFile, 'file')
    checkpoint = load(checkpointFile);
    totalSaves = checkpoint.totalSaves;
    siteProgress = checkpoint.siteProgress;
    processedFiles = checkpoint.processedFiles;
    fprintf('Resuming from checkpoint: %d/%d segments completed\n', ...
        totalSaves, nNoiseOutputs);
else
    totalSaves = 0;
    siteProgress = zeros(nSiteYears, 1);
    processedFiles = cell(nSiteYears, 1);
end

% Ensure no old figures remain
close all

%% Process files

while totalSaves < nNoiseOutputs
    % Prioritize underrepresented sites
    eligibleSites = find(siteProgress < minPerSite);
    if isempty(eligibleSites)
        eligibleSites = 1:nSiteYears;
    end
    folderIdx = eligibleSites(randi(length(eligibleSites)));

    % Get list of unprocessed files
    fileList = dir(fullfile(datasetPath, siteYearList(folderIdx).name, 'wav', '*.wav'));
    if isempty(processedFiles{folderIdx})
        processedFiles{folderIdx} = false(length(fileList), 1);
    end

    % Select from unprocessed files if available
    unprocessedIdx = find(~processedFiles{folderIdx});
    if isempty(unprocessedIdx)
        processedFiles{folderIdx} = false(length(fileList), 1); % Reset if all files used
        unprocessedIdx = 1:length(fileList);
    end

    fileIdx = unprocessedIdx(randi(length(unprocessedIdx)));

    % Progress Update
    fprintf('Starting next file...\n') 
    fprintf('Folder: %s\n', siteYearList(folderIdx).name)
    fprintf('File: %s\n', fileList(fileIdx).name)

    % Generate new filename from siteyear and filename
    saveName = [siteYearList(folderIdx).name, '_', fileList(fileIdx).name];

    % Read audio file
    [x, fs] = audioread(fullfile(fileList(fileIdx).folder, fileList(fileIdx).name));

    % Use pre-computed signal version
    signalOfInterest = signalVersions{folderIdx};

    % Run Energy Ratio (ER) 'Chorus' Detection
    [signalPresenceMask_ER, confidence] = detectChorus(x, bandOfInterest, ...
        middleBandProportion, extraEdgeBandwidth, minChorusDuration, ...
        lowRatioTolerance, ratioThresh, fs);

    fprintf('Mean Normd Energy Ratio for this file: %.2f (0 = Chorus Absent)\n', mean(confidence))

    % If all samples contain chorus, don't bother running the CWT, and just
    % update the checkpoint, then move on to the next file...
    if all(signalPresenceMask_ER) == true
        fprintf('Chorus is present for entire file. \nUpdating tracking and skipping further analysis...\n')
        processedFiles{folderIdx}(fileIdx) = true;
        save(checkpointFile, 'totalSaves', 'siteProgress', 'processedFiles');
        continue;
    end
    
    % Run CWT Signal Detection
    [signalPresenceMask_CWT, corrStrength] = detectSignalPresenceCWT(...
        x, signalOfInterest, bandOfInterest, targetCallType, ...
        corrThresh, bufferSec, fs);

    fprintf('Mean Correlation Magnitude for this file: %.2f (0 = Signal Absent)\n', mean(corrStrength))

    % Combine Masks
    signalPresenceMask = signalPresenceMask_CWT | signalPresenceMask_ER;

    % Run the noise segment extraction
    if inspectSegments == true
        try
            delete(findall(0, 'Type', 'figure')); % Ensure no lingering figures
            drawnow;
            [nSuccessfulSaves, continueToNextFile] = extractNoiseSegments(x, fs, ...
                outputNoiseLen, signalPresenceMask, saveName, savePath, ...
                inspectSegments, bandOfInterest);

            % Ensure cleanup after each iteration
            delete(findall(0, 'Type', 'figure'));
            drawnow;

            % Update tracking regardless of whether we continue or stop
            processedFiles{folderIdx}(fileIdx) = true;
            siteProgress(folderIdx) = siteProgress(folderIdx) + nSuccessfulSaves;
            totalSaves = totalSaves + nSuccessfulSaves;

            % Save checkpoint
            save(checkpointFile, 'totalSaves', 'siteProgress', 'processedFiles');

            % Report progress
            fprintf('Site %d: %d/%d segments (Total: %d/%d)\n', ...
                folderIdx, siteProgress(folderIdx), minPerSite, totalSaves, nNoiseOutputs);

            % If user chose to stop here, break the loop
            if ~continueToNextFile
                disp('Finishing library building session.')
                break;
            end
        catch ME
            warning(ME.identifier, 'Error in noise segment extraction: %s', ME.message);
            delete(findall(0, 'Type', 'figure'));
            drawnow;
            continue;
        end
    else
        [nSuccessfulSaves, ~] = extractNoiseSegments(x, fs, outputNoiseLen, ...
            signalPresenceMask, saveName, savePath, inspectSegments, bandOfInterest);

        % Update tracking
        processedFiles{folderIdx}(fileIdx) = true;
        siteProgress(folderIdx) = siteProgress(folderIdx) + nSuccessfulSaves;
        totalSaves = totalSaves + nSuccessfulSaves;

        % Save checkpoint
        save(checkpointFile, 'totalSaves', 'siteProgress', 'processedFiles');

        % Report progress
        fprintf('Site %d: %d/%d segments (Total: %d/%d)\n', ...
            folderIdx, siteProgress(folderIdx), minPerSite, totalSaves, nNoiseOutputs);
    end
end