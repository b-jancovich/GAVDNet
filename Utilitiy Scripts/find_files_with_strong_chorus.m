% find_chorus_examples.m
%
% Script to identify and review whale song recordings containing "chorus"
% phenomena, characterized by elevated power in the 28-46 Hz band.
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

clear;
close all;
clc;


%% Configuration

wavPath = 'D:\Diego Garcia South\DiegoGarcia2017\wav';
targetMonths = {'08', '09', '10'};  % August, September, October
minDuration = 60;  % Minimum file duration (seconds)

% Frequency bands (Hz)
band1 = [0, 27]; % No call
band2 = [28, 46]; % Call
band3 = [47, Inf];  % No Call, Up to Nyquist

% Analysis parameters
windowDuration = 15;  % seconds
overlapPercent = 75;
minContinuousDuration = 240;  % seconds

% Spectrogram display parameters
specWin = 512;
specOvlp = 500;
specNFFT = 2048;
specYLim = [0, 60];
pageDuration = 600;  % Duration of each page in seconds (default 10 minutes)
defaultDynamicRange = 50;  % Default dynamic range in dB

%% Step 1: Find all wav files

fprintf('Finding wav files in: %s\n', wavPath);
wavFiles = dir(fullfile(wavPath, '*.wav'));
fprintf('Found %d wav files\n', length(wavFiles));

%% Get audio properties from first file

fprintf('Reading audio properties from first file...\n');
firstFile = fullfile(wavFiles(1).folder, wavFiles(1).name);
info = audioinfo(firstFile);
fs = info.SampleRate;
bitDepth = info.BitsPerSample;
numChannels = info.NumChannels;
bytesPerSample = bitDepth / 8;
wavHeaderBytes = 44;  % Standard WAV header size

fprintf('Audio properties: %d Hz, %d-bit, %d channel(s)\n', fs, bitDepth, numChannels);


%% Step 2: Filter by month and duration

fprintf('Filtering by month (Aug, Sep, Oct) and duration (>%d sec)...\n', minDuration);
validFiles = {};
for i = 1:length(wavFiles)
    fname = wavFiles(i).name;
    
    % Check month
    if length(fname) >= 8
        monthStr = fname(9:10);
        if ismember(monthStr, targetMonths)
            % Calculate duration from file size
            dataBytes = wavFiles(i).bytes - wavHeaderBytes;
            totalSamples = dataBytes / (bytesPerSample * numChannels);
            duration = totalSamples / fs;
            
            % Check duration
            if duration >= minDuration
                validFiles{end+1} = fullfile(wavFiles(i).folder, fname);
            end
        end
    end
end
fprintf('Retained %d files after month and duration filter\n', length(validFiles));

if isempty(validFiles)
    error('No files found matching month and duration criteria');
end

%% Step 3 & 4: Measure power and filter by Band 2 dominance
fprintf('Analyzing power in frequency bands...\n');

% Pre-allocate arrays for results
numFiles = length(validFiles);
filenames = cell(numFiles, 1);
maxDurations = zeros(numFiles, 1);

parfor i = 1:numFiles
    [y, fs_local] = audioread(validFiles{i});
    if size(y, 2) > 1
        y = mean(y, 2);  % Convert to mono
    end
    
    % Calculate window parameters in samples
    windowSamples = round(windowDuration * fs_local);
    overlapSamples = round(windowSamples * overlapPercent / 100);
    hopSamples = windowSamples - overlapSamples;
    
    % Calculate spectrogram
    [S, F, T] = spectrogram(y, windowSamples, overlapSamples, [], fs_local);
    powerSpec = abs(S).^2;
    
    % Integrate power in each band
    nyquist = fs_local / 2;
    band3_upper = min(band3(2), nyquist);
    
    idx1 = F >= band1(1) & F <= band1(2);
    idx2 = F >= band2(1) & F <= band2(2);
    idx3 = F >= band3(1) & F <= band3_upper;
    
    power1 = sum(powerSpec(idx1, :), 1);
    power2 = sum(powerSpec(idx2, :), 1);
    power3 = sum(powerSpec(idx3, :), 1);
    
    % Find windows where Band 2 dominates both Band 1 and Band 3
    band2Dominates = (power2 > power1) & (power2 > power3);
    
    % Find continuous regions
    maxDuration = findMaxContinuousDuration(band2Dominates, hopSamples / fs_local);
    
    % Store results
    filenames{i} = validFiles{i};
    maxDurations(i) = maxDuration;
end

% Filter files that meet the minimum continuous duration criterion
validIdx = maxDurations >= minContinuousDuration;

if any(validIdx)
    validFilenames = filenames(validIdx);
    validDurations = num2cell(maxDurations(validIdx));
    fileMetrics = struct('filename', validFilenames, 'maxDuration', validDurations);
else
    fileMetrics = struct('filename', {}, 'maxDuration', {});
end

fprintf('Retained %d files with >%d sec continuous Band 2 dominance\n', ...
    length(fileMetrics), minContinuousDuration);

%% Step 5: Rank files by maximum continuous duration

[~, sortIdx] = sort([fileMetrics.maxDuration], 'descend');
fileMetrics = fileMetrics(sortIdx);

fprintf('\nTop 5 files by Band 2 dominance duration:\n');
for i = 1:min(5, length(fileMetrics))
    [~, name, ext] = fileparts(fileMetrics(i).filename);
    fprintf('%d. %s%s (%.1f sec)\n', i, name, ext, fileMetrics(i).maxDuration);
end

%% Step 6: Display GUI for reviewing spectrograms

createReviewGUI(fileMetrics, specWin, specOvlp, specNFFT, specYLim, pageDuration, defaultDynamicRange);


%% Helper function: Find maximum continuous duration
function maxDuration = findMaxContinuousDuration(logicalArray, timeStep)
    % Find runs of consecutive true values
    d = diff([0; logicalArray(:); 0]);
    startIdx = find(d == 1);
    endIdx = find(d == -1) - 1;
    
    if isempty(startIdx)
        maxDuration = 0;
        return;
    end
    
    % Calculate duration of each run
    runLengths = endIdx - startIdx + 1;
    maxRunLength = max(runLengths);
    
    % Convert to time duration
    maxDuration = maxRunLength * timeStep;
end


%% Helper function: Create review GUI
function createReviewGUI(fileMetrics, specWin, specOvlp, specNFFT, specYLim, pageDuration, defaultDynamicRange)
    % Create figure
    fig = uifigure('Name', 'Chorus Example Review', 'Position', [100, 100, 1200, 650]);
    
    % Create axes for spectrogram
    ax = uiaxes(fig, 'Position', [50, 150, 1100, 450]);
    
    % Current file and page indices
    currentIdx = 1;
    currentPage = 1;
    shortlistedFiles = {};
    
    % Spectrogram data cache
    currentAudio = [];
    currentFs = [];
    currentS = [];
    currentF = [];
    currentT = [];
    totalPages = 1;
    
    % Create buttons - File navigation
    btnNext = uibutton(fig, 'Position', [50, 80, 150, 40], ...
        'Text', 'Display Next File', 'ButtonPushedFcn', @(btn, event) displayNext());
    
    btnShortlist = uibutton(fig, 'Position', [220, 80, 150, 40], ...
        'Text', 'Shortlist', 'ButtonPushedFcn', @(btn, event) shortlistFile());
    
    btnExport = uibutton(fig, 'Position', [390, 80, 150, 40], ...
        'Text', 'Export Shortlist', 'ButtonPushedFcn', @(btn, event) exportShortlist());
    
    % Page navigation buttons
    btnPrevPage = uibutton(fig, 'Position', [50, 30, 120, 40], ...
        'Text', 'Previous Page', 'ButtonPushedFcn', @(btn, event) previousPage());
    
    btnNextPage = uibutton(fig, 'Position', [180, 30, 120, 40], ...
        'Text', 'Next Page', 'ButtonPushedFcn', @(btn, event) nextPage());
    
    % Dynamic range control
    lblDynRange = uilabel(fig, 'Position', [320, 30, 100, 40], ...
        'Text', 'Dynamic Range (dB):');
    
    txtDynRange = uieditfield(fig, 'numeric', 'Position', [430, 35, 80, 30], ...
        'Value', defaultDynamicRange, 'ValueChangedFcn', @(txt, event) updateDisplay());
    
    % Info label
    lblInfo = uilabel(fig, 'Position', [530, 30, 650, 40], ...
        'Text', sprintf('File 1/%d | Page 1/1 | Shortlisted: 0', length(fileMetrics)));
    
    % Display first file
    loadNewFile();
    
    % Nested function: Load new file and compute full spectrogram
    function loadNewFile()
        if currentIdx > length(fileMetrics)
            currentIdx = 1;  % Wrap around
        end
        
        fname = fileMetrics(currentIdx).filename;
        [currentAudio, currentFs] = audioread(fname);
        if size(currentAudio, 2) > 1
            currentAudio = mean(currentAudio, 2);
        end
        
        % Calculate full spectrogram
        [currentS, currentF, currentT] = spectrogram(currentAudio, specWin, specOvlp, specNFFT, currentFs, 'yaxis');
        
        % Calculate total pages
        totalDuration = length(currentAudio) / currentFs;
        totalPages = ceil(totalDuration / pageDuration);
        
        % Reset to first page
        currentPage = 1;
        
        % Display the first page
        updateDisplay();
    end
    
    % Nested function: Update display (page or dynamic range changed)
    function updateDisplay()
        if isempty(currentS)
            return;
        end
        
        % Get dynamic range
        dynRange = txtDynRange.Value;
        
        % Calculate time limits for current page
        tStart = (currentPage - 1) * pageDuration;
        tEnd = min(currentPage * pageDuration, currentT(end));
        
        % Find time indices for current page
        pageIdx = currentT >= tStart & currentT <= tEnd;
        
        % Extract page data
        S_page = currentS(:, pageIdx);
        T_page = currentT(pageIdx);
        
        % Convert to power in dB
        S_dB = 10*log10(abs(S_page).^2);
        
        % Calculate dynamic range limits
        cMax = max(S_dB(:));
        cMin = cMax - dynRange;
        
        % Display spectrogram
        cla(ax);
        imagesc(ax, T_page, currentF, S_dB);
        axis(ax, 'xy');
        ylim(ax, specYLim);
        xlabel(ax, 'Time (s)');
        ylabel(ax, 'Frequency (Hz)');
        colormap(ax, 'jet');
        cbar = colorbar(ax);
        cbar.Label.String = 'Power (dB)';
        clim(ax, [cMin, cMax]);
        
        [~, name, ext] = fileparts(fileMetrics(currentIdx).filename);
        title(ax, sprintf('%s%s | Max Band 2 Duration: %.1f sec | Page %d/%d', ...
            name, ext, fileMetrics(currentIdx).maxDuration, currentPage, totalPages));
        
        % Update info label
        lblInfo.Text = sprintf('File %d/%d | Page %d/%d | Shortlisted: %d', ...
            currentIdx, length(fileMetrics), currentPage, totalPages, length(shortlistedFiles));
    end
    
    % Nested function: Display next file
    function displayNext()
        currentIdx = currentIdx + 1;
        if currentIdx > length(fileMetrics)
            currentIdx = 1;
        end
        loadNewFile();
    end
    
    % Nested function: Next page
    function nextPage()
        if currentPage < totalPages
            currentPage = currentPage + 1;
            updateDisplay();
        end
    end
    
    % Nested function: Previous page
    function previousPage()
        if currentPage > 1
            currentPage = currentPage - 1;
            updateDisplay();
        end
    end
    
    % Nested function: Shortlist current file with page info
    function shortlistFile()
        fname = fileMetrics(currentIdx).filename;
        
        % Calculate page time and sample indices
        tStart = (currentPage - 1) * pageDuration;
        tEnd = min(currentPage * pageDuration, length(currentAudio) / currentFs);
        sampleStart = round(tStart * currentFs) + 1;
        sampleEnd = round(tEnd * currentFs);
        
        % Create shortlist entry
        entry = struct('filename', fname, ...
                      'page', currentPage, ...
                      'timeStart', tStart, ...
                      'timeEnd', tEnd, ...
                      'sampleStart', sampleStart, ...
                      'sampleEnd', sampleEnd);
        
        % Check if this file/page combination already exists
        isDuplicate = false;
        for i = 1:length(shortlistedFiles)
            if strcmp(shortlistedFiles{i}.filename, fname) && shortlistedFiles{i}.page == currentPage
                isDuplicate = true;
                break;
            end
        end
        
        if ~isDuplicate
            shortlistedFiles{end+1} = entry;
            fprintf('Shortlisted: %s (Page %d, Samples %d-%d, Time %.1f-%.1f s)\n', ...
                fname, currentPage, sampleStart, sampleEnd, tStart, tEnd);
            lblInfo.Text = sprintf('File %d/%d | Page %d/%d | Shortlisted: %d', ...
                currentIdx, length(fileMetrics), currentPage, totalPages, length(shortlistedFiles));
        else
            fprintf('File/page already shortlisted: %s (Page %d)\n', fname, currentPage);
        end
    end
    
    % Nested function: Export shortlist
    function exportShortlist()
        if isempty(shortlistedFiles)
            uialert(fig, 'No files have been shortlisted yet.', 'Empty Shortlist');
            return;
        end
        
        % Create table from shortlist
        numEntries = length(shortlistedFiles);
        filenames = cell(numEntries, 1);
        fullpaths = cell(numEntries, 1);
        pages = zeros(numEntries, 1);
        timeStarts = zeros(numEntries, 1);
        timeEnds = zeros(numEntries, 1);
        sampleStarts = zeros(numEntries, 1);
        sampleEnds = zeros(numEntries, 1);
        
        for i = 1:numEntries
            [~, name, ext] = fileparts(shortlistedFiles{i}.filename);
            filenames{i} = [name, ext];
            fullpaths{i} = shortlistedFiles{i}.filename;
            pages(i) = shortlistedFiles{i}.page;
            timeStarts(i) = shortlistedFiles{i}.timeStart;
            timeEnds(i) = shortlistedFiles{i}.timeEnd;
            sampleStarts(i) = shortlistedFiles{i}.sampleStart;
            sampleEnds(i) = shortlistedFiles{i}.sampleEnd;
        end
        
        shortlistTable = table(filenames, fullpaths, pages, timeStarts, timeEnds, ...
            sampleStarts, sampleEnds, ...
            'VariableNames', {'Filename', 'FullPath', 'Page', 'TimeStart_s', ...
            'TimeEnd_s', 'SampleStart', 'SampleEnd'});
        
        % Save to file
        [file, path] = uiputfile('chorus_shortlist.csv', 'Save Shortlist');
        if file ~= 0
            writetable(shortlistTable, fullfile(path, file));
            fprintf('Shortlist saved to: %s\n', fullfile(path, file));
            uialert(fig, sprintf('Shortlist saved successfully!\n%d entries', length(shortlistedFiles)), ...
                'Export Complete');
        end
    end
end

% % find_chorus_examples.m
% %
% % Script to identify and review whale song recordings containing "chorus"
% % phenomena, characterized by elevated power in the 28-46 Hz band.
% %
% % Ben Jancovich, 2025
% % Centre for Marine Science and Innovation
% % School of Biological, Earth and Environmental Sciences
% % University of New South Wales, Sydney, Australia
% %
% 
% clear;
% close all;
% 
% %% Configuration
% 
% wavPath = 'D:\Diego Garcia South\DiegoGarcia2017\wav';
% targetMonths = {'08', '09', '10'};  % August, September, October
% minDuration = 60;  % Minimum file duration (seconds)
% 
% % Frequency bands (Hz)
% band1 = [0, 27]; % No call
% band2 = [28, 46]; % Call
% band3 = [47, Inf];  % No Call, Up to Nyquist
% 
% % Analysis parameters
% windowDuration = 30;  % seconds
% overlapPercent = 50;
% minContinuousDuration = 120;  % seconds
% 
% % Spectrogram display parameters
% specWin = 1024;
% specOvlp = 1000;
% specNFFT = 2048;
% specYLim = [0, 60];
% 
% %% Step 1: Find all wav files
% 
% fprintf('Finding wav files in: %s\n', wavPath);
% wavFiles = dir(fullfile(wavPath, '*.wav'));
% fprintf('Found %d wav files\n', length(wavFiles));
% 
% %% Get audio properties from first file
% 
% fprintf('Reading audio properties from first file...\n');
% firstFile = fullfile(wavFiles(1).folder, wavFiles(1).name);
% info = audioinfo(firstFile);
% fs = info.SampleRate;
% bitDepth = info.BitsPerSample;
% numChannels = info.NumChannels;
% bytesPerSample = bitDepth / 8;
% wavHeaderBytes = 44;  % Standard WAV header size
% 
% fprintf('Audio properties: %d Hz, %d-bit, %d channel(s)\n', fs, bitDepth, numChannels);
% 
% 
% %% Step 2: Filter by month and duration
% 
% fprintf('Filtering by month (Aug, Sep, Oct) and duration (>%d sec)...\n', minDuration);
% validFiles = {};
% for i = 1:length(wavFiles)
%     fname = wavFiles(i).name;
% 
%     % Check month
%     if length(fname) >= 8
%         monthStr = fname(9:10);
%         if ismember(monthStr, targetMonths)
%             % Calculate duration from file size
%             dataBytes = wavFiles(i).bytes - wavHeaderBytes;
%             totalSamples = dataBytes / (bytesPerSample * numChannels);
%             duration = totalSamples / fs;
% 
%             % Check duration
%             if duration >= minDuration
%                 validFiles{end+1} = fullfile(wavFiles(i).folder, fname);
%             end
%         end
%     end
% end
% fprintf('Retained %d files after month and duration filter\n', length(validFiles));
% 
% if isempty(validFiles)
%     error('No files found matching month and duration criteria');
% end
% 
% %% Step 3 & 4: Measure power and filter by Band 2 dominance
% fprintf('Analyzing power in frequency bands...\n');
% 
% % Pre-allocate arrays for results
% numFiles = length(validFiles);
% filenames = cell(numFiles, 1);
% maxDurations = zeros(numFiles, 1);
% 
% parfor i = 1:numFiles
%     [y, fs_local] = audioread(validFiles{i});
%     if size(y, 2) > 1
%         y = mean(y, 2);  % Convert to mono
%     end
% 
%     % Calculate window parameters in samples
%     windowSamples = round(windowDuration * fs_local);
%     overlapSamples = round(windowSamples * overlapPercent / 100);
%     hopSamples = windowSamples - overlapSamples;
% 
%     % Calculate spectrogram
%     [S, F, T] = spectrogram(y, windowSamples, overlapSamples, [], fs_local);
%     powerSpec = abs(S).^2;
% 
%     % Integrate power in each band
%     nyquist = fs_local / 2;
%     band3_upper = min(band3(2), nyquist);
% 
%     idx1 = F >= band1(1) & F <= band1(2);
%     idx2 = F >= band2(1) & F <= band2(2);
%     idx3 = F >= band3(1) & F <= band3_upper;
% 
%     power1 = sum(powerSpec(idx1, :), 1);
%     power2 = sum(powerSpec(idx2, :), 1);
%     power3 = sum(powerSpec(idx3, :), 1);
% 
%     % Find windows where Band 2 dominates both Band 1 and Band 3
%     band2Dominates = (power2 > power1) & (power2 > power3);
% 
%     % Find continuous regions
%     maxDuration = findMaxContinuousDuration(band2Dominates, hopSamples / fs_local);
% 
%     % Store results
%     filenames{i} = validFiles{i};
%     maxDurations(i) = maxDuration;
% end
% 
% % Filter files that meet the minimum continuous duration criterion
% validIdx = maxDurations >= minContinuousDuration;
% 
% if any(validIdx)
%     validFilenames = filenames(validIdx);
%     validDurations = num2cell(maxDurations(validIdx));
%     fileMetrics = struct('filename', validFilenames, 'maxDuration', validDurations);
% else
%     fileMetrics = struct('filename', {}, 'maxDuration', {});
% end
% 
% fprintf('Retained %d files with >%d sec continuous Band 2 dominance\n', ...
%     length(fileMetrics), minContinuousDuration);
% 
% %% Step 5: Rank files by maximum continuous duration
% 
% [~, sortIdx] = sort([fileMetrics.maxDuration], 'descend');
% fileMetrics = fileMetrics(sortIdx);
% 
% fprintf('\nTop 5 files by Band 2 dominance duration:\n');
% for i = 1:min(5, length(fileMetrics))
%     [~, name, ext] = fileparts(fileMetrics(i).filename);
%     fprintf('%d. %s%s (%.1f sec)\n', i, name, ext, fileMetrics(i).maxDuration);
% end
% 
% %% Step 6: Display GUI for reviewing spectrograms
% 
% createReviewGUI(fileMetrics, specWin, specOvlp, specNFFT, specYLim);
% 
% 
% %% Helper function: Find maximum continuous duration
% function maxDuration = findMaxContinuousDuration(logicalArray, timeStep)
%     % Find runs of consecutive true values
%     d = diff([0; logicalArray(:); 0]);
%     startIdx = find(d == 1);
%     endIdx = find(d == -1) - 1;
% 
%     if isempty(startIdx)
%         maxDuration = 0;
%         return;
%     end
% 
%     % Calculate duration of each run
%     runLengths = endIdx - startIdx + 1;
%     maxRunLength = max(runLengths);
% 
%     % Convert to time duration
%     maxDuration = maxRunLength * timeStep;
% end
% 
% 
% %% Helper function: Create review GUI
% 
% function createReviewGUI(fileMetrics, specWin, specOvlp, specNFFT, specYLim)
%     % Create figure
%     fig = uifigure('Name', 'Chorus Example Review', 'Position', [100, 100, 1000, 600]);
% 
%     % Create axes for spectrogram
%     ax = uiaxes(fig, 'Position', [50, 100, 900, 450]);
% 
%     % Current file index
%     currentIdx = 1;
%     shortlistedFiles = {};
% 
%     % Create buttons
%     btnNext = uibutton(fig, 'Position', [50, 30, 150, 40], ...
%         'Text', 'Display Next', 'ButtonPushedFcn', @(btn, event) displayNext());
% 
%     btnShortlist = uibutton(fig, 'Position', [220, 30, 150, 40], ...
%         'Text', 'Shortlist', 'ButtonPushedFcn', @(btn, event) shortlistFile());
% 
%     btnExport = uibutton(fig, 'Position', [390, 30, 150, 40], ...
%         'Text', 'Export Shortlist', 'ButtonPushedFcn', @(btn, event) exportShortlist());
% 
%     % Info label
%     lblInfo = uilabel(fig, 'Position', [560, 30, 400, 40], ...
%         'Text', sprintf('File 1/%d | Shortlisted: 0', length(fileMetrics)));
% 
%     % Display first file
%     displaySpectrogram();
% 
%     % Nested function: Display spectrogram
%     function displaySpectrogram()
%         if currentIdx > length(fileMetrics)
%             currentIdx = 1;  % Wrap around
%         end
% 
%         fname = fileMetrics(currentIdx).filename;
%         [y, fs] = audioread(fname);
%         if size(y, 2) > 1
%             y = mean(y, 2);
%         end
% 
%         % Calculate and display spectrogram
%         [S, F, T] = spectrogram(y, specWin, specOvlp, specNFFT, fs, 'yaxis');
% 
%         cla(ax);
%         imagesc(ax, T, F, 10*log10(abs(S)));
%         axis(ax, 'xy');
%         ylim(ax, specYLim);
%         xlabel(ax, 'Time (s)');
%         ylabel(ax, 'Frequency (Hz)');
%         colormap(ax, 'jet');
%         colorbar(ax);
% 
%         [~, name, ext] = fileparts(fname);
%         title(ax, sprintf('%s%s | Max Band 2 Duration: %.1f sec', ...
%             name, ext, fileMetrics(currentIdx).maxDuration));
% 
%         % Update info label
%         lblInfo.Text = sprintf('File %d/%d | Shortlisted: %d', ...
%             currentIdx, length(fileMetrics), length(shortlistedFiles));
%     end
% 
%     % Nested function: Display next file
%     function displayNext()
%         currentIdx = currentIdx + 1;
%         if currentIdx > length(fileMetrics)
%             currentIdx = 1;
%         end
%         displaySpectrogram();
%     end
% 
%     % Nested function: Shortlist current file
%     function shortlistFile()
%         fname = fileMetrics(currentIdx).filename;
%         if ~ismember(fname, shortlistedFiles)
%             shortlistedFiles{end+1} = fname;
%             fprintf('Shortlisted: %s\n', fname);
%             lblInfo.Text = sprintf('File %d/%d | Shortlisted: %d', ...
%                 currentIdx, length(fileMetrics), length(shortlistedFiles));
%         else
%             fprintf('File already shortlisted: %s\n', fname);
%         end
%     end
% 
%     % Nested function: Export shortlist
%     function exportShortlist()
%         if isempty(shortlistedFiles)
%             uialert(fig, 'No files have been shortlisted yet.', 'Empty Shortlist');
%             return;
%         end
% 
%         % Create table
%         [~, names, exts] = cellfun(@fileparts, shortlistedFiles, 'UniformOutput', false);
%         filenames = strcat(names, exts);
%         fullpaths = shortlistedFiles';
% 
%         shortlistTable = table(filenames', fullpaths, 'VariableNames', {'Filename', 'FullPath'});
% 
%         % Save to file
%         [file, path] = uiputfile('chorus_shortlist.csv', 'Save Shortlist');
%         if file ~= 0
%             writetable(shortlistTable, fullfile(path, file));
%             fprintf('Shortlist saved to: %s\n', fullfile(path, file));
%             uialert(fig, sprintf('Shortlist saved successfully!\n%d files', length(shortlistedFiles)), ...
%                 'Export Complete');
%         end
%     end
% end