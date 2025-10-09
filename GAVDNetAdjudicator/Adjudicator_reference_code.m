classdef Adjudicator < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                       matlab.ui.Figure
        GridLayout                     matlab.ui.container.GridLayout
        TopBarGridLayout               matlab.ui.container.GridLayout
        SetAudioSourcePathButton       matlab.ui.control.Button
        SaveProgressExitButton         matlab.ui.control.Button
        ReviewPreviousDecisionButton   matlab.ui.control.Button
        AdjudicationTaskTextArea       matlab.ui.control.TextArea
        AdjudicationTaskTextAreaLabel  matlab.ui.control.Label
        ControlPanelGridLayout         matlab.ui.container.GridLayout
        SpectrogramControlPanel        matlab.ui.container.Panel
        GridLayout4                    matlab.ui.container.GridLayout
        DynamicRangedBEditField        matlab.ui.control.NumericEditField
        DynamicRangedBEditFieldLabel   matlab.ui.control.Label
        FFTSizepointsEditField         matlab.ui.control.NumericEditField
        FFTSizepointsEditFieldLabel    matlab.ui.control.Label
        WindowOverlapPercentEditField  matlab.ui.control.NumericEditField
        WindowOverlapPercentEditFieldLabel  matlab.ui.control.Label
        WindowSizesecondsEditField     matlab.ui.control.NumericEditField
        WindowSizesecondsEditFieldLabel  matlab.ui.control.Label
        DecisionPanel                  matlab.ui.container.Panel
        GridLayout5                    matlab.ui.container.GridLayout
        NoCallsORChorusButton          matlab.ui.control.Button
        DiscreteCallsANDChorusButton   matlab.ui.control.Button
        ChorusButton                   matlab.ui.control.Button
        DiscreteCallsButton            matlab.ui.control.Button
        SpectrogramPanel               matlab.ui.container.Panel
        TimeZoomSlider                 matlab.ui.control.RangeSlider
        FreqZoomSlider                 matlab.ui.control.RangeSlider
        UIAxes                         matlab.ui.control.UIAxes
    end

    % Properties that store app data
    properties (Access = private)
        DisagreementsData     % Structure containing the disagreements data
        CurrentIndex = 1      % Current adjudication index
        DataLoaded = false    % Flag indicating if data is loaded
        DataModified = false  % Flag indicating if data has been modified
        DataFilePath          % Path to the loaded data file
        AudioSourcePath       % Path to source audio files
        CurrentType           % Current disagreement type (FP or FN)
        FalsePositives        % False positive disagreements
        FalseNegatives        % False negative disagreements
        AudioFs               % Sample rate of current audio
        FullTimeRange         % Full time range for the current spectrogram
        FullFreqRange         % Full frequency range for the current spectrogram
        TotalAdjudications    % Total number of disagreements to adjudicate
        CurrentFreqRange = [] % Current frequency range for y-axis zoom
        SpectrogramData       % The spectrogram power data (s_display)
        SpectrogramFreq       % Frequency vector (f)
        SpectrogramTime       % Time vector (t)
    end

    % App Functions
    methods (Access = private)
        function loadDisagreementsData(app, filePath)
            % Loads the disagreement data file, either automatically finding it
            % or prompting the user to select it

            try
                if nargin < 2 || isempty(filePath)
                    % Look for files with the expected pattern in the current directory
                    appPath = fileparts(mfilename('fullpath'));
                    filePattern = fullfile(appPath, 'detector_vs_GT_disagreements_*.mat');
                    files = dir(filePattern);

                    if isempty(files)
                        % No files found, prompt user to locate
                        [fileName, filePath] = uigetfile('*.mat', 'Select disagreements data file');
                        if isequal(fileName, 0)
                            % User canceled
                            uialert(app.UIFigure, 'No data file selected. App will close.', 'Warning');
                            delete(app);
                            return;
                        end
                        filePath = fullfile(filePath, fileName);
                    else
                        % Use the most recent file
                        [~, idx] = sort([files.datenum], 'descend');
                        filePath = fullfile(files(idx(1)).folder, files(idx(1)).name);
                    end
                end

                % Load the data
                data = load(filePath);
                if ~isfield(data, 'disagreements')
                    error('Selected file does not contain disagreements data');
                end

                % Store the data
                app.FalsePositives = data.disagreements.falsePositives;
                app.FalseNegatives = data.disagreements.falseNegatives;
                app.DataFilePath = filePath;
                app.DataLoaded = true;

                % Calculate total adjudications
                app.TotalAdjudications = length(app.FalsePositives) + length(app.FalseNegatives);

                % Initialize analyst decisions if they don't exist
                if ~isfield(app.FalsePositives, 'analystDecision')
                    for i = 1:length(app.FalsePositives)
                        app.FalsePositives(i).analystDecision = '';
                    end
                end

                if ~isfield(app.FalseNegatives, 'analystDecision')
                    for i = 1:length(app.FalseNegatives)
                        app.FalseNegatives(i).analystDecision = '';
                    end
                end

                % Start with false positives
                app.CurrentType = 'FP';
                app.CurrentIndex = 1;

                % Find first unadjudicated item
                foundUnadjudicated = false;

                % First check false positives
                for i = 1:length(app.FalsePositives)
                    if isempty(app.FalsePositives(i).analystDecision)
                        app.CurrentType = 'FP';
                        app.CurrentIndex = i;
                        foundUnadjudicated = true;
                        break;
                    end
                end

                % If not found in false positives, check false negatives
                if ~foundUnadjudicated && ~isempty(app.FalseNegatives)
                    for i = 1:length(app.FalseNegatives)
                        if isempty(app.FalseNegatives(i).analystDecision)
                            app.CurrentType = 'FN';
                            app.CurrentIndex = i;
                            % foundUnadjudicated = true;
                            break;
                        end
                    end
                end

                % If no audio source path is set, prompt user
                if isempty(app.AudioSourcePath)
                    app.setAudioSourcePath();
                end

                % Update UI
                updateAdjudicationDisplay(app);
            catch ex
                uialert(app.UIFigure, ['Error loading data: ' ex.message], 'Error');
            end
        end

        function setAudioSourcePath(app)
            % Sets the path to the source audio files
            sourcePath = uigetdir('', 'Select folder containing audio files');

            if ~isequal(sourcePath, 0)
                app.AudioSourcePath = sourcePath;

                % Update display if data is already loaded
                if app.DataLoaded
                    updateAdjudicationDisplay(app);
                end
            end
        end

function updateSpectrogram(app)
            % Updates the spectrogram display based on current settings and audio
            %
            % Ben Jancovich, 2025
            % Centre for Marine Science and Innovation
            % School of Biological, Earth and Environmental Sciences
            % University of New South Wales, Sydney, Australia
            %

            if ~app.DataLoaded || isempty(app.AudioSourcePath)
                return;
            end

            % Only save current frequency range if it's not the default [0,1]
            if ~isempty(app.UIAxes.YLim) && isvalid(app.UIAxes) && ~isequal(app.UIAxes.YLim, [0,1])
                app.CurrentFreqRange = app.UIAxes.YLim;
            end

            % Get current disagreement
            if strcmp(app.CurrentType, 'FP')
                currentDisagreement = app.FalsePositives(app.CurrentIndex);
            else
                currentDisagreement = app.FalseNegatives(app.CurrentIndex);
            end

            % Get audio filename
            audioFileName = currentDisagreement.AudioFilename;

            % Get detection boundaries in samples
            if ~isnan(currentDisagreement.DetectionStartSamp) && ~isnan(currentDisagreement.DetectionEndSamp)
                startSample = currentDisagreement.DetectionStartSamp;
                % End sample index is either 'DetectionEndSamp' or 40
                % seconds from the start samp, whichever is larger.
                endSample = max(currentDisagreement.DetectionEndSamp, (startSample + (40 * currentDisagreement.AudioFs)));
            else
                % If sample boundaries are not available, try to calculate from times
                if ~isnat(currentDisagreement.DetectionStartTime) && ~isempty(audioFileName)
                    % Try to extract file start time from filename
                    fileStartTime = app.extractDatetimeFromFilename(app, audioFileName, 'datetime');
                    if ~isnat(fileStartTime)
                        % Calculate samples from time difference
                        startTimeDiff = seconds(currentDisagreement.DetectionStartTime - fileStartTime);
                        % endTimeDiff = seconds(currentDisagreement.DetectionEndTime - fileStartTime);

                        % Get file info to determine sample rate if not available
                        if isfield(currentDisagreement, 'AudioFs') && ~isnan(currentDisagreement.AudioFs)
                            app.AudioFs = currentDisagreement.AudioFs;
                        else
                            try
                                fileInfo = audioinfo(fullfile(app.AudioSourcePath, audioFileName));
                                app.AudioFs = fileInfo.SampleRate;
                            catch
                                % Default sample rate if can't determine
                                app.AudioFs = 250; % Common for baleen whale recordings
                                warning('Could not determine sample rate, using default: %d Hz', app.AudioFs);
                            end
                        end

                        startSample = max(1, round(startTimeDiff * app.AudioFs));
                        % End sample index is either 'DetectionEndSamp' or
                        % 40 seconds after the start samp, whichever is larger.
                        endSample = max(currentDisagreement.DetectionEndSamp, (startSample + (40 * currentDisagreement.AudioFs)));
                    else
                        % Could not determine file start time
                        error('Could not extract datetime from filename: %s', audioFileName);
                    end
                else
                    % No timing information available
                    error('No valid detection boundaries available');
                end
            end

            try
                % Get audio file info
                fileInfo = audioinfo(fullfile(app.AudioSourcePath, audioFileName));
                app.AudioFs = fileInfo.SampleRate;

                % Store original detection boundaries for drawing the box
                originalStartSample = startSample;
                originalEndSample = currentDisagreement.DetectionEndSamp;

                % Add 5 seconds of context before the start time
                startSample = startSample - (5 * app.AudioFs);

                % Ensure valid sample boundaries
                startSample = max(1, startSample);
                endSample = min(fileInfo.TotalSamples, endSample);

                % Check that startSample is less than endSample
                if startSample >= endSample
                    warning('Detection start sample (%d) >= end sample (%d). Adjusting...', startSample, endSample);
                    endSample = min(fileInfo.TotalSamples, startSample + round(0.5 * app.AudioFs)); % Add half a second
                end

                % Load just the section of audio we need
                [audioSegment, ~] = audioread(fullfile(app.AudioSourcePath, audioFileName), [startSample, endSample]);

                % Check if audio segment is valid
                if isempty(audioSegment) || all(audioSegment == 0)
                    warning('Audio segment is empty or contains only zeros.');
                    return;
                end
            catch ME
                warning('Could not load audio file: %s\nError: %s', audioFileName, ME.message);
                cla(app.UIAxes);
                title(app.UIAxes, sprintf('Error loading audio: %s', audioFileName));
                return;
            end
            
            % Build filter
            n = 4; % Order
            fc = 5; % Cutoff Frequency (Hz)
            nyq = app.AudioFs/2; % Nyquist frequency
            Wp = fc/nyq; % Normalized cutoff
            Rp = 0.01; % Max Passband ripple (dB)
            Rs = 100; % Target stopband rejection (dB)
            [b, a] = ellip(n, Rp, Rs, Wp, "high", "ctf");

            % High pass filter the audio
            audioSegment = ctffilt(b, a, audioSegment);

            % DC filter and normalize
            audioSegment = audioSegment - mean(audioSegment);
            maxAmp = max(abs(audioSegment));
            if maxAmp > 0
                audioSegment = audioSegment / maxAmp;
            end

            % Get spectrogram parameters
            windowSizeSec = app.WindowSizesecondsEditField.Value;
            windowSizeSamples = round(windowSizeSec * app.AudioFs);
            overlap = app.WindowOverlapPercentEditField.Value / 100;
            overlapSamples = round(windowSizeSamples * overlap);
            nfft = app.FFTSizepointsEditField.Value;

            % Ensure window size is valid
            if windowSizeSamples > length(audioSegment)
                windowSizeSamples = length(audioSegment);
                app.WindowSizesecondsEditField.Value = windowSizeSamples / app.AudioFs;
            end

            % Ensure window size is greater than 0
            if windowSizeSamples <= 0
                windowSizeSamples = min(round(app.AudioFs * 0.05), length(audioSegment));
                app.WindowSizesecondsEditField.Value = windowSizeSamples / app.AudioFs;
            end

            % Compute spectrogram
            [s, f, t] = spectrogram(audioSegment, windowSizeSamples, overlapSamples, nfft, app.AudioFs, 'yaxis');

            % Ensure spectrogram output is valid
            if isempty(s) || isempty(f) || isempty(t)
                warning('Spectrogram returned empty output. Using default values.');
                f = linspace(0, app.AudioFs/2, 100)';
                t = linspace(0, length(audioSegment)/app.AudioFs, 50);
                s = zeros(length(f), length(t));
            end

            % Convert to power in dB
            s_display = pow2db((abs(s).^2) + eps);

            % Store spectrogram data for color limit updates
            app.SpectrogramData = s_display;
            app.SpectrogramFreq = f;
            app.SpectrogramTime = t;

            % Store full time and frequency ranges
            app.FullTimeRange = [min(t), max(t)];
            app.FullFreqRange = [min(f), max(f)];

            % Double-check ranges are valid
            if app.FullTimeRange(1) >= app.FullTimeRange(2)
                app.FullTimeRange = [0, max(1, length(audioSegment)/app.AudioFs)];
            end
            if app.FullFreqRange(1) >= app.FullFreqRange(2)
                app.FullFreqRange = [0, app.AudioFs/2];
            end

            % Display spectrogram with error handling
            cla(app.UIAxes);
            imagesc(app.UIAxes, t, f, s_display);
            colormap(app.UIAxes, 'parula');
            colorbar(app.UIAxes);

            % Add red dashed box around detection boundaries
            if ~isnan(originalStartSample) && ~isnan(originalEndSample)
                hold(app.UIAxes, 'on');
                
                % Calculate detection boundaries in spectrogram time coordinates
                detectionStartTimeSpec = (originalStartSample - startSample) / app.AudioFs;
                detectionEndTimeSpec = (originalEndSample - startSample) / app.AudioFs;
                detectionDurationSec = detectionEndTimeSpec - detectionStartTimeSpec;
                
                % Use full frequency range for the box height
                freqMin = min(f);
                freqMax = max(f);
                
                % Draw rectangle around detection
                rectangle(app.UIAxes, 'Position', [detectionStartTimeSpec, freqMin, detectionDurationSec, freqMax-freqMin], ...
                    'EdgeColor', 'red', 'LineStyle', '--', 'LineWidth', 1);
                
                hold(app.UIAxes, 'off');
            end
            
            % Set axes properties
            app.UIAxes.YDir = 'normal';
            xlim(app.UIAxes, app.FullTimeRange);

            % Update time slider with valid range
            app.TimeZoomSlider.Limits = app.FullTimeRange;
            app.TimeZoomSlider.Value = app.FullTimeRange;

            % Frequency range handling with thorough validation
            if isempty(app.CurrentFreqRange) || isequal(app.CurrentFreqRange, [0,1]) || ...
                    app.CurrentFreqRange(1) >= app.CurrentFreqRange(2)
                % Invalid or default range - use full range
                app.CurrentFreqRange = app.FullFreqRange;
                ylim(app.UIAxes, app.FullFreqRange);
                app.FreqZoomSlider.Limits = app.FullFreqRange;
                app.FreqZoomSlider.Value = app.FullFreqRange;
            else
                % Check if stored range is completely outside new spectrogram limits
                if app.CurrentFreqRange(1) > max(f) || app.CurrentFreqRange(2) < min(f)
                    % Reset to full range if completely outside
                    app.CurrentFreqRange = app.FullFreqRange;
                    ylim(app.UIAxes, app.FullFreqRange);
                    app.FreqZoomSlider.Limits = app.FullFreqRange;
                    app.FreqZoomSlider.Value = app.FullFreqRange;
                else
                    % Ensure the stored range is within the current spectrogram limits
                    validFreqRange = [max(min(f), app.CurrentFreqRange(1)), min(max(f), app.CurrentFreqRange(2))];

                    % Double-check that validFreqRange is valid
                    if validFreqRange(1) >= validFreqRange(2)
                        validFreqRange = app.FullFreqRange;
                    end

                    ylim(app.UIAxes, validFreqRange);
                    app.FreqZoomSlider.Limits = app.FullFreqRange;
                    app.FreqZoomSlider.Value = validFreqRange;
                    app.CurrentFreqRange = validFreqRange;
                end
            end

            % Set axes labels
            xlabel(app.UIAxes, 'Time (seconds)');
            ylabel(app.UIAxes, 'Frequency (Hz)');
            title(app.UIAxes, 'Detection');
            grid(app.UIAxes, 'on');

            % Update color limits based on visible frequency range
            updateColorLimits(app);
        end

        function updateAdjudicationDisplay(app)
            % Updates the display for the current adjudication task

            if ~app.DataLoaded
                return;
            end

            % Calculate total decision progress
            totalDecided = 0;
            for i = 1:length(app.FalsePositives)
                if ~isempty(app.FalsePositives(i).analystDecision)
                    totalDecided = totalDecided + 1;
                end
            end

            for i = 1:length(app.FalseNegatives)
                if ~isempty(app.FalseNegatives(i).analystDecision)
                    totalDecided = totalDecided + 1;
                end
            end

            % Calculate current overall index (independent of FP/FN distinction)
            % currentOverallIndex = 0;
            if strcmp(app.CurrentType, 'FP')
                currentOverallIndex = app.CurrentIndex;
            else
                currentOverallIndex = length(app.FalsePositives) + app.CurrentIndex;
            end

            % Get current disagreement
            if strcmp(app.CurrentType, 'FP')
                currentDisagreement = app.FalsePositives(app.CurrentIndex);
            else
                currentDisagreement = app.FalseNegatives(app.CurrentIndex);
            end

            % Update adjudication task information (without revealing FP/FN status)
            taskText = sprintf('Adjudication %d of %d\nProgress: %d/%d decisions made\nFile: %s', ...
                currentOverallIndex, app.TotalAdjudications, totalDecided, app.TotalAdjudications, ...
                currentDisagreement.AudioFilename);

            app.AdjudicationTaskTextArea.Value = taskText;

            % Update spectrogram
            try
                updateSpectrogram(app);
            catch ME
                warning(ME.identifier, 'Error updating spectrogram: %s', ME.message);
                % Clear the axes and display error message
                cla(app.UIAxes);
                text(app.UIAxes, 0.5, 0.5, sprintf('Error: %s', ME.message), ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
            end

            % Enable/disable review button (can go back if not at the first item)
            app.ReviewPreviousDecisionButton.Enable = (currentOverallIndex > 1);
        end

        function saveProgress(app)
            % Saves the current progress to the data file

            if ~app.DataLoaded || ~app.DataModified
                return;
            end

            try
                % Load the entire original file to preserve all variables
                originalData = load(app.DataFilePath);

                % Update the disagreements structure with the analyst decisions
                originalData.disagreements.falsePositives = app.FalsePositives;
                originalData.disagreements.falseNegatives = app.FalseNegatives;

                % Save all variables back to the file
                save(app.DataFilePath, '-struct', 'originalData', '-v7.3');

                app.DataModified = false;
                uialert(app.UIFigure, 'Progress saved successfully', 'Success', 'Icon', 'success');
            catch ex
                uialert(app.UIFigure, ['Error saving progress: ' ex.message], 'Error');
            end
        end

        function decisionButtonCallback(app, decision)
            % Handles decision button clicks

            if ~app.DataLoaded
                return;
            end

            % Record decision
            if strcmp(app.CurrentType, 'FP')
                if app.CurrentIndex <= length(app.FalsePositives)
                    app.FalsePositives(app.CurrentIndex).analystDecision = decision;
                    app.DataModified = true;
                end
            else
                if app.CurrentIndex <= length(app.FalseNegatives)
                    app.FalseNegatives(app.CurrentIndex).analystDecision = decision;
                    app.DataModified = true;
                end
            end

            % Find next unadjudicated item
            found = false;

            % If we're in FP and there are more FPs to check
            if strcmp(app.CurrentType, 'FP') && app.CurrentIndex < length(app.FalsePositives)
                for i = app.CurrentIndex + 1:length(app.FalsePositives)
                    if isempty(app.FalsePositives(i).analystDecision)
                        app.CurrentIndex = i;
                        found = true;
                        break;
                    end
                end
            end

            % If not found and we're still in FP, check FN
            if ~found && strcmp(app.CurrentType, 'FP') && ~isempty(app.FalseNegatives)
                app.CurrentType = 'FN';
                for i = 1:length(app.FalseNegatives)
                    if isempty(app.FalseNegatives(i).analystDecision)
                        app.CurrentIndex = i;
                        found = true;
                        break;
                    end
                end
            end

            % If we're in FN and there are more FNs to check
            if ~found && strcmp(app.CurrentType, 'FN') && app.CurrentIndex < length(app.FalseNegatives)
                for i = app.CurrentIndex + 1:length(app.FalseNegatives)
                    if isempty(app.FalseNegatives(i).analystDecision)
                        app.CurrentIndex = i;
                        found = true;
                        break;
                    end
                end
            end

            if found
                updateAdjudicationDisplay(app);
            else
                % All items adjudicated
                msg = 'All disagreements have been adjudicated!';
                uialert(app.UIFigure, msg, 'Complete', 'Icon', 'success');
                saveProgress(app);
            end
        end

        function findPreviousDecision(app)
            % Finds the previous item to review

            if strcmp(app.CurrentType, 'FP')
                if app.CurrentIndex > 1
                    app.CurrentIndex = app.CurrentIndex - 1;
                end
            else % FN
                if app.CurrentIndex > 1
                    app.CurrentIndex = app.CurrentIndex - 1;
                else
                    % Switch to FP and go to last FP
                    if ~isempty(app.FalsePositives)
                        app.CurrentType = 'FP';
                        app.CurrentIndex = length(app.FalsePositives);
                    end
                end
            end
            updateAdjudicationDisplay(app);
        end

        function dt = extractDatetimeFromFilename(app, filename, outFormat)

            % Initialize output
            if strcmp(outFormat, 'datetime') == true
                dt = NaT; % Not-a-Time for datetime
            else
                dt = NaN;
            end

            % Remove file extension and any directory path
            [~, filename, ~] = fileparts(filename);

            % Try the most common and efficient patterns first

            % Pattern 1: YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS (most efficient)
            tokens = regexp(filename, '(\d{8})[_-](\d{6})', 'tokens', 'once');
            if ~isempty(tokens)
                try
                    if strcmp(outFormat, 'datetime') == true
                        dt = datetime([tokens{1}, tokens{2}], 'InputFormat', 'yyyyMMddHHmmss');
                    else
                        dt = datenum([tokens{1}, tokens{2}], 'yyyymmddHHMMSS');
                    end
                    return;
                catch
                    % Continue if this fails
                end
            end

            % Pattern 2: YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS broken down
            tokens = regexp(filename, '^(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})', 'tokens', 'once');
            if ~isempty(tokens)
                try
                    % Pre-convert to numbers for faster datetime creation
                    year = str2double(tokens{1});
                    month = str2double(tokens{2});
                    day = str2double(tokens{3});
                    hour = str2double(tokens{4});
                    minute = str2double(tokens{5});
                    second = str2double(tokens{6});
                    if strcmp(outFormat, 'datetime') == true
                        dt = datetime(year, month, day, hour, minute, second);
                    else
                        dt = datenum(year, month, day, hour, minute, second);
                    end
                    return;
                catch
                    % Continue if this fails
                end
            end

            % Pattern 3: xxx_YYMMDD-HHMMSS (special case with 2-digit year)
            tokens = regexp(filename, '([A-Za-z0-9]+)_(\d{6})-(\d{6})', 'tokens', 'once');
            if ~isempty(tokens)
                try
                    date_str = tokens{2};
                    time_str = tokens{3};
                    year = str2double(['20', date_str(1:2)]); % Assuming 20xx for the century
                    month = str2double(date_str(3:4));
                    day = str2double(date_str(5:6));
                    hour = str2double(time_str(1:2));
                    minute = str2double(time_str(3:4));
                    second = str2double(time_str(5:6));
                    if strcmp(outFormat, 'datetime') == true
                        dt = datetime(year, month, day, hour, minute, second);
                    else
                        dt = datenum(year, month, day, hour, minute, second);
                    end
                    return;
                catch
                    % Continue if this fails
                end
            end

            % Check remaining patterns in order of likely frequency
            remaining_patterns = {
                '^(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})', ... % YYYY-MM-DD_HH-mm-SS or YYYY-MM-DD-HH-mm-SS
                '^(\d{4})-(\d{2})-(\d{2})[_-](\d{2}):(\d{2}):(\d{2})', ... % YYYY-MM-DD_HH:mm:SS or YYYY-MM-DD-HH:mm:SS
                '^(\d{4})(\d{2})(\d{2})[_-](\d{2}):(\d{2}):(\d{2})', ... % YYYYMMDD_HH:mm:SS or YYYYMMDD-HH:mm:SS
                '^(\d+)_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})' ... % xxx_YYYY-MM-DD_HH-mm-SS
                };

            for i = 1:length(remaining_patterns)
                tokens = regexp(filename, remaining_patterns{i}, 'tokens', 'once');
                if ~isempty(tokens)
                    try
                        % For the first three patterns
                        if i <= 3
                            % Pre-convert to numbers for faster datetime creation
                            year = str2double(tokens{1});
                            month = str2double(tokens{2});
                            day = str2double(tokens{3});
                            hour = str2double(tokens{4});
                            minute = str2double(tokens{5});
                            second = str2double(tokens{6});
                        else
                            % For the pattern with prefix
                            year = str2double(tokens{2});
                            month = str2double(tokens{3});
                            day = str2double(tokens{4});
                            hour = str2double(tokens{5});
                            minute = str2double(tokens{6});
                            second = str2double(tokens{7});
                        end
                        if strcmp(outFormat, 'datetime') == true
                            dt = datetime(year, month, day, hour, minute, second);
                        else
                            dt = datenum(year, month, day, hour, minute, second);
                        end
                        return;
                    catch
                        continue;
                    end
                end
            end

        end

        function updateColorLimits(app)
            % Updates color limits based on currently visible frequency range

            if isempty(app.SpectrogramData) || isempty(app.SpectrogramFreq)
                return;
            end

            % Get current y-axis limits (frequency range)
            currentFreqLimits = ylim(app.UIAxes);

            % Find frequency indices corresponding to visible range
            freqIndices = (app.SpectrogramFreq >= currentFreqLimits(1)) & ...
                (app.SpectrogramFreq <= currentFreqLimits(2));

            if ~any(freqIndices)
                return; % No valid frequency indices
            end

            % Extract visible portion of spectrogram
            visibleSpectrogram = app.SpectrogramData(freqIndices, :);

            % Calculate color limits based on visible data
            dynamicRange = app.DynamicRangedBEditField.Value;
            cMax = max(visibleSpectrogram, [], 'all');
            cMin = cMax - dynamicRange;

            % Apply new color limits
            clim(app.UIAxes, [cMin, cMax]);
        end
    end


    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)

            % Initialize audio source path
            app.AudioSourcePath = '';

            % Load data file at startup
            loadDisagreementsData(app);
        end

        % Button pushed function: DiscreteCallsButton
        function DiscreteCallsButtonPushed(app, event)
            decisionButtonCallback(app, "DiscreteCallsPresent");
        end

        % Button pushed function: ChorusButton
        function ChorusButtonPushed(app, event)
            decisionButtonCallback(app, "ChorusPresent");
        end

        % Button pushed function: DiscreteCallsANDChorusButton
        function DiscreteCallsANDChorusButtonPushed(app, event)
            decisionButtonCallback(app, "DiscreteCallsChorusPresent");
        end

        % Button pushed function: NoCallsORChorusButton
        function NoCallsORChorusButtonPushed(app, event)
            decisionButtonCallback(app, "CallChorusAbsent");
        end

        % Button pushed function: ReviewPreviousDecisionButton
        function ReviewPreviousDecisionButtonPushed(app, event)
            findPreviousDecision(app);
        end

        % Button pushed function: SaveProgressExitButton
        function SaveProgressExitButtonPushed(app, event)
            saveProgress(app);
            delete(app);
        end

        % Value changed function: TimeZoomSlider
        function TimeZoomSliderValueChanged(app, event)
            xlim(app.UIAxes, app.TimeZoomSlider.Value);
        end

        % Value changed function: FreqZoomSlider
        function FreqZoomSliderValueChanged(app, event)
            ylim(app.UIAxes, app.FreqZoomSlider.Value);
            updateColorLimits(app); % Update color limits for new frequency range
        end

        % Value changed function: WindowSizesecondsEditField
        function WindowSizesecondsEditFieldValueChanged(app, event)
            updateSpectrogram(app);
        end

        % Value changed function: WindowOverlapPercentEditField
        function WindowOverlapPercentEditFieldValueChanged(app, event)
            updateSpectrogram(app);
        end

        % Value changed function: FFTSizepointsEditField
        function FFTSizepointsEditFieldValueChanged(app, event)
            updateSpectrogram(app);
        end

        % Value changed function: DynamicRangedBEditField
        function DynamicRangedBEditFieldValueChanged(app, event)
            updateSpectrogram(app);
        end

        % Close request function: UIFigure
        function UIFigureCloseRequest(app, event)
            saveProgress(app);
            delete(app)
        end

        % Button pushed function: SetAudioSourcePathButton
        function SetAudioSourcePathButtonPushed(app, event)
            setAudioSourcePath(app);
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 1000 700];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {'1x'};
            app.GridLayout.RowHeight = {'0.5x', '3x', '1.5x'};

            % Create SpectrogramPanel
            app.SpectrogramPanel = uipanel(app.GridLayout);
            app.SpectrogramPanel.Title = 'Spectrogram';
            app.SpectrogramPanel.Layout.Row = 2;
            app.SpectrogramPanel.Layout.Column = 1;

            % Create UIAxes
            app.UIAxes = uiaxes(app.SpectrogramPanel);
            xlabel(app.UIAxes, 'Time (s)')
            ylabel(app.UIAxes, 'Frequency (Hz)')
            zlabel(app.UIAxes, 'Power (dB)')
            app.UIAxes.Layer = 'top';
            app.UIAxes.XGrid = 'on';
            app.UIAxes.YGrid = 'on';
            app.UIAxes.Position = [53 44 909 331];

            % Create FreqZoomSlider
            app.FreqZoomSlider = uislider(app.SpectrogramPanel, 'range');
            app.FreqZoomSlider.Orientation = 'vertical';
            app.FreqZoomSlider.ValueChangedFcn = createCallbackFcn(app, @FreqZoomSliderValueChanged, true);
            app.FreqZoomSlider.Position = [7 78 3 274];

            % Create TimeZoomSlider
            app.TimeZoomSlider = uislider(app.SpectrogramPanel, 'range');
            app.TimeZoomSlider.ValueChangedFcn = createCallbackFcn(app, @TimeZoomSliderValueChanged, true);
            app.TimeZoomSlider.Position = [91 27 868 3];

            % Create ControlPanelGridLayout
            app.ControlPanelGridLayout = uigridlayout(app.GridLayout);
            app.ControlPanelGridLayout.RowHeight = {'1x'};
            app.ControlPanelGridLayout.Layout.Row = 3;
            app.ControlPanelGridLayout.Layout.Column = 1;

            % Create DecisionPanel
            app.DecisionPanel = uipanel(app.ControlPanelGridLayout);
            app.DecisionPanel.Title = 'Decision';
            app.DecisionPanel.Layout.Row = 1;
            app.DecisionPanel.Layout.Column = 1;

            % Create GridLayout5
            app.GridLayout5 = uigridlayout(app.DecisionPanel);
            app.GridLayout5.ColumnWidth = {'1x', '1x', '1x', '1x'};
            app.GridLayout5.RowHeight = {'1x'};

            % Create DiscreteCallsButton
            app.DiscreteCallsButton = uibutton(app.GridLayout5, 'push');
            app.DiscreteCallsButton.ButtonPushedFcn = createCallbackFcn(app, @DiscreteCallsButtonPushed, true);
            app.DiscreteCallsButton.WordWrap = 'on';
            app.DiscreteCallsButton.BackgroundColor = [0.3922 0.8314 0.0745];
            app.DiscreteCallsButton.FontSize = 18;
            app.DiscreteCallsButton.FontWeight = 'bold';
            app.DiscreteCallsButton.Layout.Row = 1;
            app.DiscreteCallsButton.Layout.Column = 1;
            app.DiscreteCallsButton.Text = 'Discrete Call(s)';

            % Create ChorusButton
            app.ChorusButton = uibutton(app.GridLayout5, 'push');
            app.ChorusButton.ButtonPushedFcn = createCallbackFcn(app, @ChorusButtonPushed, true);
            app.ChorusButton.WordWrap = 'on';
            app.ChorusButton.BackgroundColor = [0.3922 0.8314 0.0745];
            app.ChorusButton.FontSize = 18;
            app.ChorusButton.FontWeight = 'bold';
            app.ChorusButton.Layout.Row = 1;
            app.ChorusButton.Layout.Column = 2;
            app.ChorusButton.Text = 'Chorus';

            % Create DiscreteCallsANDChorusButton
            app.DiscreteCallsANDChorusButton = uibutton(app.GridLayout5, 'push');
            app.DiscreteCallsANDChorusButton.ButtonPushedFcn = createCallbackFcn(app, @DiscreteCallsANDChorusButtonPushed, true);
            app.DiscreteCallsANDChorusButton.WordWrap = 'on';
            app.DiscreteCallsANDChorusButton.BackgroundColor = [0.3922 0.8314 0.0745];
            app.DiscreteCallsANDChorusButton.FontSize = 18;
            app.DiscreteCallsANDChorusButton.FontWeight = 'bold';
            app.DiscreteCallsANDChorusButton.Layout.Row = 1;
            app.DiscreteCallsANDChorusButton.Layout.Column = 3;
            app.DiscreteCallsANDChorusButton.Text = 'Discrete Call(s) AND Chorus';

            % Create NoCallsORChorusButton
            app.NoCallsORChorusButton = uibutton(app.GridLayout5, 'push');
            app.NoCallsORChorusButton.ButtonPushedFcn = createCallbackFcn(app, @NoCallsORChorusButtonPushed, true);
            app.NoCallsORChorusButton.WordWrap = 'on';
            app.NoCallsORChorusButton.BackgroundColor = [1 0 0];
            app.NoCallsORChorusButton.FontSize = 18;
            app.NoCallsORChorusButton.FontWeight = 'bold';
            app.NoCallsORChorusButton.Layout.Row = 1;
            app.NoCallsORChorusButton.Layout.Column = 4;
            app.NoCallsORChorusButton.Text = 'No Call(s) OR Chorus';

            % Create SpectrogramControlPanel
            app.SpectrogramControlPanel = uipanel(app.ControlPanelGridLayout);
            app.SpectrogramControlPanel.Title = 'Spectrogram Control';
            app.SpectrogramControlPanel.Layout.Row = 1;
            app.SpectrogramControlPanel.Layout.Column = 2;

            % Create GridLayout4
            app.GridLayout4 = uigridlayout(app.SpectrogramControlPanel);
            app.GridLayout4.RowHeight = {'1x', '1x', '1x', '1x'};

            % Create WindowSizesecondsEditFieldLabel
            app.WindowSizesecondsEditFieldLabel = uilabel(app.GridLayout4);
            app.WindowSizesecondsEditFieldLabel.HorizontalAlignment = 'right';
            app.WindowSizesecondsEditFieldLabel.Layout.Row = 1;
            app.WindowSizesecondsEditFieldLabel.Layout.Column = 1;
            app.WindowSizesecondsEditFieldLabel.Text = 'Window Size (seconds)';

            % Create WindowSizesecondsEditField
            app.WindowSizesecondsEditField = uieditfield(app.GridLayout4, 'numeric');
            app.WindowSizesecondsEditField.Limits = [0.01 Inf];
            app.WindowSizesecondsEditField.ValueChangedFcn = createCallbackFcn(app, @WindowSizesecondsEditFieldValueChanged, true);
            app.WindowSizesecondsEditField.HorizontalAlignment = 'left';
            app.WindowSizesecondsEditField.Layout.Row = 1;
            app.WindowSizesecondsEditField.Layout.Column = 2;
            app.WindowSizesecondsEditField.Value = 1.25;

            % Create WindowOverlapPercentEditFieldLabel
            app.WindowOverlapPercentEditFieldLabel = uilabel(app.GridLayout4);
            app.WindowOverlapPercentEditFieldLabel.HorizontalAlignment = 'right';
            app.WindowOverlapPercentEditFieldLabel.Layout.Row = 2;
            app.WindowOverlapPercentEditFieldLabel.Layout.Column = 1;
            app.WindowOverlapPercentEditFieldLabel.Text = 'Window Overlap (Percent)';

            % Create WindowOverlapPercentEditField
            app.WindowOverlapPercentEditField = uieditfield(app.GridLayout4, 'numeric');
            app.WindowOverlapPercentEditField.Limits = [0 99];
            app.WindowOverlapPercentEditField.RoundFractionalValues = 'on';
            app.WindowOverlapPercentEditField.ValueDisplayFormat = '%.0f';
            app.WindowOverlapPercentEditField.ValueChangedFcn = createCallbackFcn(app, @WindowOverlapPercentEditFieldValueChanged, true);
            app.WindowOverlapPercentEditField.HorizontalAlignment = 'left';
            app.WindowOverlapPercentEditField.Layout.Row = 2;
            app.WindowOverlapPercentEditField.Layout.Column = 2;
            app.WindowOverlapPercentEditField.Value = 99;

            % Create FFTSizepointsEditFieldLabel
            app.FFTSizepointsEditFieldLabel = uilabel(app.GridLayout4);
            app.FFTSizepointsEditFieldLabel.HorizontalAlignment = 'right';
            app.FFTSizepointsEditFieldLabel.Layout.Row = 3;
            app.FFTSizepointsEditFieldLabel.Layout.Column = 1;
            app.FFTSizepointsEditFieldLabel.Text = 'FFT Size (points)';

            % Create FFTSizepointsEditField
            app.FFTSizepointsEditField = uieditfield(app.GridLayout4, 'numeric');
            app.FFTSizepointsEditField.Limits = [64 10240];
            app.FFTSizepointsEditField.RoundFractionalValues = 'on';
            app.FFTSizepointsEditField.ValueChangedFcn = createCallbackFcn(app, @FFTSizepointsEditFieldValueChanged, true);
            app.FFTSizepointsEditField.HorizontalAlignment = 'left';
            app.FFTSizepointsEditField.Layout.Row = 3;
            app.FFTSizepointsEditField.Layout.Column = 2;
            app.FFTSizepointsEditField.Value = 2048;

            % Create DynamicRangedBEditFieldLabel
            app.DynamicRangedBEditFieldLabel = uilabel(app.GridLayout4);
            app.DynamicRangedBEditFieldLabel.HorizontalAlignment = 'right';
            app.DynamicRangedBEditFieldLabel.Layout.Row = 4;
            app.DynamicRangedBEditFieldLabel.Layout.Column = 1;
            app.DynamicRangedBEditFieldLabel.Text = 'Dynamic Range (dB)';

            % Create DynamicRangedBEditField
            app.DynamicRangedBEditField = uieditfield(app.GridLayout4, 'numeric');
            app.DynamicRangedBEditField.Limits = [0 Inf];
            app.DynamicRangedBEditField.ValueChangedFcn = createCallbackFcn(app, @DynamicRangedBEditFieldValueChanged, true);
            app.DynamicRangedBEditField.HorizontalAlignment = 'left';
            app.DynamicRangedBEditField.Layout.Row = 4;
            app.DynamicRangedBEditField.Layout.Column = 2;
            app.DynamicRangedBEditField.Value = 30;

            % Create TopBarGridLayout
            app.TopBarGridLayout = uigridlayout(app.GridLayout);
            app.TopBarGridLayout.ColumnWidth = {'0.2x', '0.3x', '0.2x', '0.2x', '0.2x'};
            app.TopBarGridLayout.RowHeight = {'1x'};
            app.TopBarGridLayout.Layout.Row = 1;
            app.TopBarGridLayout.Layout.Column = 1;

            % Create AdjudicationTaskTextAreaLabel
            app.AdjudicationTaskTextAreaLabel = uilabel(app.TopBarGridLayout);
            app.AdjudicationTaskTextAreaLabel.FontWeight = 'bold';
            app.AdjudicationTaskTextAreaLabel.Layout.Row = 1;
            app.AdjudicationTaskTextAreaLabel.Layout.Column = 1;
            app.AdjudicationTaskTextAreaLabel.Text = 'Adjudication Task #';

            % Create AdjudicationTaskTextArea
            app.AdjudicationTaskTextArea = uitextarea(app.TopBarGridLayout);
            app.AdjudicationTaskTextArea.FontWeight = 'bold';
            app.AdjudicationTaskTextArea.Layout.Row = 1;
            app.AdjudicationTaskTextArea.Layout.Column = 2;

            % Create ReviewPreviousDecisionButton
            app.ReviewPreviousDecisionButton = uibutton(app.TopBarGridLayout, 'push');
            app.ReviewPreviousDecisionButton.ButtonPushedFcn = createCallbackFcn(app, @ReviewPreviousDecisionButtonPushed, true);
            app.ReviewPreviousDecisionButton.Layout.Row = 1;
            app.ReviewPreviousDecisionButton.Layout.Column = 4;
            app.ReviewPreviousDecisionButton.Text = 'Review Previous Decision';

            % Create SaveProgressExitButton
            app.SaveProgressExitButton = uibutton(app.TopBarGridLayout, 'push');
            app.SaveProgressExitButton.ButtonPushedFcn = createCallbackFcn(app, @SaveProgressExitButtonPushed, true);
            app.SaveProgressExitButton.Layout.Row = 1;
            app.SaveProgressExitButton.Layout.Column = 5;
            app.SaveProgressExitButton.Text = 'Save Progress & Exit';

            % Create SetAudioSourcePathButton
            app.SetAudioSourcePathButton = uibutton(app.TopBarGridLayout, 'push');
            app.SetAudioSourcePathButton.ButtonPushedFcn = createCallbackFcn(app, @SetAudioSourcePathButtonPushed, true);
            app.SetAudioSourcePathButton.Layout.Row = 1;
            app.SetAudioSourcePathButton.Layout.Column = 3;
            app.SetAudioSourcePathButton.Text = 'Set Audio Source Path';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Adjudicator

            runningApp = getRunningApp(app);

            % Check for running singleton app
            if isempty(runningApp)

                % Create UIFigure and components
                createComponents(app)

                % Register the app with App Designer
                registerApp(app, app.UIFigure)

                % Execute the startup function
                runStartupFcn(app, @startupFcn)
            else

                % Focus the running singleton app
                figure(runningApp.UIFigure)

                app = runningApp;
            end

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end