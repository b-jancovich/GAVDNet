function timeVector = createDateTimeVector(startTime, numPoints, fs)
% CREATETIMEVECTOR Creates a time vector with specified parameters
% timeVector = createDateTimeVector(startTime, durationSeconds, dt, numPoints)
% creates a time vector starting at startTime, lasting for durationSeconds,
% with numPoints data points. The function validates that dt = durationSeconds/(numPoints-1).
%
% Inputs:
% - startTime: 1x1 datetime object or MATLAB serial datenum representing the start time
% - numPoints: Number of points in the time vector
% - fs: sampling frequency (numPoints per second aka. Hz)
%
% Output:
% - timeVector: Time vector with the same type as startTime (datetime or datenum)
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

% Calculate time interval
dt = 1/fs;

% Calculate duration in seconds
durationSeconds = numPoints / fs;

% Determine input type and process accordingly
isDatetime = isa(startTime, 'datetime');

if isDatetime
    % For datetime input, create vector using linspace for full duration
    secondsOffset = linspace(0, durationSeconds, numPoints);
    timeVector = startTime + seconds(secondsOffset);
else
    % For datenum input, use precise time increments in days
    % Convert dt from seconds to days
    dtInDays = dt / (24 * 60 * 60);
    
    % Create timeVector by adding precise increments
    timeVector = zeros(1, numPoints);
    timeVector(1) = startTime;
    
    for i = 2:numPoints
        timeVector(i) = timeVector(1) + (i-1) * dtInDays;
    end
end
end