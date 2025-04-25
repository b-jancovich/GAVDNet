function dt = extractDatetimeFromFilename(varargin)
% This function attempts to extract a datetime from a filename string
% containing a timestamp. Returns a MATLAB datetime object by default, or 
% can return datenum (double) if requested.
%
% The function attempts to match the following datetime patterns:
% - YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS
% - YYYY(MM)(DD)_(HH)(MM)(SS) or YYYY(MM)(DD)-(HH)(MM)(SS)
% - YYYY-MM-DD_HH-mm-SS or YYYY-MM-DD-HH-mm-SS
% - YYYY-MM-DD_HH:mm:SS or YYYY-MM-DD-HH:mm:SS
% - YYYYMMDD_HH:mm:SS or YYYYMMDD-HH:mm:SS
% - xxx_YYYY-MM-DD_HH-mm-SS (where xxx is any prefix)
% - xxx_YYMMDD-HHMMSS (where xxx is any prefix, and YY is a 2-digit year)
%
% Input:
% filename - String containing a timestamp in one of the above formats
% outFormat -   "datetime" returns dt as matlab datetime object.
%               "datenum" returns dt as matlab serial date number 
%               representing the number of whole and fractional days since 
%               January 0, 0000 in the proleptic ISO calendar. 
%               (optional, default: "datetime")
%
% Output:
% dt - MATLAB datetime object if successful, NaN if no
% matching pattern found or conversion fails, datenum if requested.
%
% Example:
% dt = extractDatetimeFromFilename('20240115_123456.txt')
% dt = extractDatetimeFromFilename('2024-01-15_12-34-56.csv')
% dt = extractDatetimeFromFilename('prefix_2024-01-15_12-34-56.mat')
% dt = extractDatetimeFromFilename('H08S1_151231-200000.wav')
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
if nargin ==  1
    filename = varargin{1};
elseif nargin == 2
    filename = varargin{1};
    outFormat = varargin{2};
else
    error('Unexpected number of input arguments.')
end

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

