function dt = extractDatetimeFromFilename(filename)
% This function attempts to extract a datetime from a filename string
% containing a timestamp. Returns a MATLAB datetime object.
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
%
% Output:
% dt - MATLAB datetime object if successful, NaN if no
% matching pattern found or conversion fails
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
% Initialize output
dt = NaT; % Not-a-Time for datetime

% Remove file extension and any directory path
[~, filename, ~] = fileparts(filename);

% Try the most common and efficient patterns first

% Pattern 1: YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS (most efficient)
tokens = regexp(filename, '(\d{8})[_-](\d{6})', 'tokens', 'once');
if ~isempty(tokens)
    try
        dt = datetime([tokens{1}, tokens{2}], 'InputFormat', 'yyyyMMddHHmmss');
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
        dt = datetime(year, month, day, hour, minute, second);
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
        dt = datetime(year, month, day, hour, minute, second);
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
            dt = datetime(year, month, day, hour, minute, second);
            return;
        catch
            continue;
        end
    end
end
end

% Works, but slow.
% function dt = extractDatetimeFromFilename(filename)
% % This function attempts to extract a datetime from a filename string
% % containing a timestamp. Returns a MATLAB datetime object.
% %
% % The function attempts to match the following datetime patterns:
% % - YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS
% % - YYYY(MM)(DD)_(HH)(MM)(SS) or YYYY(MM)(DD)-(HH)(MM)(SS)
% % - YYYY-MM-DD_HH-mm-SS or YYYY-MM-DD-HH-mm-SS
% % - YYYY-MM-DD_HH:mm:SS or YYYY-MM-DD-HH:mm:SS
% % - YYYYMMDD_HH:mm:SS or YYYYMMDD-HH:mm:SS
% % - xxx_YYYY-MM-DD_HH-mm-SS (where xxx is any prefix)
% % - xxx_YYMMDD-HHMMSS (where xxx is any prefix, and YY is a 2-digit year)
% %
% % Input:
% % filename - String containing a timestamp in one of the above formats
% %
% % Output:
% % dt - MATLAB datetime object if successful, NaN if no
% % matching pattern found or conversion fails
% %
% % Example:
% % dt = extractDatetimeFromFilename('20240115_123456.txt')
% % dt = extractDatetimeFromFilename('2024-01-15_12-34-56.csv')
% % dt = extractDatetimeFromFilename('prefix_2024-01-15_12-34-56.mat')
% % dt = extractDatetimeFromFilename('H08S1_151231-200000.wav')
% %
% % Ben Jancovich, 2024
% % Centre for Marine Science and Innovation
% % School of Biological, Earth and Environmental Sciences
% % University of New South Wales, Sydney, Australia
% %
% % Initialize output
% dt = NaT; % Not-a-Time for datetime
% 
% % Remove file extension and any directory path
% [~, filename, ~] = fileparts(filename);
% 
% % Try different datetime patterns
% patterns = {
%     '^(\d{8})[_-](\d{6})', ... % YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS
%     '^(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})', ... % YYYYMMDD_HHMMSS broken down
%     '^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})', ... % YYYYMMDD-HHMMSS
%     '^(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})', ... % YYYY-MM-DD_HH-mm-SS or YYYY-MM-DD-HH-mm-SS
%     '^(\d{4})-(\d{2})-(\d{2})[_-](\d{2}):(\d{2}):(\d{2})', ... % YYYY-MM-DD_HH:mm:SS or YYYY-MM-DD-HH:mm:SS
%     '^(\d{4})(\d{2})(\d{2})[_-](\d{2}):(\d{2}):(\d{2})', ... % YYYYMMDD_HH:mm:SS or YYYYMMDD-HH:mm:SS
%     '^(\d+)_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', ... % xxx_YYYY-MM-DD_HH-mm-SS
%     '^([A-Za-z0-9]+)_(\d{6})-(\d{6})' ... % xxx_YYMMDD-HHMMSS (new format)
%     };
% 
% % Go through patterns and set datetime according to token layout
% for i = 1:length(patterns)
%     tokens = regexp(filename, patterns{i}, 'tokens');
%     if ~isempty(tokens)
%         try
%             switch i
%                 case 1
%                     % Format: YYYYMMDD_HHMMSS
%                     date_str = tokens{1}{1};
%                     time_str = tokens{1}{2};
%                     dt = datetime([date_str, time_str], 'InputFormat', 'yyyyMMddHHmmss');
% 
%                 case {2, 3, 4, 5, 6}
%                     % Formats with year, month, day, hour, minute, second as separate components
%                     dt = datetime(str2double(tokens{1}{1}), ... % year
%                         str2double(tokens{1}{2}), ... % month
%                         str2double(tokens{1}{3}), ... % day
%                         str2double(tokens{1}{4}), ... % hour
%                         str2double(tokens{1}{5}), ... % minute
%                         str2double(tokens{1}{6})); % second
% 
%                 case 7 % Handle the prefixed format
%                     dt = datetime(str2double(tokens{1}{2}), ... % year
%                         str2double(tokens{1}{3}), ... % month
%                         str2double(tokens{1}{4}), ... % day
%                         str2double(tokens{1}{5}), ... % hour
%                         str2double(tokens{1}{6}), ... % minute
%                         str2double(tokens{1}{7})); % second
% 
%                 case 8 % Handle the new format: xxx_YYMMDD-HHMMSS
%                     date_str = tokens{1}{2};
%                     time_str = tokens{1}{3};
%                     year = str2double(['20', date_str(1:2)]); % Assuming 20xx for the century
%                     month = str2double(date_str(3:4));
%                     day = str2double(date_str(5:6));
%                     hour = str2double(time_str(1:2));
%                     minute = str2double(time_str(3:4));
%                     second = str2double(time_str(5:6));
%                     dt = datetime(year, month, day, hour, minute, second);
%             end
%             break
%         catch
%             continue
%         end
%     end
% end
% end
