function filename = findClosestWav(filelist, target_datestring)
    % FINDCLOSESTWAV Find the closest WAV file to a target date and time
    %
    % Inputs:
    %   FILELIST - A struct array containing information about WAV files,
    %              obtained from the DIR function
    %   target_datestring - A string representing the target date and time in
    %                       the format 'yymmdd-HHMMSS'.
    %
    % Output:
    %   FILENAME - A string containing the name of the WAV file closest to,
    %              but not exceeding, the target date and time
    %
    % This function searches through a list of WAV files and finds the one
    % with a name containing a timestamp closest to, but not exceeding, 
    % the specified target date and time. 
    % 
    % The function assumes that the wav filenames follow the CTBTO IMS
    % naming convention "H08S1_011014-080000.wav" The characters before the 
    % underscore are the station code, which is not used, and the 
    % characters after it are the file's start date-time stamp in the 
    % format 'yymmdd-HHMMSS'.
    %
    % If no suitable file is found (i.e., all files are after the target
    % date or no files have a valid timestamp), an empty string is returned.
    %
    % Ben Jancovich, 2024
    % Centre for Marine Science and Innovation
    % School of Biological, Earth and Environmental Sciences
    % University of New South Wales, Sydney, Australia
    %
    
    % Convert the target date-time stamp to serial_datenum    
    target_serial_datenum = datenum(target_datestring, 'yymmdd-HHMMSS');

    % Initialize the closest serial_datenum and filename
    closest_serial_datenum = -inf;  % Changed from inf to -inf
    filename = '';
    
    % Iterate over the files
    for i = 1:length(filelist)
        % Extract the date-time stamp from the file name
        file_datestring = regexp(filelist(i).name, '\d+', 'match');
        
        % If there's no date-time stamp, skip this file
        if isempty(file_datestring)
            continue;
        end
        
        % Convert the date-time stamp to serial_datenum
        file_serial_datenum = datenum([file_datestring{1,3}, '-', file_datestring{1,4}], 'yymmdd-HHMMSS');

        % If the file's datetime is after the target, skip this file
        if file_serial_datenum > target_serial_datenum
            continue;
        end
        
        % If the file's datetime is closer to the target than the current closest, update the closest
        if file_serial_datenum > closest_serial_datenum  % Changed comparison logic
            closest_serial_datenum = file_serial_datenum;
            filename = filelist(i).name;
        end
    end
end