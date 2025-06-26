% Known parameters (Chagos, from [1])
startFreq_Chagos = 32.97;  % Hz 
decline_rate_Chagos = 0.33;  % Hz/year
reference_year_Chagos = 2017;

% Known parameters (Z Call, from [2] & [3])
decline_rate_BmAntZ = 0.135; % Hz/year [2]
startFreq_BmAntZ = 28.5; % Hz [3]
reference_year_BmAntZ = 1995; % [3]

%   [1] Leroy, Emmanuelle C., Jean-Yves Royer, Abigail Alling, Ben Maslen, 
%       and Tracey L. Rogers. “Multiple Pygmy Blue Whale Acoustic 
%       Populations in the Indian Ocean: Whale Song Identifies a Possible 
%       New Population.” Scientific Reports 11, no. 1 (December 2021): 8762. 
%       https://doi.org/10.1038/s41598-021-88062-5.
%
%   [2] Gavrilov AN, McCauley RD, Gedamke J.Steady inter and intra-annual
%       decrease in the vocalization frequency of Antarctic blue whales. 
%       The Journal of the Acoustical Society of America.2012; 131 (6):4476–4480. 
%       doi: 10.1121/1.4707425 PMID: 22712920
%
%   [3] McDonald, Mark A., Ja Hildebrand, and S Mesnick. “Worldwide Decline 
%       in Tonal Frequencies of Blue Whale Songs.” Endangered Species 
%       Research 9 (October 23, 2009): 13–21. 
%       https://doi.org/10.3354/esr00217.

% Years to estimate
years = 2000:2025;

% Calculate frequencies using linear relationship
frequencies_Chagos = startFreq_Chagos - decline_rate_Chagos * (years - reference_year_Chagos);
frequencies_BmAntZ = startFreq_BmAntZ - decline_rate_BmAntZ * (years - reference_year_BmAntZ);

% Create table
freq_table = table(years', frequencies_Chagos', frequencies_BmAntZ', ...
    'VariableNames', {'Year', 'Chagos Song, unit 1, subunit 1',...
    'Bm Ant. Z-Call, Unit-A'});

writetable(freq_table, 'frequencyShiftTable.xlsx')
