% Known parameters (Chagos)
startFreq_Chagos = 32.97;  % Hz
decline_rate_Chagos = 0.33;  % Hz/year
reference_year_Chagos = 2017;

% Known parameters (Z Call)
startFreq_BmAntZ = 28.5;  % Hz
decline_rate_BmAntZ = 0.135;  % Hz/year
reference_year_BmAntZ = 1995;

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
