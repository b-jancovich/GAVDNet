function [adjStats, FP_becomes_TP, FN_becomes_TN] = reclassifyDisagreementsByLogic(disagreements, logicName)
% reclassifyDisagreementsByLogic
% Reclassifies false positives and false negatives based on analyst decisions
% and the specified decision logic.
%
% Inputs:
%   disagreements - Struct containing falsePositives and falseNegatives arrays
%                   Each must have 'analystDecision' field with one of:
%                   'DiscreteCallsPresent', 'ChorusPresent', 
%                   'DiscreteCallsChorusPresent', 'CallChorusAbsent'
%   logicName     - String specifying decision logic:
%                   'Inclusive': All vocal activity = TP
%                   'Discrete-only': Discrete calls (with/without chorus) = TP
%                   'Strict-discrete': Only pure discrete calls = TP
%                   'Chorus-aware': Same as Discrete-only, tracks chorus
%
% Outputs:
%   adjStats      - Struct containing adjudication statistics breakdown
%   FP_becomes_TP - Logical array indicating which FPs become TPs
%   FN_becomes_TN - Logical array indicating which FNs become TNs
%
% Ben Jancovich, 2025
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

FP = disagreements.falsePositives;
FN = disagreements.falseNegatives;

nFP = length(FP);
nFN = length(FN);

% Initialize output arrays
FP_becomes_TP = false(nFP, 1);
FN_becomes_TN = false(nFN, 1);

% Initialize statistics counters
adjStats = struct();
adjStats.logicName = logicName;
adjStats.nDiscreteCallsPresent = 0;
adjStats.nChorusPresent = 0;
adjStats.nDiscreteCallsChorusPresent = 0;
adjStats.nCallChorusAbsent = 0;
adjStats.nFP_to_TP = 0;
adjStats.nFP_remain_FP = 0;
adjStats.nFN_to_TN = 0;
adjStats.nFN_remain_FN = 0;

%% Process False Positives

fprintf('Reclassifying False Positives (%s logic)...\n', logicName);

for i = 1:nFP
    decision = FP(i).analystDecision;
    
    % Count decision types
    switch decision
        case 'DiscreteCallsPresent'
            adjStats.nDiscreteCallsPresent = adjStats.nDiscreteCallsPresent + 1;
        case 'ChorusPresent'
            adjStats.nChorusPresent = adjStats.nChorusPresent + 1;
        case 'DiscreteCallsChorusPresent'
            adjStats.nDiscreteCallsChorusPresent = adjStats.nDiscreteCallsChorusPresent + 1;
        case 'CallChorusAbsent'
            adjStats.nCallChorusAbsent = adjStats.nCallChorusAbsent + 1;
        otherwise
            warning('Unexpected analyst decision for FP %d: %s', i, decision);
    end
    
    % Apply logic to determine if FP becomes TP
    switch logicName
        case 'Inclusive'
            % All vocal activity = TP
            if strcmp(decision, 'DiscreteCallsPresent') || ...
               strcmp(decision, 'ChorusPresent') || ...
               strcmp(decision, 'DiscreteCallsChorusPresent')
                FP_becomes_TP(i) = true;
            end
            
        case 'Discrete-only'
            % Discrete calls (with or without chorus) = TP
            if strcmp(decision, 'DiscreteCallsPresent') || ...
               strcmp(decision, 'DiscreteCallsChorusPresent')
                FP_becomes_TP(i) = true;
            end
            
        case 'Strict-discrete'
            % Only pure discrete calls = TP
            if strcmp(decision, 'DiscreteCallsPresent')
                FP_becomes_TP(i) = true;
            end
            
        case 'Chorus-aware'
            % Same as Discrete-only
            if strcmp(decision, 'DiscreteCallsPresent') || ...
               strcmp(decision, 'DiscreteCallsChorusPresent')
                FP_becomes_TP(i) = true;
            end
            
        otherwise
            error('Unknown logic name: %s', logicName);
    end
end

adjStats.nFP_to_TP = sum(FP_becomes_TP);
adjStats.nFP_remain_FP = nFP - adjStats.nFP_to_TP;

fprintf('  %d FPs remain FPs\n', adjStats.nFP_remain_FP);
fprintf('  %d FPs become TPs\n', adjStats.nFP_to_TP);

%% Process False Negatives

fprintf('Reclassifying False Negatives (%s logic)...\n', logicName);

for i = 1:nFN
    decision = FN(i).analystDecision;
    
    % FN becomes TN if analyst confirms no call present
    if strcmp(decision, 'CallChorusAbsent')
        FN_becomes_TN(i) = true;
    end
    
    % Note: Other decisions (call present) mean FN remains FN
    % The detector missed a call that truly was there
end

adjStats.nFN_to_TN = sum(FN_becomes_TN);
adjStats.nFN_remain_FN = nFN - adjStats.nFN_to_TN;

fprintf('  %d FNs remain FNs\n', adjStats.nFN_remain_FN);
fprintf('  %d FNs become TNs\n', adjStats.nFN_to_TN);

%% Additional stats for Chorus-aware logic

if strcmp(logicName, 'Chorus-aware')
    % Track how many TPs contain chorus
    adjStats.nTP_withChorus = 0;
    adjStats.nTP_withoutChorus = 0;
    
    for i = 1:nFP
        if FP_becomes_TP(i)
            decision = FP(i).analystDecision;
            if strcmp(decision, 'DiscreteCallsChorusPresent')
                adjStats.nTP_withChorus = adjStats.nTP_withChorus + 1;
            elseif strcmp(decision, 'DiscreteCallsPresent')
                adjStats.nTP_withoutChorus = adjStats.nTP_withoutChorus + 1;
            end
        end
    end
    
    if adjStats.nFP_to_TP > 0
        adjStats.chorusPrevalence = adjStats.nTP_withChorus / adjStats.nFP_to_TP;
        fprintf('  Chorus prevalence in new TPs: %.1f%% (%d/%d)\n', ...
            adjStats.chorusPrevalence * 100, adjStats.nTP_withChorus, adjStats.nFP_to_TP);
    end
end

fprintf('\n');

end