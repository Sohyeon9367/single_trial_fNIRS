function [subjectFeatures, trialTable, rulesUsed] = build_subject_features_latency(matPath, rules)
% ============================================================
% build_subject_features_latency
% Unified latency feature extraction for both A2 and A5.
%
% This implementation matches the current data structure:
%   allSubjects(s).results -> subj
%   subj.Trials(t).results -> tr
% where:
%   subj.ID, subj.Group, subj.SNR
%   tr.HbO (time x channels), tr.p_HbO (channels x 1), tr.ACC, tr.duration
%
% Outputs:
%   subjectFeatures: subject-level mean latency (and #trials used)
%   trialTable: trial-level latency table (optional, useful for debugging/QC)
%   rulesUsed: rules struct actually applied
%
% NOTE:
% - This function reproduces the same filtering logic as the original A5:
%   correct-only, significant channels (p<0.05), SNR threshold, smoothing, peak latency.
% ============================================================

% ---------------- Default rules ----------------
if nargin < 2 || isempty(rules)
    rules.fs = 8.138;                 % sampling rate (Hz)
    rules.snrThreshold = 30;          % matches A5
    rules.pThreshold = 0.05;          % channel significance threshold
    rules.correctOnly = true;         % use correct trials only
    rules.minDuration = 1.2131;       % matches A5 trial duration filter
    rules.latencyWindow = [0 15];     % keep peak time within this window (s)
    rules.smoothing = 'sgolay';       % 'sgolay' or 'movmean' or 'none'
    rules.sgolayOrder = 3;            % sgolay order
    rules.maxFrame = 31;              % max frame length used in A5
    rules.movmeanK = 3;               % movmean window if too short for sgolay
end

rulesUsed = rules;

% ---------------- Load file ----------------
S = load(matPath);

assert(isfield(S, 'allSubjects'), 'MAT file must contain variable "allSubjects".');

allSubjects = S.allSubjects;
fs = rules.fs;

% ---------------- Prepare outputs ----------------
subjectFeatures = table();
trialRows = {};  % will become table later

% ---------------- Main loop (matches A5 logic) ----------------
for sIdx = 1:numel(allSubjects)
    subj = allSubjects(sIdx).results;

    % Required subject fields
    assert(isfield(subj,'ID') && isfield(subj,'Group') && isfield(subj,'SNR') && isfield(subj,'Trials'), ...
        'Subject results must contain ID, Group, SNR, Trials.');

    subjID = subj.ID;
    subjGroup = subj.Group;
    subjSNR = subj.SNR;  % 1 x Nch or Nch x 1

    subj_latencies = [];

    for tIdx = 1:numel(subj.Trials)
        tr = subj.Trials(tIdx).results;

        % Required trial fields
        if ~isfield(tr,'HbO') || ~isfield(tr,'p_HbO') || ~isfield(tr,'ACC') || ~isfield(tr,'duration')
            continue;
        end

        % Duration filter (same as A5)
        if tr.duration < rules.minDuration
            continue;
        end

        % Correct-only filter (same as A5)
        if rules.correctOnly
            % ACC in your code is a string like 'correct'
            if ~ischar(tr.ACC) && ~isstring(tr.ACC)
                continue;
            end
            if ~strcmpi(string(tr.ACC), "correct")
                continue;
            end
        end

        HbO = tr.HbO;        % [T x channels]
        pHbO = tr.p_HbO(:);  % [channels x 1]

        nCh = size(HbO, 2);

        % Significant channels (p < threshold)
        sigCh = find(pHbO < rules.pThreshold);

        for ch = sigCh(:)'
            if ch < 1 || ch > nCh
                continue;
            end

            % SNR threshold at subject-level per channel (same as A5)
            if numel(subjSNR) >= ch && subjSNR(ch) < rules.snrThreshold
                continue;
            end

            ts = HbO(:, ch);
            L = numel(ts);

            % Smoothing (same strategy as A5)
            ts_s = ts;
            if strcmpi(rules.smoothing, 'sgolay')
                F = min(rules.maxFrame, L);
                if mod(F,2) == 0, F = F - 1; end
                if F > 3
                    ts_s = sgolayfilt(ts, rules.sgolayOrder, F);
                else
                    ts_s = movmean(ts, rules.movmeanK);
                end
            elseif strcmpi(rules.smoothing, 'movmean')
                ts_s = movmean(ts, rules.movmeanK);
            end

            % Peak time (convert index to seconds)
            [~, idxPeak] = max(ts_s);
            pTime = (idxPeak - 1) / fs;

            % Latency window filter
            if pTime >= rules.latencyWindow(1) && pTime <= rules.latencyWindow(2)
                subj_latencies = [subj_latencies; pTime]; %#ok<AGROW>

                % Store trial-level row for debugging/QC
                trialRows(end+1,:) = {string(subjID), subjGroup, string(tr.ACC), tIdx, ch, pTime};
            end
        end
    end

    % Subject-level aggregate (mean latency, same as A5)
    if ~isempty(subj_latencies)
        res = table();
        res.ID = string(subjID);
        res.Group = subjGroup;
        res.Latency = mean(subj_latencies, 'omitnan');
        res.NPeaksUsed = numel(subj_latencies);
        subjectFeatures = [subjectFeatures; res]; %#ok<AGROW>
    end
end

% ---------------- Trial-level table output ----------------
if isempty(trialRows)
    trialTable = table();
else
    trialTable = cell2table(trialRows, ...
    'VariableNames', {'ID','Group','ACC','TrialIndex','Channel','PeakTime'});
end
end