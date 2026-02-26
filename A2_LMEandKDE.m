%% ============================================================
% A2_LMEandKDE (UPDATED)
% - Use build_subject_features_latency.m (shared with A5) to generate trialTable (PeakTime).
% - Do NOT modify build_subject_features_latency.m.
% - Compute robustness timing metrics (Onset10 / HalfMax / COM) inside A2 only.
% - Fit LME repeatedly across timing definitions to demonstrate robustness.
%% ============================================================

clear; clc;

%% ----------------- 1) Paths & settings -----------------
dataPath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed\Stroop_AllSubjects_NIRS_HRF_TTest_05.mat';
infoPath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Raw data\subject_info.xlsx';
savePath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed_Latency_Master.mat';

fs = 8.138;
snrThreshold = 30;
maxWindow = 20;                         % seconds, for latency window end
groupNames = {'HC','MCI','AD'};

%% ----------------- 2) Build trialTable via shared function -----------------

fprintf('✨ Preprocessing starts (via build_subject_features_latency)...\n');

% IMPORTANT: rules are assigned here (A2-only); build_subject_features_latency.m is not modified.
rules = struct();
rules.fs = fs;
rules.snrThreshold = snrThreshold;
rules.pThreshold = 0.05;
rules.correctOnly = false;              % A2 needs both correct and incorrect
rules.minDuration = 1.2131;
rules.latencyWindow = [0.5 maxWindow];  % match original A2 definition
rules.smoothing = 'sgolay';
rules.sgolayOrder = 3;
rules.maxFrame = 31;
rules.movmeanK = 3;

% Shared function (also used in A5) produces PeakTime and trial identifiers
[~, trialTable, rulesUsed] = build_subject_features_latency(dataPath, rules);

% ---- Merge covariates (Age/Education) ----
s_info = readtable(infoPath);

% Robust ID handling (trim spaces, unify type)
trialTable.ID = strtrim(string(trialTable.ID));
s_info.SubjectID = strtrim(string(s_info.SubjectID));

cov = s_info(:, {'SubjectID','Age','Education'});
cov.Properties.VariableNames{'SubjectID'} = 'ID';   % rename first

trialTable = outerjoin(trialTable, cov, ...
    'Keys','ID', ...
    'Type','left', ...
    'MergeKeys', true);

% ---- Standardize variable types ----
% Group is assumed to be numeric (1/2/3). Map to HC/MCI/AD.
trialTable.Group = categorical(trialTable.Group, [1 2 3], groupNames);

% ACC should be 'correct'/'incorrect' (case-insensitive)
trialTable.ACC = categorical(lower(string(trialTable.ACC)), {'correct','incorrect'});

% ID as categorical for LME random effect
trialTable.ID = categorical(trialTable.ID);

% ---- Add robustness timing metrics in A2 only ----
fprintf('🔧 Computing robustness metrics (Onset10 / HalfMax / COM) inside A2...\n');
[trialTable, qc] = add_latency_robustness_metrics_A2(trialTable, dataPath, rules);

% ---- Save for reuse ----
save(savePath, 'trialTable', 'qc', 'rulesUsed');
fprintf('✅ Saved: %s\n', savePath);

%% ----------------- 3) Quick QC to prevent "no observations" errors -----------------
fprintf('\n[Covariate QC]\n');
fprintf('Rows total: %d\n', height(trialTable));
fprintf('Rows missing Age/Education: %d\n', sum(any(ismissing(trialTable(:,{'Age','Education'})),2)));
fprintf('Unique subjects total: %d\n', numel(categories(trialTable.ID)));
fprintf('Unique subjects with covariates: %d\n', ...
    numel(unique(trialTable.ID(~any(ismissing(trialTable(:,{'Age','Education'})),2)))));

fprintf('\n[Robustness QC]\n');
if exist('qc','var')
    disp(qc);
end
%% ----------------- 4. KDE Plot (Shared Legend, Updated) -----------------

% Select which timing metric to visualize
plotMetric = 'PeakTime';   % options: 'PeakTime','Onset10','HalfMax','COM'

groupNames = {'HC','MCI','AD'};
colors = [0, 0.447, 0.741; 
          0.85, 0.325, 0.098; 
          0.929, 0.694, 0.125];

figure('Color','w','Position', [100 100 1200 500]);
tlo = tiledlayout(1, 2, 'Padding', 'loose', 'TileSpacing', 'compact');

titles = {['Correct Trials: ' plotMetric], ...
          ['Incorrect Trials: ' plotMetric]};
acc_types = {'correct', 'incorrect'};

p_h = [];

for i = 1:2
    ax = nexttile(tlo); hold on;

    for g = 1:3
        % Extract data from trialTable (instead of masterData)
        data = trialTable.(plotMetric)( ...
            trialTable.Group == groupNames{g} & ...
            trialTable.ACC == acc_types{i});

        if isempty(data), continue; end

        % Kernel density estimation
        [f, xi] = ksdensity(data, ...
            'Support', [0, maxWindow+0.5], ...
            'Boundary', 'reflection');

        % Fill area
        fill(xi, f, colors(g,:), ...
            'FaceAlpha', 0.1, ...
            'EdgeColor', 'none', ...
            'HandleVisibility', 'off');

        % Density line
        line_h = plot(xi, f, ...
            'Color', colors(g,:), ...
            'LineWidth', 2.5, ...
            'DisplayName', groupNames{g});

        if i == 1
            p_h = [p_h, line_h];
        end
    end

    title(titles{i}, 'FontSize', 15);

    if strcmp(plotMetric,'PeakTime')
        xlabel('Time to Peak (s)');
    elseif strcmp(plotMetric,'Onset10')
        xlabel('Onset 10% (s)');
    elseif strcmp(plotMetric,'HalfMax')
        xlabel('Half Maximum (s)');
    else
        xlabel('Center of Mass Time (s)');
    end

    ylabel('Density');
    xlim([0 12]);
    ylim([0 0.8]);
    grid on;
    ax.GridAlpha = 0.1;
end

lgd = legend(p_h, groupNames);
lgd.Layout.Tile = 'north';
lgd.Orientation = "horizontal";
lgd.Box = 'off';
%% ----------------- 4) LME robustness: repeat model for multiple timing definitions -----------------
% This is the core "robustness" evidence for reviewers:
% If Group and/or Group×ACC effects persist across alternative latency definitions, credibility increases.

metrics = {'PeakTime','Onset10','HalfMax','COM'};

% Remove missing covariates once (avoid "no usable observations")
baseT = rmmissing(trialTable, 'DataVariables', {'Age','Education'});

robustTbl = table();
ri = 1;

for m = 1:numel(metrics)
    y = metrics{m};

    T = baseT;
    T = rmmissing(T, 'DataVariables', {y}); % metric-specific NaNs

    fprintf('\n--- LME: %s ~ Group * ACC + Age + Education + (1|ID) ---\n', y);
    fprintf('Rows used: %d | Subjects used: %d\n', height(T), numel(unique(T.ID)));

    form = sprintf('%s ~ Group * ACC + Age + Education + (1|ID)', y);
    lme = fitlme(T, form);
    disp(lme);

    % Type III ANOVA (fixed effects significance)
    a = dataset2table(anova(lme)); % columns: Term, DF, FStat, pValue, ...

    % Extract key p-values (defensive coding in case term naming differs)
    pGroup = NaN; pACC = NaN; pInt = NaN;
    if any(strcmp(a.Term,'Group')),      pGroup = a.pValue(strcmp(a.Term,'Group')); end
    if any(strcmp(a.Term,'ACC')),        pACC   = a.pValue(strcmp(a.Term,'ACC')); end
    if any(strcmp(a.Term,'Group:ACC')),  pInt   = a.pValue(strcmp(a.Term,'Group:ACC')); end

    robustTbl.Metric(ri,1) = string(y);
    robustTbl.RowsUsed(ri,1) = height(T);
    robustTbl.SubjectsUsed(ri,1) = numel(unique(T.ID));
    robustTbl.p_Group(ri,1) = pGroup;
    robustTbl.p_ACC(ri,1) = pACC;
    robustTbl.p_GroupXACC(ri,1) = pInt;

    ri = ri + 1;
end

fprintf('\n================== Robustness Summary (Type III p-values) ==================\n');
disp(robustTbl);


%% ----------------- Descriptive latency summaries (robust across MATLAB versions) -----------------
% Computes observed (non-model) Mean / SD / CV / N by Group × ACC.
% Handles MATLAB version differences in groupsummary() output variable names.

metricForText = 'PeakTime';  % or 'Onset10','HalfMax','COM'

Tdesc = rmmissing(trialTable, 'DataVariables', {metricForText});

% Trial-level summary
statsGA = groupsummary(Tdesc, {'Group','ACC'}, {'mean','std'}, metricForText);

% 1) Rename mean/std columns robustly
vars = statsGA.Properties.VariableNames;

meanVar = vars(startsWith(vars, 'mean_'));
stdVar  = vars(startsWith(vars,  'std_'));

if ~isempty(meanVar)
    idx = find(strcmp(vars, meanVar{1}), 1);
    statsGA.Properties.VariableNames{idx} = 'Mean';
end
vars = statsGA.Properties.VariableNames; % refresh

if ~isempty(stdVar)
    idx = find(strcmp(vars, stdVar{1}), 1);
    statsGA.Properties.VariableNames{idx} = 'SD';
end
vars = statsGA.Properties.VariableNames; % refresh

% 2) Create N robustly (MATLAB often provides GroupCount)
if ismember('GroupCount', vars)
    statsGA.N = statsGA.GroupCount;
else
    % Fallback: compute N manually if GroupCount is not present
    tmpN = groupsummary(Tdesc, {'Group','ACC'}, 'numel', metricForText);
    varsN = tmpN.Properties.VariableNames;
    nVar = varsN(startsWith(varsN, 'numel_'));
    if isempty(nVar)
        error('Could not determine N: neither GroupCount nor numel_* found.');
    end
    statsGA.N = tmpN.(nVar{1});
end

% 3) Compute CV
if ~all(ismember({'Mean','SD'}, statsGA.Properties.VariableNames))
    error('Mean/SD columns not found after renaming.');
end
statsGA.CV = statsGA.SD ./ statsGA.Mean;

% 4) Sort for readability
statsGA = sortrows(statsGA, {'ACC','Group'});

fprintf('\n================== Descriptive (Observed) %s by Group × ACC ==================\n', metricForText);
disp(statsGA(:, {'Group','ACC','Mean','CV','N'}));

%% ----------------- Selection-bias check: exclusion due to short trial duration -----------------
% This section quantifies how many trials were excluded because duration < minDuration,
% reported overall and by diagnostic group (HC/MCI/AD).
% It does NOT require modifying build_subject_features_latency.m.

S = load(dataPath);
assert(isfield(S,'allSubjects'), 'MAT file must contain allSubjects');
allSubjects = S.allSubjects;

minDur = rulesUsed.minDuration;  % should be 1.2131
grpLevels = {'HC','MCI','AD'};

% Counters per group
count_totalTrials = zeros(3,1);      % all Stroop trials encountered (regardless of correctness)
count_shortTrials = zeros(3,1);      % duration < minDur
count_longTrials  = zeros(3,1);      % duration >= minDur

% Optional: check whether short trials are systematically "faster" in behavior (if RT exists)
hasRT = false;
shortRT = cell(3,1);
longRT  = cell(3,1);

for sIdx = 1:numel(allSubjects)
    subj = allSubjects(sIdx).results;
    gNum = subj.Group;  % expected 1/2/3
    if ~(gNum>=1 && gNum<=3), continue; end

    for tIdx = 1:numel(subj.Trials)
        tr = subj.Trials(tIdx).results;

        if ~isfield(tr,'duration') || isempty(tr.duration) || ~isfinite(tr.duration)
            continue;
        end

        count_totalTrials(gNum) = count_totalTrials(gNum) + 1;

        isShort = tr.duration < minDur;
        if isShort
            count_shortTrials(gNum) = count_shortTrials(gNum) + 1;
        else
            count_longTrials(gNum) = count_longTrials(gNum) + 1;
        end

        % If behavioral RT is available, collect it to support the "fast trials" argument
        % Common field names might be 'RT', 'reactionTime', or similar.
        rt = NaN;
        if isfield(tr,'RT'), rt = tr.RT; end
        if ~isfinite(rt) && isfield(tr,'reactionTime'), rt = tr.reactionTime; end

        if isfinite(rt)
            hasRT = true;
            if isShort
                shortRT{gNum}(end+1,1) = rt;
            else
                longRT{gNum}(end+1,1) = rt;
            end
        end
    end
end

% Summarize exclusion rates
exclRate = 100 * (count_shortTrials ./ max(count_totalTrials,1));

biasTbl = table(grpLevels(:), count_totalTrials, count_shortTrials, count_longTrials, exclRate, ...
    'VariableNames', {'Group','TotalTrials','ShortTrials_ltMinDur','KeptTrials_geMinDur','ExcludedPercent'});

fprintf('\n================== Exclusion due to short trial duration (< %.2fs) ==================\n', minDur);
disp(biasTbl);

% Optional: if RT exists, compare RT distributions for short vs long trials by group
if hasRT
    fprintf('\n[Optional RT check] Comparing RT for short vs long trials (if RT available)\n');
    for g = 1:3
        if ~isempty(shortRT{g}) && ~isempty(longRT{g})
            fprintf('%s: RT(short) mean=%.3f, RT(long) mean=%.3f (n_short=%d, n_long=%d)\n', ...
                grpLevels{g}, mean(shortRT{g},'omitnan'), mean(longRT{g},'omitnan'), numel(shortRT{g}), numel(longRT{g}));
        end
    end
else
    fprintf('\n[Note] No RT field found in trials; duration-based exclusion is quantified but "fast trials" cannot be directly validated.\n');
end
%% ============================================================
% Local function(s): A2-only (do not touch build_subject_features_latency.m)
%% ============================================================
function [trialTable, qc] = add_latency_robustness_metrics_A2(trialTable, dataPath, rules)
% Add Onset10 / HalfMax / COM to trialTable, matching rows by (ID, TrialIndex, Channel).
% PeakTime is kept as provided by build_subject_features_latency.m.
%
% Definitions (within rules.latencyWindow):
% - Onset10: first time reaching 10% of (peak - baseline)
% - HalfMax: first time reaching 50% of (peak - baseline)
% - COM: center-of-mass time of positive signal above baseline

S = load(dataPath);
assert(isfield(S,'allSubjects'), 'MAT file must contain allSubjects');
allSubjects = S.allSubjects;

fs = rules.fs;

% Initialize columns
trialTable.Onset10 = NaN(height(trialTable),1);
trialTable.HalfMax = NaN(height(trialTable),1);
trialTable.COM     = NaN(height(trialTable),1);

% Build a fast lookup from key -> row index
% key format: "ID|TrialIndex|Channel"
keys = strings(height(trialTable),1);
for i = 1:height(trialTable)
    keys(i) = string(trialTable.ID(i)) + "|" + string(trialTable.TrialIndex(i)) + "|" + string(trialTable.Channel(i));
end
key2row = containers.Map(keys, num2cell(1:height(trialTable)));

% QC counters
qc = struct();
qc.robust_missing_key = 0;
qc.robust_amp_nonpos  = 0;
qc.robust_rows_filled = 0;

winStartS = rules.latencyWindow(1);
winEndS   = rules.latencyWindow(2);

for sIdx = 1:numel(allSubjects)
    subj = allSubjects(sIdx).results;
    subjID = strtrim(string(subj.ID));
    subjSNR = subj.SNR(:);

    for tIdx = 1:numel(subj.Trials)
        tr = subj.Trials(tIdx).results;

        if ~isfield(tr,'HbO') || ~isfield(tr,'p_HbO') || ~isfield(tr,'duration')
            continue;
        end
        if tr.duration < rules.minDuration
            continue;
        end

        HbO  = tr.HbO;          % [T x nCh]
        pHbO = tr.p_HbO(:);     % [nCh x 1]
        nCh  = size(HbO,2);
        L    = size(HbO,1);

        % Window indices
        i0 = max(1, round(winStartS*fs) + 1);
        i1 = min(L, round(winEndS*fs) + 1);
        if i0 >= i1, continue; end
        winIdx = i0:i1;

        % Baseline window: [0, winStartS)
        baseIdx = 1:max(1, i0-1);

        sigCh = find(pHbO < rules.pThreshold);

        for ch = sigCh(:)'
            if ch < 1 || ch > nCh, continue; end

            % SNR filter (same as main extraction)
            if numel(subjSNR) >= ch && subjSNR(ch) < rules.snrThreshold
                continue;
            end

            key = subjID + "|" + string(tIdx) + "|" + string(ch);
            if ~isKey(key2row, key)
                qc.robust_missing_key = qc.robust_missing_key + 1;
                continue;
            end
            row = key2row(key);

            ts = HbO(:,ch);

            % Apply the same smoothing rule as extraction
            ts_s = ts;
            if isfield(rules,'smoothing') && strcmpi(rules.smoothing,'sgolay')
                F = min(rules.maxFrame, L);
                if mod(F,2)==0, F = F-1; end
                if F > 3
                    ts_s = sgolayfilt(ts, rules.sgolayOrder, F);
                else
                    ts_s = movmean(ts, rules.movmeanK);
                end
            elseif isfield(rules,'smoothing') && strcmpi(rules.smoothing,'movmean')
                ts_s = movmean(ts, rules.movmeanK);
            end

            % Use PeakTime already in trialTable
            peakTime = trialTable.PeakTime(row);
            if ~isfinite(peakTime), continue; end
            idxPeak = round(peakTime*fs) + 1;
            if idxPeak < 1 || idxPeak > L, continue; end

            baseVal = mean(ts_s(baseIdx), 'omitnan');
            peakVal = ts_s(idxPeak);
            amp = peakVal - baseVal;

            % If amplitude is not positive, onset/half/COM are not meaningful
            if ~(isfinite(amp) && amp > 0)
                qc.robust_amp_nonpos = qc.robust_amp_nonpos + 1;
                continue;
            end

            % Onset10 (10% rise)
            thr10 = baseVal + 0.10 * amp;
            idx10 = find(ts_s(winIdx) >= thr10, 1, 'first');
            if ~isempty(idx10)
                trialTable.Onset10(row) = (winIdx(idx10)-1)/fs;
            end

            % HalfMax (50% rise)
            thr50 = baseVal + 0.50 * amp;
            idx50 = find(ts_s(winIdx) >= thr50, 1, 'first');
            if ~isempty(idx50)
                trialTable.HalfMax(row) = (winIdx(idx50)-1)/fs;
            end

            % Center-of-mass time (positive area above baseline)
            w = ts_s(winIdx) - baseVal;
            w(w < 0) = 0;
            if sum(w) > 0
                tsec = (winIdx-1)/fs;
                trialTable.COM(row) = sum(tsec(:).*w(:)) / sum(w(:));
            end

            qc.robust_rows_filled = qc.robust_rows_filled + 1;
        end
    end
end

fprintf('✅ Robustness metrics added: rows_filled=%d, missing_key=%d, amp_nonpos=%d\n', ...
    qc.robust_rows_filled, qc.robust_missing_key, qc.robust_amp_nonpos);
end