%% ============================================================
% A6_run_loso_full (UPDATED FULL VERSION)
% - Uses build_subject_features_latency.m for latency feature extraction
% - Keeps A6 self-contained (no modification to build_subject_features_latency.m)
% - Fixes permutation test for AUC (shuffle labels only)
% - Extracts subject-level features:
%     Latency = mean PeakTime from correct trials (from trialTable)
%     BetaDiff = mean(beta_HbO(correct)) - mean(beta_HbO(incorrect))
%       using the SAME (TrialIndex, Channel) pairs present in trialTable
% - Runs LOSO logistic regression for pairwise group classification
%% ============================================================

clear; clc;

%% ----------------- 0) Paths -----------------
dataPath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed\Stroop_AllSubjects_NIRS_HRF_TTest_05.mat';
load(dataPath);  % expects allSubjects

%% ----------------- 1) Parameters -----------------
fs = 8.138;
snrThreshold = 30;
minDuration  = 1.2131;
latWindow    = [0.5 15];   % seconds

groupNames = {'HC','MCI','AD'};  % mapping Group=1/2/3 in your MAT
P_perm = 5000;                  % permutation iterations (adjust as needed)
rng(2);

%% ----------------- 2) Build trialTable via build_subject_features_latency -----------------
fprintf('Building trialTable using build_subject_features_latency()...\n');

rules = struct();
rules.fs = fs;
rules.snrThreshold = snrThreshold;
rules.pThreshold = 0.05;
rules.correctOnly = false;          % we need both correct/incorrect for BetaDiff
rules.minDuration = minDuration;
rules.latencyWindow = latWindow;
rules.smoothing = 'sgolay';
rules.sgolayOrder = 3;
rules.maxFrame = 31;
rules.movmeanK = 3;

% trialTable contains: ID, Group, ACC, TrialIndex, Channel, PeakTime (at minimum)
[~, trialTable, rulesUsed] = build_subject_features_latency(dataPath, rules);

% Standardize formats
trialTable.ID  = strtrim(string(trialTable.ID));
trialTable.ACC = lower(strtrim(string(trialTable.ACC)));

fprintf('trialTable rows: %d | unique IDs: %d\n', height(trialTable), numel(unique(trialTable.ID)));

%% ----------------- 3) Subject-level feature extraction (Latency + BetaDiff) -----------------
fprintf('Extracting subject-level features...\n');

% Map subject ID -> index in allSubjects
id2idx = containers.Map('KeyType','char','ValueType','double');
for s = 1:numel(allSubjects)
    sid = strtrim(string(allSubjects(s).results.ID));
    id2idx(char(sid)) = s;
end

uIDs = unique(trialTable.ID);
subjectData = table();

% Optional: quantify SNR-based rejection among p<0.05 channels (for reporting)
rejection_ratios = [];

for ui = 1:numel(uIDs)
    sid = uIDs(ui);
    if ~isKey(id2idx, char(sid))
        continue;
    end
    sIdx = id2idx(char(sid));
    subj = allSubjects(sIdx).results;

    gNum = subj.Group;
    if ~(gNum >= 1 && gNum <= 3)
        continue;
    end

    % Subject-specific trialTable subset
    TT = trialTable(trialTable.ID == sid, :);
    if isempty(TT), continue; end

    % Latency feature: mean PeakTime from correct trials only (as in original A6 logic)
    lat_corr = TT.PeakTime(TT.ACC == "correct");
    if isempty(lat_corr)
        continue;
    end

    trials = subj.Trials;
    rawSNR = subj.SNR(:);

    % Use first 48 channels if available (consistent with typical fNIRS montage)
    if numel(rawSNR) >= 48
        subj_ch_snr = rawSNR(1:48);
    else
        subj_ch_snr = rawSNR;
    end

    % Compute SNR rejection ratio among significant channels (p<0.05) across trials
    total_sig = 0;
    rejected_sig = 0;
    for t = 1:numel(trials)
        tr = trials(t).results;
        if ~isfield(tr,'duration') || ~isfinite(tr.duration) || tr.duration < minDuration
            continue;
        end
        if ~isfield(tr,'p_HbO') || isempty(tr.p_HbO)
            continue;
        end
        sigCh = find(tr.p_HbO(:) < rules.pThreshold);
        if isempty(sigCh), continue; end
        for ch = sigCh(:)'
            total_sig = total_sig + 1;
            if numel(subj_ch_snr) >= ch && subj_ch_snr(ch) < snrThreshold
                rejected_sig = rejected_sig + 1;
            end
        end
    end
    if total_sig > 0
        rejection_ratios(end+1,1) = 100 * (rejected_sig / total_sig); 
    end

    % Beta features: collect beta_HbO for the SAME (TrialIndex, Channel) pairs in TT
    betas_corr = [];
    betas_incor = [];

    for r = 1:height(TT)
        tIdx = TT.TrialIndex(r);
        ch   = TT.Channel(r);
        acc  = TT.ACC(r);

        if tIdx < 1 || tIdx > numel(trials), continue; end
        tr = trials(tIdx).results;

        if ~isfield(tr,'beta_HbO') || isempty(tr.beta_HbO) || numel(tr.beta_HbO) < ch
            continue;
        end

        b = tr.beta_HbO(ch);
        if ~isfinite(b), continue; end

        if acc == "correct"
            betas_corr(end+1,1) = b; 
        elseif acc == "incorrect"
            betas_incor(end+1,1) = b; 
        end
    end

    % Match original A6 behavior: require incorrect betas to compute BetaDiff
    if isempty(betas_incor)
        continue;
    end

    res = table();
    res.SubjectID = {char(sid)};
    res.Group = gNum;
    res.Latency = mean(lat_corr, 'omitnan');
    res.BetaDiff = mean(betas_corr, 'omitnan') - mean(betas_incor, 'omitnan');

    subjectData = [subjectData; res]; %#ok<AGROW>
end

fprintf('Subjects retained for classification: %d\n', height(subjectData));
fprintf('[Group counts in subjectData]\n');
for gg = 1:3
    fprintf('  %s: n=%d\n', groupNames{gg}, sum(subjectData.Group == gg));
end

if ~isempty(rejection_ratios)
    fprintf('Mean rejection ratio (sig channels rejected by SNR): %.2f%%\n', mean(rejection_ratios, 'omitnan'));
end

%% ----------------- 4) Define comparisons -----------------
comparisons = {
    1, 3, 'HC vs AD';
    1, 2, 'HC vs MCI';
    2, 3, 'MCI vs AD'
};

featureSets = {
    {'Latency'}, 'Latency';
    {'BetaDiff'}, 'BetaDiff';
    {'Latency','BetaDiff'}, 'Latency+BetaDiff'
};

allResults = table();

%% ----------------- 5) Run LOSO classification for each comparison -----------------
for c = 1:size(comparisons,1)
    gA = comparisons{c,1};
    gB = comparisons{c,2};
    compName = comparisons{c,3};

    % Filter subjectData to the two groups
    idx = subjectData.Group == gA | subjectData.Group == gB;
    D = subjectData(idx, :);

    % Binary response: positive class = gB
    y = double(D.Group == gB);

    fprintf('\n============================================================\n');
    fprintf('Comparison: %s | N=%d (pos=%d, neg=%d)\n', compName, height(D), sum(y==1), sum(y==0));

    for f = 1:size(featureSets,1)
        feats = featureSets{f,1};
        featName = featureSets{f,2};

        % Remove subjects with missing features
        keep = true(height(D),1);
        for k = 1:numel(feats)
            keep = keep & isfinite(D.(feats{k}));
        end
        D2 = D(keep,:);
        y2 = y(keep);

        fprintf('\n  Feature set: %s | N=%d (pos=%d, neg=%d)\n', featName, height(D2), sum(y2==1), sum(y2==0));

        if height(D2) < 10 || numel(unique(y2)) < 2
            fprintf('  Skipped (insufficient data).\n');
            continue;
        end

        % Run LOSO with corrected permutation test
        out = run_loso_with_metrics_enhanced(D2, y2, feats, P_perm);

        % Store results
        row = table();
        row.Comparison = string(compName);
        row.FeatureSet = string(featName);
        row.N = height(D2);
        row.N_pos = sum(y2==1);
        row.N_neg = sum(y2==0);
        row.AUC = out.AUC;
        row.AUC_perm_p = out.AUC_perm_p;
        row.Accuracy = out.Accuracy;
        row.Sensitivity = out.Sensitivity;
        row.Specificity = out.Specificity;
        row.Threshold = out.Threshold;

        allResults = [allResults; row]; %#ok<AGROW>

        % Optional: plot ROC per run (comment out if not needed)
        % figure('Color','w'); plot(out.FPR, out.TPR, 'LineWidth', 2);
        % xlabel('False Positive Rate'); ylabel('True Positive Rate');
        % title(sprintf('%s | %s | AUC=%.3f', compName, featName, out.AUC));
        % grid on;
    end
end

%% ----------------- 6) Print summary -----------------
fprintf('\n================== LOSO Classification Summary ==================\n');
disp(allResults);
%% ----------------- Figure 7: Combined ROC (LOSO-based) -----------------

figure('Color','w','Position',[200 200 600 500]);
hold on;

colors = [0 0.447 0.741;
          0.850 0.325 0.098];

comparisons_to_plot = {"HC vs AD","HC vs MCI"};

for i = 1:2
    
    compName = comparisons_to_plot{i};
    
    % Select latency-only rows
    rowIdx = strcmp(allResults.Comparison, compName) & ...
             strcmp(allResults.FeatureSet, "Latency");
    
    if ~any(rowIdx)
        continue;
    end
    
    % Recompute LOSO probs properly
    gA = comparisons{i,1};
    gB = comparisons{i,2};
    
    idx = subjectData.Group == gA | subjectData.Group == gB;
    D = subjectData(idx,:);
    y = double(D.Group == gB);
    
    % Use LOSO helper again to get probs
    out = run_loso_with_metrics_enhanced(D, y, {'Latency'}, 100);
    
    plot(out.FPR, out.TPR, ...
        'LineWidth', 2.5, ...
        'Color', colors(i,:), ...
        'DisplayName', sprintf('%s (AUC = %.3f)', compName, out.AUC));
end

plot([0 1],[0 1],'k--','HandleVisibility','off');

xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('Location','southeast');
grid on;
axis square;
title('Diagnostic Utility of Hemodynamic Latency (LOSO)');
%% ============================================================
% Helper: LOSO logistic regression + metrics + corrected permutation AUC p-value
%% ============================================================
function out = run_loso_with_metrics_enhanced(D, y, feats, P_perm)
% D: table with features
% y: binary labels (0/1), positive class = 1
% feats: cellstr feature names
% P_perm: number of permutations for AUC p-value

n = height(D);
probs = nan(n,1);
thr_train = nan(n,1);

% LOSO
for i = 1:n
    trainIdx = true(n,1); trainIdx(i) = false;
    testIdx  = ~trainIdx;

    Xtrain = D{trainIdx, feats};
    ytrain = y(trainIdx);
    Xtest  = D{testIdx, feats};

    % Logistic regression (binomial GLM)
    tblTrain = array2table(Xtrain, 'VariableNames', feats);
    tblTrain.y = ytrain;
    formula = ['y ~ ' strjoin(feats, ' + ')];

    mdl = fitglm(tblTrain, formula, 'Distribution', 'binomial');

    % Predict probability for held-out subject
    tblTest = array2table(Xtest, 'VariableNames', feats);
    probs(i) = predict(mdl, tblTest);

    % Choose Youden threshold on TRAINING fold (for reference)
    pTrain = predict(mdl, tblTrain(:,feats));
    [~,~,~,aucTrain] = perfcurve(ytrain, pTrain, 1); %#ok<ASGLU>
    [FPR,TPR,THR] = perfcurve(ytrain, pTrain, 1);
    youden = TPR - FPR;
    [~,j] = max(youden);
    if ~isempty(j) && j >= 1 && j <= numel(THR)
        thr_train(i) = THR(j);
    end
end

% Final threshold: median of fold-wise Youden thresholds
thr = median(thr_train, 'omitnan');
if ~isfinite(thr)
    thr = 0.5; % fallback
end

yhat = probs >= thr;

% Metrics on LOSO predictions
TP = sum((yhat==1) & (y==1));
TN = sum((yhat==0) & (y==0));
FP = sum((yhat==1) & (y==0));
FN = sum((yhat==0) & (y==1));

acc  = (TP + TN) / n;
sens = TP / max((TP + FN),1);
spec = TN / max((TN + FP),1);
% ROC/AUC (based on LOSO predicted probabilities)
[FPR,TPR,~,AUC] = perfcurve(y, probs, 1);

% Bootstrap CI for AUC (resample subjects with replacement)
B = 2000;
boot_auc = nan(B,1);
rng(3);

nBoot = numel(y);

for b = 1:B
    idx = randsample(nBoot, nBoot, true);
    y_b = y(idx);
    p_b = probs(idx);

    % perfcurve requires both classes
    if numel(unique(y_b)) < 2
        continue;
    end

    [~,~,~,boot_auc(b)] = perfcurve(y_b, p_b, 1);
end

boot_auc = boot_auc(~isnan(boot_auc));
if ~isempty(boot_auc)
    CI = prctile(boot_auc, [2.5 97.5]);
else
    CI = [NaN NaN];
end

% Correct permutation test for AUC: shuffle labels only (keep probs fixed)
perm_auc = nan(P_perm,1);
rng(2);

for k = 1:P_perm
    y_perm = y(randperm(nBoot));   % shuffle labels only
    [~,~,~,perm_auc(k)] = perfcurve(y_perm, probs, 1);
end

AUC_perm_p = mean(perm_auc >= AUC);

% Pack outputs
out = struct();
out.probs = probs;
out.Threshold = thr;
out.Accuracy = acc;
out.Sensitivity = sens;
out.Specificity = spec;
out.FPR = FPR;
out.TPR = TPR;
out.AUC = AUC;
out.AUC_CI_low  = CI(1);
out.AUC_CI_high = CI(2);
out.AUC_perm_p = AUC_perm_p;
end