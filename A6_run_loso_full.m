%% run_loso_full.m

clear; clc; close all;
rng(0);

%% ---------------- User parameters ----------------
load('D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed\Stroop_AllSubjects_NIRS_HRF_TTest_05.mat')

rng(0);

%% ---------------- User parameters ----------------
fs = 8.138;
snrThreshold = 30;
minDuration = 1.2131;
groupNames = {'HC','MCI','AD'};
colors = [0, 0.447, 0.741; 0.85, 0.325, 0.098; 0.929, 0.694, 0.125];

P_perm = 5000;   % permutation
B_boot = 1000;   % AUC bootstrap 

%% ------------- 1. Extract features from allSubjects -------------
fprintf('Extracting features and calculating channel rejection stats...\n');
subjectData = table();
rejection_ratios = [];

for s = 1:numel(allSubjects)
    subj = allSubjects(s).results;
    subjID = subj.ID;
    g = subj.Group;                % numeric label expected (1,2,3)
    trials = subj.Trials;
    rawSNR = subj.SNR;             % assume vector, take first 48 channels
    subj_ch_snr = rawSNR(1:48);
    
    subj_latencies = [];
    subj_betas_corr = [];
    subj_betas_incor = [];
    total_sig_channels_count = 0;
    rejected_sig_channels_count = 0;
    
    for t = 1:numel(trials)
        tr = trials(t).results;
        if tr.duration < minDuration, continue; end
        sigCh = find(tr.p_HbO < 0.05);
        if isempty(sigCh), continue; end
        
        for ch = sigCh'
            total_sig_channels_count = total_sig_channels_count + 1;
            if subj_ch_snr(ch) < snrThreshold
                rejected_sig_channels_count = rejected_sig_channels_count + 1;
                continue;
            end
            ts = tr.HbO(:, ch);
            [~, peak_idx] = max(ts);
            lat = (peak_idx - 1) / fs;
            if lat < 0.5 || lat > 15, continue; end
            if isfield(tr, 'beta_HbO') && numel(tr.beta_HbO) >= ch
                if strcmpi(tr.ACC, 'correct')
                    subj_latencies(end+1,1) = lat; %#ok<SAGROW>
                    subj_betas_corr(end+1,1) = tr.beta_HbO(ch); %#ok<SAGROW>
                elseif strcmpi(tr.ACC, 'incorrect')
                    subj_betas_incor(end+1,1) = tr.beta_HbO(ch); %#ok<SAGROW>
                end
            end
        end
    end
    
    if total_sig_channels_count > 0
        rejection_ratios(end+1) = (rejected_sig_channels_count / total_sig_channels_count) * 100; %#ok<SAGROW>
    end
    
    % Save subject row only if both latency and incorrect betas exist
    if ~isempty(subj_latencies) && ~isempty(subj_betas_incor)
        res = table();
        res.SubjectID = {subjID};
        res.Group = g;
        res.Latency = mean(subj_latencies);
        res.BetaDiff = mean(subj_betas_corr) - mean(subj_betas_incor);
        subjectData = [subjectData; res]; %#ok<AGROW>
    end
end

% Quick summary
fprintf('Extracted %d subjects with valid features.\n', height(subjectData));

%% ------------- 2. LOSO Validation (HC vs AD, HC vs MCI) -------------
fprintf('\nPerforming LOSO Validation and Calculating Metrics...\n');

% HC vs AD (labels 1 and 3)
idx_ad = (subjectData.Group == 1) | (subjectData.Group == 3);
T_ad = subjectData(idx_ad, :);
res_ad = run_loso_with_metrics_enhanced(T_ad, 1, 3, P_perm, B_boot);

% HC vs MCI (labels 1 and 2)
idx_mci = (subjectData.Group == 1) | (subjectData.Group == 2);
T_mci = subjectData(idx_mci, :);
res_mci = run_loso_with_metrics_enhanced(T_mci, 1, 2, P_perm, B_boot);

%% ------------- 3. Print Final Report -------------
line_sep = repmat('=',1,70);
fprintf('\n%s\n   FINAL VALIDATED DIAGNOSTIC REPORT\n%s\n', line_sep, line_sep);

fprintf('1) HC vs AD (LOOCV)\n');
fprintf('   - nPos (AD) = %d, nNeg (HC) = %d\n', res_ad.nPos, res_ad.nNeg);
fprintf('   - AUC = %.4f (95%% CI [%.4f, %.4f]), permutation p = %.4f\n', res_ad.AUC, res_ad.AUC_CI(1), res_ad.AUC_CI(2), res_ad.AUC_perm_p);
fprintf('   - Sensitivity (median fold threshold) = %.2f%%, Specificity = %.2f%%\n', res_ad.sens_final*100, res_ad.spec_final*100);
fprintf('   - Sensitivity (fold mean ± CI) = %.2f%% [%.2f, %.2f]\n', res_ad.sens_mean*100, res_ad.sens_CI(1)*100, res_ad.sens_CI(2)*100);
fprintf('   - Specificity (fold mean ± CI) = %.2f%% [%.2f, %.2f]\n\n', res_ad.spec_mean*100, res_ad.spec_CI(1)*100, res_ad.spec_CI(2)*100);

fprintf('2) HC vs MCI (LOOCV)\n');
fprintf('   - nPos (MCI) = %d, nNeg (HC) = %d\n', res_mci.nPos, res_mci.nNeg);
fprintf('   - AUC = %.4f (95%% CI [%.4f, %.4f]), permutation p = %.4f\n', res_mci.AUC, res_mci.AUC_CI(1), res_mci.AUC_CI(2), res_mci.AUC_perm_p);
fprintf('   - Sensitivity (median fold threshold) = %.2f%%, Specificity = %.2f%%\n', res_mci.sens_final*100, res_mci.spec_final*100);
fprintf('   - Sensitivity (fold mean ± CI) = %.2f%% [%.2f, %.2f]\n', res_mci.sens_mean*100, res_mci.sens_CI(1)*100, res_mci.sens_CI(2)*100);
fprintf('   - Specificity (fold mean ± CI) = %.2f%% [%.2f, %.2f]\n\n', res_mci.spec_mean*100, res_mci.spec_CI(1)*100, res_mci.spec_CI(2)*100);

if ~isempty(rejection_ratios)
    mean_rej = mean(rejection_ratios);
    std_rej = std(rejection_ratios);
    fprintf('3) Channel Rejection Statistics (Low SNR):\n   - Mean rejection ratio of significant channels: %.2f%% ± %.2f%%\n', mean_rej, std_rej);
else
    fprintf('3) Channel Rejection Statistics: No significant channels found.\n');
end
fprintf('%s\n', line_sep);

%% ------------- 4. ROC Plot -------------
figure('Color','w','Position',[200 200 700 520]); hold on;
plot(res_ad.FPR, res_ad.TPR, 'LineWidth', 3, 'Color', colors(3,:));
plot(res_mci.FPR, res_mci.TPR, 'LineWidth', 3, 'Color', colors(2,:));
plot([0 1],[0 1],'k--','LineWidth',1.2);
xlabel('False Positive Rate (1 - Specificity)'); ylabel('True Positive Rate (Sensitivity)');
legend({sprintf('HC vs AD (AUC=%.3f)', res_ad.AUC), sprintf('HC vs MCI (AUC=%.3f)', res_mci.AUC)}, 'Location','southeast');
grid on; axis square; title('LOOCV ROC Curves');
%% ------------- 7. Binary Classification: HC vs. Patients (MCI + AD) -------------
fprintf('\nPerforming Binary Classification: HC(Neg) vs. Patients(Pos)...\n');

T_rest = subjectData;
T_rest.Group = double(T_rest.Group ~= 1); 

% 2. LOSO
% label_neg = 0, label_pos = 1
res_rest = run_loso_with_metrics_enhanced(T_rest, 0, 1, P_perm, B_boot);

% 3. Result
line_sep = repmat('-',1,60);
fprintf('\n%s\n   HC vs. Patients (MCI + AD) Diagnostic Report\n%s\n', line_sep, line_sep);
fprintf('   - nPos (Patients) = %d\n', res_rest.nPos); % 예상: 40
fprintf('   - nNeg (HC) = %d\n', res_rest.nNeg);       % 예상: 14
fprintf('   - AUC = %.4f (95%% CI [%.4f, %.4f]), p = %.4f\n', ...
    res_rest.AUC, res_rest.AUC_CI(1), res_rest.AUC_CI(2), res_rest.AUC_perm_p);
fprintf('   - Sensitivity = %.2f%%, Specificity = %.2f%%\n', ...
    res_rest.sens_final*100, res_rest.spec_final*100);
fprintf('%s\n', line_sep);

%% ------------- 8. Confusion Matrix -------------
figure('Color','w','Position',[100 100 500 400]);
Y_cat = categorical(res_rest.bin_resp, [0 1], {'HC', 'Patients'});
P_cat = categorical(res_rest.probs >= res_rest.optThreshold_median, [0 1], {'HC', 'Patients'});

cc = confusionchart(Y_cat, P_cat);
cc.Title = ['HC vs. Patients (AUC = ', num2str(res_rest.AUC, '%.3f'), ')'];
%% ================= Helper function =================
function results = run_loso_with_metrics_enhanced(T_subset, label_neg, label_pos, P_perm, B_boot)
    rng(0);
    if nargin < 4, P_perm = 5000; end
    if nargin < 5, B_boot = 1000; end

    nSubj = height(T_subset);
    bin_resp = double(T_subset.Group == label_pos); % 1 = positive
    features = {'Latency','BetaDiff'};

    % Validate features exist
    for f = 1:numel(features)
        if ~ismember(features{f}, T_subset.Properties.VariableNames)
            error('Feature %s missing from table.', features{f});
        end
    end

    probs = nan(nSubj,1);
    optThresholds = nan(nSubj,1);
    sens_per_fold = nan(nSubj,1);
    spec_per_fold = nan(nSubj,1);

    for i = 1:nSubj
        train_idx = setdiff(1:nSubj, i);
        test_idx = i;

        X_train = T_subset{train_idx, features};
        X_test = T_subset{test_idx, features};
        y_train = bin_resp(train_idx);

        % Drop columns with NaN or zero variance in training
        col_ok = all(~isnan(X_train),1) & (std(X_train,[],1) > 0);
        X_train = X_train(:, col_ok);
        X_test = X_test(:, col_ok);
        feat_names = features(col_ok);

        % Standardize using training mean/std
        mu = mean(X_train,1); sigma = std(X_train,[],1);
        sigma(sigma==0) = 1;
        X_train_z = (X_train - mu) ./ sigma;
        X_test_z = (X_test - mu) ./ sigma;

        % Fit logistic regression on training
        Ttrain = array2table(X_train_z, 'VariableNames', feat_names);
        Ttrain.Group = y_train;
        mdl = fitglm(Ttrain, ['Group ~ ', strjoin(feat_names,' + ')], 'Distribution','binomial','Link','logit');

        % Predict probability for left-out subject
        Ttest = array2table(X_test_z, 'VariableNames', feat_names);
        probs(i) = predict(mdl, Ttest);

        % Determine optimal threshold on training (Youden)
        [Xroc, Yroc, Troc] = perfcurve(y_train, mdl.Fitted.Probability, 1);
        J = Yroc + (1 - Xroc) - 1;
        [~, idxJ] = max(J);
        thr = Troc(idxJ);
        optThresholds(i) = thr;

        % Evaluate sens/spec on training at thr
        yhat_train = mdl.Fitted.Probability >= thr;
        if sum(y_train==1) > 0
            sens_per_fold(i) = sum(yhat_train == 1 & y_train == 1) / sum(y_train == 1);
        else
            sens_per_fold(i) = NaN;
        end
        if sum(y_train==0) > 0
            spec_per_fold(i) = sum(yhat_train == 0 & y_train == 0) / sum(y_train == 0);
        else
            spec_per_fold(i) = NaN;
        end
    end

    % Aggregate ROC and AUC on LOOCV predictions
    [FPR, TPR, T_roc, AUC] = perfcurve(bin_resp, probs, 1);

    % AUC bootstrap CI (subject-level bootstrap)
    auc_boot = nan(B_boot,1);
    rng(1);
    for b = 1:B_boot
        idx_b = randsample(nSubj, nSubj, true);
        [~,~,~, auc_boot(b)] = perfcurve(bin_resp(idx_b), probs(idx_b), 1);
    end
    AUC_CI = prctile(auc_boot, [2.5 97.5]);

    % Permutation test for AUC
    perm_auc = nan(P_perm,1);
    rng(2);
    for k = 1:P_perm
        perm_idx = randsample(nSubj, nSubj, false);
        [~,~,~, perm_auc(k)] = perfcurve(bin_resp(perm_idx), probs, 1);
    end
    AUC_perm_p = mean(perm_auc >= AUC);

    % Median threshold across folds and final confusion
    median_thr = median(optThresholds, 'omitnan');
    yhat_final = probs >= median_thr;
    TP = sum(yhat_final == 1 & bin_resp == 1);
    FP = sum(yhat_final == 1 & bin_resp == 0);
    TN = sum(yhat_final == 0 & bin_resp == 0);
    FN = sum(yhat_final == 0 & bin_resp == 1);
    sens_final = TP / max((TP + FN),1);
    spec_final = TN / max((TN + FP),1);

    % sens/spec CI from fold distributions
    sens_CI = prctile(sens_per_fold, [2.5 97.5]);
    spec_CI = prctile(spec_per_fold, [2.5 97.5]);

    % Pack results
    results.FPR = FPR; results.TPR = TPR; results.T_roc = T_roc;
    results.AUC = AUC; results.AUC_CI = AUC_CI; results.AUC_perm_p = AUC_perm_p;
    results.sens_mean = mean(sens_per_fold,'omitnan'); results.sens_CI = sens_CI;
    results.spec_mean = mean(spec_per_fold,'omitnan'); results.spec_CI = spec_CI;
    results.optThresholds = optThresholds; results.optThreshold_median = median_thr;
    results.confusion = struct('TP',TP,'FP',FP,'TN',TN,'FN',FN);
    results.sens_final = sens_final; results.spec_final = spec_final;
    results.nPos = sum(bin_resp==1); results.nNeg = sum(bin_resp==0);
    results.probs = probs; results.bin_resp = bin_resp;
    results.FPR = FPR; results.TPR = TPR;
end

