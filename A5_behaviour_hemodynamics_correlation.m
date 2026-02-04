%% ============================================================
% ADVANCED BRAIN-BEHAVIOR INTEGRATION
% Incorporating SNR Filtering, ID Matching, and Partial Correlation
%% ============================================================
clear; clc; close all;
set(0, 'DefaultFigureColor', 'w');
%% ----------------- 1. Load Data -----------------
load('D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed\Stroop_AllSubjects_NIRS_HRF_TTest_05.mat');
behavTable = readtable('D:\1_MPI_CBS\0_Research\0_MCI_classification\Raw data\subject_info.xlsx');

fs = 8.138;
snrThreshold = 30;
groupNames = {'HC', 'MCI', 'AD'};
colors = [0, 0.447, 0.741; 0.85, 0.325, 0.098; 0.929, 0.694, 0.125];

%% ----------------- 2. Robust Feature Extraction (with SNR Filter) -----------------
subjectFeatures = table();

for s = 1:numel(allSubjects)
    subj = allSubjects(s).results;
    subjID = subj.ID; % 예: 'A1', 'B2' 등
    subjSNR = subj.SNR;
    
    subj_latencies = [];
    for t = 1:numel(subj.Trials)
        tr = subj.Trials(t).results;    
        if tr.duration < 1.2131, continue; end
        if ~strcmpi(tr.ACC, 'correct'), continue; end
        
        sigCh = find(tr.p_HbO < 0.05);
        for ch = sigCh'
            if subjSNR(ch) < snrThreshold, continue; end
            
            ts = tr.HbO(:, ch);
            L = length(ts);
            F = min(31, L); if mod(F, 2) == 0, F = F - 1; end 
            if F > 3, ts_s = sgolayfilt(ts, 3, F); else, ts_s = movmean(ts, 3); end
            
            [~, idx] = max(ts_s);
            pTime = (idx - 1) / fs;
            if pTime >= 0 && pTime <= 15, subj_latencies = [subj_latencies; pTime]; end
        end
    end
    
    if ~isempty(subj_latencies)
        res = table();
        res.ID = {subjID}; 
        res.Group = subj.Group;
        res.Latency = mean(subj_latencies);
        subjectFeatures = [subjectFeatures; res];
    end
end

%% ----------------- 3. Smart Merge by ID -----------------
finalData = table();
for i = 1:height(subjectFeatures)
    targetID = subjectFeatures.ID{i};
    rowIdx = find(strcmpi(behavTable.SubjectID, targetID));
    
    if ~isempty(rowIdx)
        temp = subjectFeatures(i, :);
        temp.MMSE = (behavTable.MMSE(rowIdx) + behavTable.MMSE_1(rowIdx)) / 2;
        temp.MoCA = behavTable.Moca(rowIdx);
        temp.Accuracy = behavTable.K_CWST(rowIdx);
        temp.Education = behavTable.Education(rowIdx);
        temp.Age = behavTable.Age(rowIdx);
        finalData = [finalData; temp];
    end
end

%% ----------------- 4. Statistical Analysis & Visualization -----------------
metrics = {'MMSE', 'MoCA', 'Accuracy', 'Education'};
metricLabels = {'MMSE Score', 'MoCA Score', 'Stroop Accuracy (%)', 'Education (Years)'};

figure('Color', 'w', 'Position', [100, 100, 1200, 900]);
tlo = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, 'Brain-Behavior Coupling: Hemodynamic Lag vs. Clinical Scores', ...
    'FontSize', 18, 'FontWeight', 'bold');

for m = 1:4
    target = metrics{m};
    
    lat = finalData.Latency(:);
    val = finalData.(target)(:);
    
    % --- [Step 1] Partial Spearman Correlation (Stats) ---
    if ~strcmpi(target, 'Education') && ~strcmpi(target, 'Age')
        covars = [finalData.Education(:), finalData.Age(:)];
        [rho_val, p_perm, ci_boot] = partial_spearman_robust(lat, val, covars, 2000, 5000);
        adjStr = '(adj. Edu, Age)';
    else
        [rho_val, p_perm, ci_boot] = partial_spearman_robust(lat, val, [], 2000, 5000);
        adjStr = '';
    end
    
    % --- [Step 2] Visualization ---
    ax = nexttile; hold on;
    
    % (A) 전체 데이터에 대한 회귀선 추가 (Global Trend)
    % NaN 제거 후 Polyfit
    valid_mask = ~isnan(lat) & ~isnan(val);
    gX = lat(valid_mask); gY = val(valid_mask);
    
    if ~isempty(gX)
        [p_glob, S_glob] = polyfit(gX, gY, 1);
        x_lin = linspace(min(lat), max(lat), 100);
        [y_fit, delta] = polyval(p_glob, x_lin, S_glob);
        
        % 1. Global 95% CI (회색 음영)
        fill([x_lin, fliplr(x_lin)], [y_fit-delta, fliplr(y_fit+delta)], ...
             'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
         
        % 2. Global Regression Line (검은색 굵은 실선)
        plot(x_lin, y_fit, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Overall Trend');
    end
    
    % (B) 그룹별 산점도 (Scatter Only)
    % 그룹별 회귀선은 복잡해 보일 수 있으므로 제거하거나 얇게 처리 (여기선 점만 표시)
    for g = 1:3
        idx = (finalData.Group == g);
        gx = finalData.Latency(idx); gy = finalData.(target)(idx);
        if isempty(gx), continue; end
        
        % 산점도 (Scatter)
        scatter(gx, gy, 80, colors(g,:), 'filled', 'MarkerFaceAlpha', 0.6, ...
            'DisplayName', groupNames{g});
    end
    
    % --- [Step 3] Labels & Style ---
    pStr = sprintf('p_{perm} = %.4f', p_perm);
    if p_perm < 0.001, pStr = 'p_{perm} < .001'; end
    
    subStr = sprintf('\\rho = %.3f, 95%% CI [%.2f, %.2f], %s %s', ...
        rho_val, ci_boot(1), ci_boot(2), pStr, adjStr);
    
    title(metricLabels{m}, 'FontSize', 14, 'FontWeight', 'bold');
    subtitle(subStr, 'FontSize', 11, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);
    
    xlabel('Hemodynamic Latency (s)');
    ylabel(metricLabels{m});
    grid on; set(gca, 'TickDir', 'out', 'Box', 'on', 'LineWidth', 1.2);
    
    % 범례는 첫 번째 그래프에만 표시 (깔끔하게)
    if m == 1
        legend('Location', 'best', 'Box', 'off', 'FontSize', 10); 
    end
end

% 저장
%exportgraphics(tlo, fullfile(outputFolder, 'Figure_BrainBehavior_Subtitles.png'), 'Resolution', 300);

% normalize IDs
behavIDs = lower(strtrim(string(behavTable.SubjectID)));
subjectFeatures.ID = string(subjectFeatures.ID); % ensure string
subjectFeatures.ID = lower(strtrim(subjectFeatures.ID));

finalData = table();
for i = 1:height(subjectFeatures)
    targetID = subjectFeatures.ID(i);
    rowIdx = find(behavIDs == targetID);
    if isempty(rowIdx), continue; end
    temp = subjectFeatures(i,:);
    % safe column access (use exist checks)
    if any(strcmpi(behavTable.Properties.VariableNames,'MMSE')) && any(strcmpi(behavTable.Properties.VariableNames,'MMSE_1'))
        temp.MMSE = mean([behavTable.MMSE(rowIdx), behavTable.MMSE_1(rowIdx)], 'omitnan');
    elseif any(strcmpi(behavTable.Properties.VariableNames,'MMSE'))
        temp.MMSE = behavTable.MMSE(rowIdx);
    else
        temp.MMSE = NaN;
    end
    % MoCA (case-insensitive)
    if any(strcmpi(behavTable.Properties.VariableNames,'Moca')) || any(strcmpi(behavTable.Properties.VariableNames,'MoCA'))
        temp.MoCA = behavTable{rowIdx, strcmpi(behavTable.Properties.VariableNames,'Moca') | strcmpi(behavTable.Properties.VariableNames,'MoCA')};
    else
        temp.MoCA = NaN;
    end
    % Stroop / Education / Age
    if any(strcmpi(behavTable.Properties.VariableNames,'K_CWST')), temp.Accuracy = behavTable.K_CWST(rowIdx); else temp.Accuracy = NaN; end
    if any(strcmpi(behavTable.Properties.VariableNames,'Education')), temp.Education = behavTable.Education(rowIdx); else temp.Education = NaN; end
    if any(strcmpi(behavTable.Properties.VariableNames,'Age')), temp.Age = behavTable.Age(rowIdx); else temp.Age = NaN; end

    finalData = [finalData; temp];
end

% Ensure numeric types
finalData.Latency = double(finalData.Latency);
finalData.Education = double(finalData.Education);
finalData.Age = double(finalData.Age);
finalData.MMSE = double(finalData.MMSE);
finalData.MoCA = double(finalData.MoCA);
finalData.Accuracy = double(finalData.Accuracy);

% Ensure Group is categorical with known levels
if ~iscategorical(finalData.Group)
    finalData.Group = categorical(finalData.Group, unique(finalData.Group));
end
%% --- Example usage for MMSE (and Age sensitivity) ---
[rho_mmse, p_mmse, ci_mmse] = partial_spearman_boot(finalData.Latency, finalData.MMSE, finalData.Education, 2000);
% additionally adjust for Age by residualizing on two covariates
% create combined covariate matrix by projecting out both (use linear regression on ranks)
% (or call function twice with residualization on [Education Age] implemented)

fprintf('MMSE partial Spearman (adj Education): rho=%.3f, p~%.3f, 95CI=[%.3f, %.3f], n=%d\n', rho_mmse, p_mmse, ci_mmse(1), ci_mmse(2), sum(~isnan(finalData.Latency) & ~isnan(finalData.MMSE) & ~isnan(finalData.Education)));
[rho_mmse, p_mmse, ci_mmse] = partial_spearman_boot(finalData.Latency, finalData.MoCA, finalData.Education, 2000);
% additionally adjust for Age by residualizing on two covariates
% create combined covariate matrix by projecting out both (use linear regression on ranks)
% (or call function twice with residualization on [Education Age] implemented)

fprintf('MoCA partial Spearman (adj Education): rho=%.3f, p~%.3f, 95CI=[%.3f, %.3f], n=%d\n', rho_mmse, p_mmse, ci_mmse(1), ci_mmse(2), sum(~isnan(finalData.Latency) & ~isnan(finalData.MMSE) & ~isnan(finalData.Education)));
[rho_mmse, p_mmse, ci_mmse] = partial_spearman_boot(finalData.Latency, finalData.Accuracy, finalData.Education, 2000);
% additionally adjust for Age by residualizing on two covariates
% create combined covariate matrix by projecting out both (use linear regression on ranks)
% (or call function twice with residualization on [Education Age] implemented)

fprintf('Accuracy partial Spearman (adj Education): rho=%.3f, p~%.3f, 95CI=[%.3f, %.3f], n=%d\n', rho_mmse, p_mmse, ci_mmse(1), ci_mmse(2), sum(~isnan(finalData.Latency) & ~isnan(finalData.MMSE) & ~isnan(finalData.Education)));

%% --- Pooled residual plot (adjust for Education & Age) ---
targets = {'MMSE','MoCA','Accuracy'}; 
B = 2000; P = 5000;

for i = 1:numel(targets)
    Yname = targets{i};
    X = finalData.Latency;
    Y = finalData.(Yname);
    C = finalData.Education; % adjust for education
    % remove missing
    ok = ~isnan(X) & ~isnan(Y) & ~isnan(C);
    Xv = X(ok); Yv = Y(ok); Cv = C(ok);
    n = numel(Xv);
    if n < 6
        fprintf('%s: insufficient data (n=%d). Skipping.\n', Yname, n);
        continue;
    end

    % rank-transform
    rx = tiedrank(Xv); ry = tiedrank(Yv); rc = tiedrank(Cv);

    % residualize ranks on covariate
    bx = regress(rx, [ones(n,1), rc]); rx_res = rx - [ones(n,1), rc]*bx;
    by = regress(ry, [ones(n,1), rc]); ry_res = ry - [ones(n,1), rc]*by;

    % observed rho
    rho_obs = corr(rx_res, ry_res, 'Type', 'Pearson');

    % bootstrap
    rng(0);
    bootrho = nan(B,1);
    for b = 1:B
        idx = randsample(n, n, true);
        rx_b = rx(idx); ry_b = ry(idx); rc_b = rc(idx);
        bx_b = regress(rx_b, [ones(n,1), rc_b]); rxr_b = rx_b - [ones(n,1), rc_b]*bx_b;
        by_b = regress(ry_b, [ones(n,1), rc_b]); ryr_b = ry_b - [ones(n,1), rc_b]*by_b;
        bootrho(b) = corr(rxr_b, ryr_b, 'Type', 'Pearson');
    end
    ci = prctile(bootrho, [2.5 97.5]);
    % bootstrap p approximated by null-centered comparison (recommended: permutation)
    p_boot = mean(abs(bootrho) >= abs(rho_obs)); % alternative: permutation below

    % permutation test
    rng(1);
    perm_rho = nan(P,1);
    for k = 1:P
        perm_idx = randsample(n, n, false);
        ry_p = ry(perm_idx);
        by_p = regress(ry_p, [ones(n,1), rc]); ryr_p = ry_p - [ones(n,1), rc]*by_p;
        perm_rho(k) = corr(rx_res, ryr_p, 'Type', 'Pearson');
    end
    p_perm = mean(abs(perm_rho) >= abs(rho_obs));

    % print
    fprintf('%s: rho = %.3f, n = %d\n', Yname, rho_obs, n);
    fprintf('  Bootstrap 95%% CI = [%.3f, %.3f]\n', ci(1), ci(2));
    fprintf('  Bootstrap p (null-approx) = %.4f\n', p_boot);
    fprintf('  Permutation p (two-sided) = %.4f\n\n', p_perm);
end

%% --- Partial Spearman with bootstrap CI (function) ---
function [rho, pval, ci] = partial_spearman_boot(x, y, covar, B)
    if nargin < 4, B = 2000; end
    % remove missing
    ok = ~isnan(x) & ~isnan(y) & ~isnan(covar);
    x = x(ok); y = y(ok); c = covar(ok);
    n = numel(x);
    if n < 6
        rho = NaN; pval = NaN; ci = [NaN NaN]; return;
    end
    % rank transform
    rx = tiedrank(x); ry = tiedrank(y); rc = tiedrank(c);
    % residualize ranks on covariate
    bx = regress(rx, [ones(n,1), rc]); rxr = rx - [ones(n,1), rc]*bx;
    by = regress(ry, [ones(n,1), rc]); ryr = ry - [ones(n,1), rc]*by;
    rho = corr(rxr, ryr, 'Type', 'Pearson');
    % p-value via permutation (optional) or analytic
    % bootstrap CI
    bootrho = nan(B,1);
    rng(0);
    for b = 1:B
        idx = randsample(n, n, true);
        rx_b = rx(idx); ry_b = ry(idx); rc_b = rc(idx);
        bx_b = regress(rx_b, [ones(n,1), rc_b]); rxr_b = rx_b - [ones(n,1), rc_b]*bx_b;
        by_b = regress(ry_b, [ones(n,1), rc_b]); ryr_b = ry_b - [ones(n,1), rc_b]*by_b;
        bootrho(b) = corr(rxr_b, ryr_b, 'Type', 'Pearson');
    end
    ci = prctile(bootrho, [2.5 97.5]);
    % approximate p-value (two-sided)
    pval = 2*min(mean(bootrho >= rho), mean(bootrho <= rho));
end
function [rho, p_perm, ci_boot] = partial_spearman_robust(x, y, covar, B, P)
    x = x(:); 
    y = y(:);
    
    if isempty(covar)
        ok = ~isnan(x) & ~isnan(y);
        x = x(ok); y = y(ok); 
        n = numel(x);
        rx = tiedrank(x); ry = tiedrank(y);
        rx_res = rx - mean(rx); ry_res = ry - mean(ry);
    else
        ok = ~isnan(x) & ~isnan(y) & all(~isnan(covar), 2);
        
        x = x(ok); 
        y = y(ok); 
        c = covar(ok, :); 
        
        n = numel(x);
        rx = tiedrank(x); 
        ry = tiedrank(y);
        rc = tiedrank(c); 
        
        X_cov = [ones(n, 1), rc];
        bx = X_cov \ rx; rx_res = rx - X_cov * bx;
        by = X_cov \ ry; ry_res = ry - X_cov * by;
    end
    
    rho = corr(rx_res, ry_res, 'Type', 'Pearson');
    
    bootrho = zeros(B, 1);
    for b = 1:B
        idx = randsample(n, n, true);
        if isempty(covar)
            bootrho(b) = corr(rx(idx), ry(idx), 'Type', 'Spearman');
        else
            rx_b = rx(idx); ry_b = ry(idx); rc_b = rc(idx);
            bx_b = regress(rx_b, [ones(n,1), rc_b]); rxr_b = rx_b - [ones(n,1), rc_b]*bx_b;
            by_b = regress(ry_b, [ones(n,1), rc_b]); ryr_b = ry_b - [ones(n,1), rc_b]*by_b;
            bootrho(b) = corr(rxr_b, ryr_b, 'Type', 'Pearson');
        end
    end
    ci_boot = prctile(bootrho, [2.5 97.5]);
    
    perm_rho = zeros(P, 1);
    for k = 1:P
        perm_idx = randsample(n, n, false);
        if isempty(covar)
            perm_rho(k) = corr(rx, ry(perm_idx), 'Type', 'Pearson');
        else
            ry_p = ry(perm_idx);
            by_p = regress(ry_p, [ones(n,1), rc]); ryr_p = ry_p - [ones(n,1), rc]*by_p;
            perm_rho(k) = corr(rx_res, ryr_p, 'Type', 'Pearson');
        end
    end
    p_perm = mean(abs(perm_rho) >= abs(rho));
end
