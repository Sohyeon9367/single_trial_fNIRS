%% ============================================================
% ADVANCED BRAIN-BEHAVIOR INTEGRATION (CLEANED / REVIEWER-READY)
% - Unified latency extraction via build_subject_features_latency()
% - Single ID merge (no duplicates)
% - Partial Spearman = Spearman on residualized ranks
% - Bootstrap for CI only, permutation for p-value only (two-sided, +1 correction)
% - Optional: within-group and group-controlled checks for reviewer defense
%% ============================================================

clear; clc; close all;
set(0, 'DefaultFigureColor', 'w');

%% ----------------- 1. Load Data -----------------
rawMatPath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed\Stroop_AllSubjects_NIRS_HRF_TTest_05.mat';
behavPath  = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Raw data\subject_info.xlsx';

behavTable = readtable(behavPath);

% Plot settings
groupNames = {'HC', 'MCI', 'AD'};
colors = [0, 0.447, 0.741; 0.85, 0.325, 0.098; 0.929, 0.694, 0.125];

%% ----------------- 2. Robust Feature Extraction (A5 logic) -----------------
% Ensure build_subject_features_latency is on path
% addpath(fullfile(pwd,'utils'));

rules.fs = 8.138;
rules.snrThreshold = 30;
rules.pThreshold = 0.05;
rules.correctOnly = false;
rules.minDuration = 1.2131;
rules.latencyWindow = [0 15];
rules.smoothing = 'sgolay';
rules.sgolayOrder = 3;
rules.maxFrame = 31;
rules.movmeanK = 3;

[subjectFeatures, trialTable, rulesUsed] = build_subject_features_latency(rawMatPath, rules);

% Expect subjectFeatures columns: ID, Group, Latency, NPeaksUsed
assert(any(strcmp(subjectFeatures.Properties.VariableNames,'ID')), 'subjectFeatures must contain ID');
assert(any(strcmp(subjectFeatures.Properties.VariableNames,'Group')), 'subjectFeatures must contain Group');
assert(any(strcmp(subjectFeatures.Properties.VariableNames,'Latency')), 'subjectFeatures must contain Latency');

%% ----------------- 3. Smart Merge by ID (single pass) -----------------
% Normalize IDs
behavIDs = lower(strtrim(string(behavTable.SubjectID)));
subjectFeatures.ID = lower(strtrim(string(subjectFeatures.ID)));

finalData = table();
for i = 1:height(subjectFeatures)
    targetID = subjectFeatures.ID(i);
    rowIdx = find(behavIDs == targetID, 1, 'first');
    if isempty(rowIdx), continue; end

    temp = subjectFeatures(i, :);

    % Safe column access (case-insensitive)
    temp.MMSE = NaN;
    if any(strcmpi(behavTable.Properties.VariableNames,'MMSE')) && any(strcmpi(behavTable.Properties.VariableNames,'MMSE_1'))
        temp.MMSE = mean([behavTable.MMSE(rowIdx), behavTable.MMSE_1(rowIdx)], 'omitnan');
    elseif any(strcmpi(behavTable.Properties.VariableNames,'MMSE'))
        temp.MMSE = behavTable.MMSE(rowIdx);
    end

    temp.MoCA = NaN;
    if any(strcmpi(behavTable.Properties.VariableNames,'Moca')) || any(strcmpi(behavTable.Properties.VariableNames,'MoCA'))
        col = strcmpi(behavTable.Properties.VariableNames,'Moca') | strcmpi(behavTable.Properties.VariableNames,'MoCA');
        temp.MoCA = behavTable{rowIdx, col};
    end

    temp.Accuracy = NaN;
    if any(strcmpi(behavTable.Properties.VariableNames,'K_CWST'))
        temp.Accuracy = behavTable.K_CWST(rowIdx);
    end

    temp.Education = NaN;
    if any(strcmpi(behavTable.Properties.VariableNames,'Education'))
        temp.Education = behavTable.Education(rowIdx);
    end

    temp.Age = NaN;
    if any(strcmpi(behavTable.Properties.VariableNames,'Age'))
        temp.Age = behavTable.Age(rowIdx);
    end

    finalData = [finalData; temp]; %#ok<AGROW>
end

% Ensure numeric types
finalData.Latency   = double(finalData.Latency);
finalData.Education = double(finalData.Education);
finalData.Age       = double(finalData.Age);
finalData.MMSE      = double(finalData.MMSE);
finalData.MoCA      = double(finalData.MoCA);
finalData.Accuracy  = double(finalData.Accuracy);

% Group should be numeric 1/2/3 or categorical; make it numeric for easy indexing
if iscategorical(finalData.Group)
    % If categorical labels exist, map explicitly if possible
    % Otherwise, fallback to category order.
    cats = categories(finalData.Group);
    gnum = double(finalData.Group);
    finalData.GroupNum = gnum;
else
    finalData.GroupNum = double(finalData.Group);
end

%% ----------------- 4. Statistical Analysis & Visualization -----------------
metrics = {'MMSE', 'MoCA', 'Accuracy', 'Education'};
metricLabels = {'MMSE Score', 'MoCA Score', 'Stroop Accuracy (%)', 'Education (Years)'};

B = 2000;   % bootstrap reps for CI
P = 5000;   % permutation reps for p-value

figure('Color','w','Position',[100, 100, 1200, 900]);
tlo = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
title(tlo,'Brain-Behavior Coupling: Hemodynamic Lag vs. Clinical Scores', ...
    'FontSize', 18, 'FontWeight', 'bold');

for m = 1:4
    target = metrics{m};
    ax = nexttile; hold on;

    X = finalData.Latency(:);
    Y = finalData.(target)(:);

    % Decide covariates:
    % - For MMSE/MoCA/Accuracy: adjust Education + Age
    % - For Education: no covariates
    if ~strcmpi(target,'Education') && ~strcmpi(target,'Age')
        C = [finalData.Education(:), finalData.Age(:)];
        adjStr = '(adj. Edu, Age)';
    else
        C = [];
        adjStr = '';
    end

    % Partial Spearman: rho, CI (bootstrap), p (permutation)
    stats = partial_spearman_perm_ci(X, Y, C, B, P);

    % ----- Global trend line (optional; not inference) -----
    valid_mask = isfinite(X) & isfinite(Y);
    gX = X(valid_mask); gY = Y(valid_mask);
    if numel(gX) >= 3
        [p_glob, S_glob] = polyfit(gX, gY, 1);
        x_lin = linspace(min(gX), max(gX), 100);
        [y_fit, delta] = polyval(p_glob, x_lin, S_glob);
        fill([x_lin, fliplr(x_lin)], [y_fit-delta, fliplr(y_fit+delta)], ...
            'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(x_lin, y_fit, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Overall Trend');
    end

    % ----- Group scatter -----
    for g = 1:3
        idx = (finalData.GroupNum == g) & isfinite(X) & isfinite(Y);
        if ~any(idx), continue; end
        scatter(X(idx), Y(idx), 80, colors(g,:), 'filled', 'MarkerFaceAlpha', 0.6, ...
            'DisplayName', groupNames{g});
    end

    % ----- Labels / subtitle -----
    if stats.p_perm < 1/(P+1)
        pStr = sprintf('p_{perm} < %.4g', 1/(P+1));
    elseif stats.p_perm < 0.001
        pStr = 'p_{perm} < .001';
    else
        pStr = sprintf('p_{perm} = %.4f', stats.p_perm);
    end

    subStr = sprintf('\\rho = %.3f, 95%% CI [%.2f, %.2f], %s %s, n=%d', ...
        stats.rho, stats.ci(1), stats.ci(2), pStr, adjStr, stats.n);

    title(metricLabels{m}, 'FontSize', 14, 'FontWeight', 'bold');
    subtitle(subStr, 'FontSize', 11, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);
    xlabel('Hemodynamic Latency (s)');
    ylabel(metricLabels{m});

    % Optional axis constraints
    if strcmpi(target, 'MoCA')
        ylim([-5, 32]); yticks(-5:5:30);
    elseif strcmpi(target, 'Education')
        ylim([0, 18]); yticks(0:3:18);
    end

    grid on; set(gca,'TickDir','out','Box','on','LineWidth',1.2);

    if m == 1
        legend('Location','best','Box','off','FontSize',10);
    end
end

%% ----------------- 5. Reviewer defense outputs (recommended) -----------------
% (A) Within-group correlations (no covariates) for Accuracy, MMSE, MoCA
fprintf('\n=== Within-group Spearman (no covariates) ===\n');
targets = {'MMSE','MoCA','Accuracy'};
for t = 1:numel(targets)
    Yname = targets{t};
    for g = 1:3
        idx = (finalData.GroupNum == g);
        [rho_g, p_g, n_g] = spearman_simple(finalData.Latency(idx), finalData.(Yname)(idx));
        fprintf('%s | %s: rho=%.3f, p=%.4f, n=%d\n', Yname, groupNames{g}, rho_g, p_g, n_g);
    end
end

% (B) Partial correlation controlling Group + Education + Age (pooled)
% This addresses: "Is correlation independent of diagnostic group?"
fprintf('\n=== Pooled partial Spearman controlling Group + Edu + Age ===\n');
Cfull = [finalData.Education(:), finalData.Age(:), dummyvar(categorical(finalData.GroupNum))];
Cfull = Cfull(:,1:end-1); % drop one dummy
stats_gc = partial_spearman_perm_ci(finalData.Latency, finalData.Accuracy, Cfull, B, P);
fprintf('Accuracy | control(Group,Edu,Age): rho=%.3f, p_perm=%.4f, CI=[%.3f, %.3f], n=%d\n', ...
    stats_gc.rho, stats_gc.p_perm, stats_gc.ci(1), stats_gc.ci(2), stats_gc.n);
fprintf('\n=== Does Latency predict global cognition beyond Accuracy? ===\n');

targets = {'MMSE','MoCA'};
B = 2000; 
P = 5000;

for t = 1:numel(targets)
    Yname = targets{t};
    
    X = finalData.Latency;
    Y = finalData.(Yname);
    
    % Control: Accuracy + Education + Age + Group
    Gdummy = dummyvar(categorical(finalData.GroupNum));
    Gdummy = Gdummy(:,1:end-1);  % drop one level
    
    C = [finalData.Accuracy, finalData.Education, finalData.Age, Gdummy];
    
    stats = partial_spearman_perm_ci(X, Y, C, B, P);
    
    fprintf('%s | control(Accuracy,Edu,Age,Group): rho=%.3f, p_perm=%.4f, CI=[%.3f, %.3f], n=%d\n', ...
        Yname, stats.rho, stats.p_perm, stats.ci(1), stats.ci(2), stats.n);
end
fprintf('\n=== Linearity check: Accuracy ~ Latency ===\n');

X = finalData.Latency;
Y = finalData.Accuracy;

ok = isfinite(X) & isfinite(Y);
X = X(ok); Y = Y(ok);

% Linear model
mdl1 = fitlm(X, Y);

% Quadratic model
mdl2 = fitlm(X, Y, 'quadratic');

fprintf('Linear R^2 = %.3f\n', mdl1.Rsquared.Ordinary);
fprintf('Quadratic R^2 = %.3f\n', mdl2.Rsquared.Ordinary);

anova_tbl = anova(mdl2,'summary');
disp('Model comparison (Linear vs Quadratic):');
disp(anova_tbl);
figure; 
plotResiduals(mdl1, 'fitted');
title('Residuals of Linear Model: Accuracy ~ Latency');

%% ============================================================
% FUNCTIONS
%% ============================================================

function stats = partial_spearman_perm_ci(x, y, covar, B, P)
% Partial Spearman via rank-residualization:
% 1) rank-transform x, y, covariates
% 2) residualize rank(x), rank(y) w.r.t rank(covariates) by OLS
% 3) rho = corr(res_x, res_y) (Pearson on residual ranks)
% CI via bootstrap resampling; p via permutation (two-sided, +1 correction)

    x = double(x(:)); y = double(y(:));

    if isempty(covar)
        ok = isfinite(x) & isfinite(y);
        C = [];
    else
        covar = double(covar);
        ok = isfinite(x) & isfinite(y) & all(isfinite(covar), 2);
        C = covar(ok, :);
    end

    x = x(ok); y = y(ok);
    n = numel(x);
    stats.n = n;

    if n < 6
        stats.rho = NaN; stats.ci = [NaN NaN]; stats.p_perm = NaN;
        return;
    end

    rx = tiedrank(x);
    ry = tiedrank(y);

    if isempty(C)
        Xcov = ones(n,1);
    else
        % Rank-transform each covariate column
        RC = zeros(n, size(C,2));
        for j = 1:size(C,2)
            RC(:,j) = tiedrank(C(:,j));
        end
        Xcov = [ones(n,1), RC];
    end

    % Residualize ranks
    bx = Xcov \ rx; rx_res = rx - Xcov*bx;
    by = Xcov \ ry; ry_res = ry - Xcov*by;

    rho_obs = corr(rx_res, ry_res, 'Type','Pearson');
    stats.rho = rho_obs;

    % ----- Bootstrap CI -----
    rng(0);
    bootrho = nan(B,1);
    for b = 1:B
        idx = randsample(n, n, true);
        rx_b = rx(idx); ry_b = ry(idx);
        Xb = Xcov(idx, :);

        bx_b = Xb \ rx_b; rxr_b = rx_b - Xb*bx_b;
        by_b = Xb \ ry_b; ryr_b = ry_b - Xb*by_b;
        bootrho(b) = corr(rxr_b, ryr_b, 'Type','Pearson');
    end
    stats.ci = prctile(bootrho, [2.5 97.5]);

    % ----- Permutation p-value (two-sided) -----
    rng(1);
    perm_rho = nan(P,1);
    for k = 1:P
        perm_idx = randperm(n);
        ry_p = ry(perm_idx);

        by_p = Xcov \ ry_p;
        ryr_p = ry_p - Xcov*by_p;

        perm_rho(k) = corr(rx_res, ryr_p, 'Type','Pearson');
    end

    % +1 correction to avoid p=0, two-sided
    stats.p_perm = (sum(abs(perm_rho) >= abs(rho_obs)) + 1) / (P + 1);
end

function [rho, p, n] = spearman_simple(x, y)
% Simple Spearman correlation (no covariates), with analytic p-value
    x = double(x(:)); y = double(y(:));
    ok = isfinite(x) & isfinite(y);
    x = x(ok); y = y(ok);
    n = numel(x);
    if n < 6
        rho = NaN; p = NaN; return;
    end
    [rho, p] = corr(x, y, 'Type','Spearman');
end