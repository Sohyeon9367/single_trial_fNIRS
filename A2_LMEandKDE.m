%% ============================================================
%  INTEGRATED NEUROVASCULAR LATENCY & LMM ANALYSIS PIPELINE
%  Version: 4.0 (Enhanced QC, Covariates, and Data Persistence)
% ============================================================
clear; clc; close all;

%% ----------------- 1. Configuration & Loading -----------------
dataPath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed\Stroop_AllSubjects_NIRS_HRF_TTest_05.mat';
infoPath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Raw data\subject_info.xlsx';
savePath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed_Latency_Master.mat';

% Analysis Parameters
fs = 8.138;               
snrThreshold = 30;        
maxWindow = 20;   
groupNames = {'HC','MCI','AD'};
colors = [0, 0.447, 0.741; 0.85, 0.325, 0.098; 0.929, 0.694, 0.125];

%% ============================================================
% 2. INTEGRATED DATA EXTRACTION & LATENCY CALCULATION
% ============================================================

% Delete the file in savePath after change the setting
if exist(savePath, 'file')
    fprintf('✅ Preprocessed file found. Data loading...\n');
    load(savePath);
else
    fprintf('🚀 New preprocessing starts...\n');
    load(dataPath);
    s_info = readtable(infoPath); % load subject information

    % --- [initialize] QC structure ---
    qc.grand_total_channels = 0;   
    qc.total_p_fail = 0;           
    qc.rejected_p = zeros(1,3);    
    qc.low_snr = 0;                
    qc.out_window = 0;             
    qc.group_processed_ch = zeros(1,3); 
    qc.total_trials_grp = zeros(1,3);    
    qc.rejected_trials_grp = zeros(1,3); 

    masterData = table(); 
    row_idx = 1;
    match_count = 0; 

    % --- Subjects ---
    for s = 1:numel(allSubjects)
        subj = allSubjects(s).results;
        subjID = subj.ID;
        group = subj.Group; % 1:HC, 2:MCI, 3:AD
        subjSNR = subj.SNR; 

        matchIdx = find(string(s_info.SubjectID) == string(subjID), 1);
        if isempty(matchIdx)
            currAge = NaN; currEdu = NaN;
        else
            currAge = s_info.Age(matchIdx);
            currEdu = s_info.Education(matchIdx);
            match_count = match_count + 1;
        end
        for t = 1:numel(subj.Trials)
            tr = subj.Trials(t).results;
            qc.total_trials_grp(group) = qc.total_trials_grp(group) + 1;
            isInvalidACC = ~ismember(lower(char(tr.ACC)), {'correct', 'incorrect'});
            isTooShort   = isfield(tr, 'duration') && tr.duration <= 1.2131;
            if isInvalidACC || isTooShort
                qc.rejected_trials_grp(group) = qc.rejected_trials_grp(group) + 1;
                continue; 
            end
            p_vals = tr.p_HbO;
            for ch = 1:48
                qc.grand_total_channels = qc.grand_total_channels + 1;
                qc.group_processed_ch(group) = qc.group_processed_ch(group) + 1;
                if p_vals(ch) >= 0.05
                    qc.total_p_fail = qc.total_p_fail + 1;
                    qc.rejected_p(group) = qc.rejected_p(group) + 1;
                    continue; 
                end
                if subjSNR(ch) < snrThreshold
                    qc.low_snr = qc.low_snr + 1;
                    continue; 
                end
                
                % --- Calculate Latency(Time to Peak) ---
                ts = tr.HbO(:, ch);
                L = length(ts);
                
                % Sgolay Smoothing
                F = min(31, L); if mod(F, 2) == 0, F = F - 1; end
                if F > 3, ts_s = sgolayfilt(ts, 3, F); else, ts_s = movmean(ts, 3); end
                
                % Find peak (0.5s ~ maxWindow)
                winIdx = round(0.5 * fs) : min(round(maxWindow * fs), L);
                if isempty(winIdx), continue; end
                
                [~, localIdx] = max(ts_s(winIdx));
                lat = (winIdx(localIdx) - 1) / fs; 
                
                % [필터 4] 최종 유효성 검사 및 테이블 삽입
                if lat >= 0.5 && lat <= maxWindow
                    masterData.Latency(row_idx) = lat;
                    masterData.Group(row_idx) = group;
                    masterData.SubjectID{row_idx} = subjID;
                    masterData.ACC{row_idx} = char(tr.ACC);
                    masterData.Age(row_idx) = currAge;
                    masterData.Education(row_idx) = currEdu;
                    row_idx = row_idx + 1;
                else
                    qc.out_window = qc.out_window + 1;
                end
            end
        end
    end
    
    fprintf('✨ ID matching Success: total %d ,  %d Complete\n', numel(allSubjects), match_count);
    save(savePath, 'masterData', 'qc');
    fprintf('💾 File saved [%s]\n', savePath);
end

masterData.Group = categorical(masterData.Group, [1, 2, 3], groupNames);
masterData.ACC = categorical(masterData.ACC);
masterData.SubjectID = categorical(masterData.SubjectID);
%% ----------------- 3. Comprehensive QC Report -----------------
fprintf('\n================== GLOBAL QC REPORT (Channel-wise) ==================\n');
fprintf('Grand Total Channels Processed: %d\n', qc.grand_total_channels);
fprintf('1. P-value Masked (p >= 0.05): %6d (%.2f%%)\n', qc.total_p_fail, (qc.total_p_fail/qc.grand_total_channels)*100);
fprintf('2. Low SNR (<%d dB) Rejection: %6d (%.2f%%)\n', snrThreshold, qc.low_snr, (qc.low_snr/qc.grand_total_channels)*100);
fprintf('3. Window Out Rejection:        %6d (%.2f%%)\n', qc.out_window, (qc.out_window/qc.grand_total_channels)*100);
fprintf('---------------------------------------------------------------------\n');
fprintf(' [P-value Rejection Rate by Group]\n');
for g = 1:3
    fprintf(' %-3s Group: %.2f%% failed p-threshold\n', groupNames{g}, (qc.rejected_p(g)/qc.group_processed_ch(g))*100);
end
fprintf('Final Data points for LMM: %d\n', height(masterData));
fprintf('=====================================================================\n\n');

%% ----------------- 4. KDE Plot (Shared Legend) -----------------
figure('Color','w','Position', [100 100 1200 500]);
tlo = tiledlayout(1, 2, 'Padding', 'loose', 'TileSpacing', 'compact');
titles = {'Correct Trials: Latency', 'Incorrect Trials: Latency'};
acc_types = {'correct', 'incorrect'};
p_h = [];

for i = 1:2
    ax = nexttile(tlo); hold on;
    for g = 1:3
        data = masterData.Latency(masterData.Group == groupNames{g} & masterData.ACC == acc_types{i});
        if isempty(data), continue; end
        [f, xi] = ksdensity(data, 'Support', [0, maxWindow+0.5], 'Boundary', 'reflection');
        fill(xi, f, colors(g,:), 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        line_h = plot(xi, f, 'Color', colors(g,:), 'LineWidth', 2.5, 'DisplayName', groupNames{g});
        if i == 1, p_h = [p_h, line_h]; end
    end
    title(titles{i}, 'FontSize', 15); xlabel('Time to Peak (s)'); ylabel('Density');
    xlim([0 12]); ylim([0 0.8]); grid on; ax.GridAlpha = 0.1;
end
lgd = legend(p_h, groupNames); lgd.Layout.Tile = 'north'; 

lgd.Orientation = "horizontal"; lgd.Box = 'off';

%% ----------------- 5. LMM with Covariates (Age & Education) -----------------
fprintf('--- LMM: Latency ~ Group * ACC + Age + Education + (1 | SubjectID) ---\n');
% Remove row have no age and education
lmmTable = rmmissing(masterData, 'DataVariables', {'Age', 'Education'});

lme = fitlme(lmmTable, 'Latency ~ Group * ACC + Age + Education + (1 | SubjectID)');
disp(lme);

% ANOVA Table for Fixed Effects
fprintf('\nANOVA (Type III Test):\n');
disp(dataset2table(anova(lme)));
% (Q-Q Plot)
figure;
qqplot(lme.residuals); 
title('LMM Model Diagnostics: Residual Normality');

% (Fitted vs Residuals)
figure;
plot(lme.fitted, lme.residuals, 'o');
xlabel('Fitted Values'); ylabel('Residuals');
title('LMM Model Diagnostics: Homoscedasticity');
% log(Latency) Sensitivity
lmmTable.logLatency = log(lmmTable.Latency);

fprintf('\n--- LMM: log(Latency) Sensitivity Analysis ---\n');
lme_log = fitlme(lmmTable, 'logLatency ~ Group * ACC + Age + Education + (1 | SubjectID)');
disp(dataset2table(anova(lme_log)));
fprintf('\n[Fixed Effects Comparison]\n');
fprintf('%-25s | %-10s | %-10s\n', 'Predictor', 'Original p', 'Log-Trans p');
fprintf('------------------------------------------------------------\n');

% Compare (lme) and (lme_log)
terms = lme.CoefficientNames;
for i = 1:length(terms)
    p_orig = lme.Coefficients.pValue(i);
    p_log = lme_log.Coefficients.pValue(i);
    fprintf('%-25s | %-10.4e | %-10.4e\n', terms{i}, p_orig, p_log);
end

%% ============================================================
%  TABLE-BASED DESCRIPTIVE STATISTICS (masterData)
%  - Stats: Mean, CV, N (Grouped by Group & ACC)
% ============================================================
%% ============================================================
%  TABLE-BASED DESCRIPTIVE STATISTICS & ABSTRACT VALIDATION
% ============================================================

% 1. Group Summary
statsTable = groupsummary(masterData, {'Group', 'ACC'}, {'mean', 'std'}, 'Latency');

% 2. CV (Temporal Jitter) 계산
statsTable.CV = statsTable.std_Latency ./ statsTable.mean_Latency;

% 카테고리 비교를 위한 변환
statsTable.Group_Str = string(statsTable.Group);
statsTable.ACC_Str = string(statsTable.ACC);

% 출력 (수정됨: [] 사용)
fprintf(['\n', repmat('=', 1, 60), '\n']);
fprintf('   FINAL STATISTICAL SUMMARY (Validation for Abstract)\n');
fprintf([repmat('=', 1, 60), '\n']);

groups = {'HC', 'MCI', 'AD'};
conditions = {'correct', 'incorrect'};

valData = struct();

for c = 1:2
    cond = conditions{c};
    fprintf('\n[%s Trials]\n', upper(cond));
    fprintf('------------------------------------------------------------\n');
    for g = 1:3
        grp = groups{g};
        row = statsTable(statsTable.Group_Str == grp & statsTable.ACC_Str == cond, :);
        
        if ~isempty(row)
            mVal = row.mean_Latency;
            cvVal = row.CV;
            nVal = row.GroupCount;
            valData.(grp).(cond) = mVal;
            
            fprintf('%-3s: Mean = %5.2f s | CV = %4.2f (Jitter) | N = %s\n', ...
                grp, mVal, cvVal, f_comma(nVal));
        else
            valData.(grp).(cond) = NaN;
            fprintf('%-3s: No Data available\n', grp);
        end
    end
end

%% 3. 초록 핵심 수치 검증
fprintf(['\n', repmat('-', 1, 60), '\n']);
fprintf('   HYPOTHESIS TESTING (Relative to HC)\n');
fprintf([repmat('-', 1, 60), '\n']);

if isfield(valData, 'AD') && isfield(valData, 'HC')
    % AD vs HC 전체 지연
    ad_delay = valData.AD.correct - valData.HC.correct;
    fprintf('1. AD vs HC Delay (Correct):  %5.2f s (Abstract: ~1.48s)\n', ad_delay);
    
    % AD 내 지연 차이
    ad_acc_diff = valData.AD.incorrect - valData.AD.correct;
    hc_acc_diff = valData.HC.incorrect - valData.HC.correct;
    
    fprintf('2. AD Incorrect vs Correct:   %5.2f s (Abstract: ~2.88s extra)\n', ad_acc_diff);
    fprintf('3. HC Incorrect vs Correct:   %5.2f s (Abstract: "Stable")\n', hc_acc_diff);
    
    if ad_acc_diff > hc_acc_diff
        fprintf('   => Interaction Trend: [MATCH] AD shows larger delay on errors.\n');
    else
        fprintf('   => Interaction Trend: [MISMATCH] Check data labeling.\n');
    end
end
fprintf([repmat('=', 1, 60), '\n']);

function str = f_comma(n), str = regexprep(num2str(n), '(?<=\d)(?=(\d{3})+(?!\d))', ','); end