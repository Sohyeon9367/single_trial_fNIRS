function build_stroop_group_structure()
% BUILD_STROOP_GROUP_STRUCTURE
% Aggregates ddm_*.csv behavior and processed *_Stroop_MBLL.mat NIRS data
% into one combined .mat file with HRF-convolved predictors and trial-level GLM.
%
% Saves:
%   D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed\Stroop_AllSubjects_NIRS_HRF_TTest.mat

%% ----------------- Configuration -----------------
rawBase = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Raw data';
stroopCsvDir = fullfile(rawBase, 'stroop_data');
subjectInfoFile = fullfile(rawBase, 'subject_info.xlsx');
processedDir = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed';
outFile = fullfile(processedDir, 'Stroop_AllSubjects_NIRS_HRF_TTest_05.mat');

%% ----------------- Load behavior CSVs -----------------
csvFiles = dir(fullfile(stroopCsvDir, 'ddm*.csv'));
fprintf('Found %d ddm CSV files.\n', numel(csvFiles));

ddm = struct();
for i = 1:numel(csvFiles)
    fn = fullfile(csvFiles(i).folder, csvFiles(i).name);
    T = readtable(fn, 'PreserveVariableNames', true);

    base = erase(csvFiles(i).name, '.csv');
    tok = regexp(base, 'subj(\d+)', 'tokens', 'once');

    if isempty(tok)
        key = matlab.lang.makeValidName(base);
    else
        subjNN = sprintf('%02d', str2double(tok{1}));
        key = ['subj' subjNN];
    end

    ddm.(key).table = T;
    ddm.(key).fname = csvFiles(i).name;
end

%% ----------------- Load subject_info.csv -----------------
subjectInfo = readtable(subjectInfoFile, 'PreserveVariableNames', true);

%% ----------------- Find all Stroop MBLL files -----------------
mbllFiles = dir(fullfile(processedDir, '**', '*_Stroop_MBLL_05.mat'));
fprintf('Found %d *_Stroop_MBLL_05.mat files.\n', numel(mbllFiles));

%% ----------------- Prepare output -----------------
allSubjects = struct([]);
subjCount = 0;

%% ==========================================================
%                 SUBJECT LOOP
% ==========================================================
for f = 1:numel(mbllFiles)
    try
        mbllPath = fullfile(mbllFiles(f).folder, mbllFiles(f).name);
        % Extract subject folder name (A1, B36, ...)
        [subjFolderPath, ~] = fileparts(mbllPath);
        [~, subjFld] = fileparts(subjFolderPath);

        % --- ADD THIS EXCLUSION CODE HERE ---
        if strcmpi(subjFld, 'B55')
            fprintf('\n⏩ Skipping %s (Exclusion List)\n', subjFld);
            continue; 
        end
        % ------------------------------------

        fprintf('\n▶ Processing %s\n', mbllFiles(f).name);

        % Convert A1→subj01, B54→subj54
        numPart = regexp(subjFld, '\d+', 'match', 'once');
        subjNN = sprintf('%02d', str2double(numPart));
        subjKey = ['subj' subjNN];

        % Get group
        idxInfo = find(strcmpi(subjectInfo.SubjectID, subjFld),1);
        if ~isempty(idxInfo)
            grpVal = subjectInfo.Group(idxInfo);
        else
            grpVal = NaN;
        end

        % Load MBLL data
        S = load(mbllPath);
        nirs = S.nirs_data;
        fs = nirs.fs;
        t = nirs.t(:);
        marker = nirs.marker(:);
        SNR = nirs.snr_all;
        HbO = fillmissing(nirs.hbo, 'nearest');
        HbR = fillmissing(nirs.hbr, 'nearest');

        %% --- HRF ---
        HRF = spm_hrf(1/fs);
        HRF = HRF(:);

        %% --- Behavior table ---
        Beh = [];
        if isfield(ddm, subjKey)
            Beh = ddm.(subjKey).table;
        end

        behUsed = [];

        %% --- Find blocks (marker==1) ---
        blockStarts = find(marker == 1);
        if isempty(blockStarts)
            warning('No block start markers for %s', subjFld);
            continue;
        end

        Ssubj = initialize_subject_struct(subjFld, subjKey, grpVal, mbllPath, Beh, SNR);

        %% ==========================================================
        %               TRIAL EXTRACTION
        % ==========================================================
        for bi = 1:numel(blockStarts)

            blkStart = blockStarts(bi);
            if bi == 3
                [blkEnd,~] = size(HbO);
            else
                blkEnd = (bi < numel(blockStarts)) * (blockStarts(bi+1)-1) + ...
                         (bi == numel(blockStarts)) * numel(marker);
            end
            blockIdx = blkStart:blkEnd;
            blockMark = marker(blockIdx);

            % marker > 10 AND NOT = 2 (block end)
            trialRel = find(blockMark > 10 & blockMark ~= 2);

            trialStartAbs = blkStart;

            for ti = 1:(numel(trialRel)-1)

                trAbsIdx = blockIdx(trialRel(ti));
                idxRange = trialStartAbs:trAbsIdx;
                tTrial = t(idxRange) - t(idxRange(1));

                % Decode marker
                mVal = marker(trAbsIdx);
                code = floor(mVal/10);     % 1 unknown, 2 correct, 3 incorrect
                blkNum = mod(mVal,10);
                if blkNum==0, blkNum = bi; end

                ACC = "unknown";
                if code==2, ACC="correct"; end
                if code==3, ACC="incorrect"; end

                %% ---- Match behavior row ----
                behRow = match_behavior_row(Beh, behUsed, blkNum);
                if ~isempty(behRow)
                    behUsed(end+1) = behRow;
                end

                %% ---- Read onset/RT from behavior ----
                [onset_s, rt_s] = extract_onset_rt(Beh, behRow);

                %% ---- Boxcar ----
                boxcar = zeros(numel(tTrial),1);
                if onset_s > 0
                    boxcar(tTrial < onset_s) = 1;
                end

                %% ---- HRF predictor ----
                pred = conv(boxcar, HRF);
                pred = pred(1:numel(boxcar));
                if std(pred) > 0
                    pred = (pred - mean(pred)) / std(pred);
                end
                pred = pred-pred(1);

                %% ---- NIRS trial data ----
                HbO_tr = HbO(idxRange,:);
                HbR_tr = HbR(idxRange,:);
                %% ---- 1) Normalize each channel (0~1), 2) Make first value = 0 ----
                %HbO_tr = normalize_trial(HbO_tr);
                %HbR_tr = normalize_trial(HbR_tr);
                HbO_tr = HbO_tr - HbO_tr(1,:);
                HbR_tr = HbR_tr - HbR_tr(1,:);

                %% ---- Channel-wise GLM ----
                [betaO, tO, pO] = glm_channelWise(HbO_tr, pred);
                [betaR, tR, pR] = glm_channelWise(HbR_tr, pred);

                %% ---- Save trial ----
                Ssubj.Trials(end+1).results = create_trial_struct( ...
                    blkNum, ACC, onset_s, rt_s, ...
                    HbO_tr, HbR_tr, tTrial, boxcar, HRF, pred, ...
                    betaO, tO, pO, betaR, tR, pR );

                trialStartAbs = trAbsIdx + 1;
            end
        end
       %% ==========================================================
        %       SUBJECT-LEVEL STATISTICS (Across-trial T-test)
        %  Objective: Each Correct & Incorrect Statistical Significance
        % ==========================================================
        nCh = 48; 
        % initial value
        Ssubj.p_HbO_corr  = ones(nCh, 1);  Ssubj.t_HbO_corr  = zeros(nCh, 1);
        Ssubj.p_HbO_incor = ones(nCh, 1);  Ssubj.t_HbO_incor = zeros(nCh, 1);
        
        % Extract trial index
        correctIdx   = find(arrayfun(@(x) strcmpi(x.results.ACC, 'correct'),   Ssubj.Trials));
        incorrectIdx = find(arrayfun(@(x) strcmpi(x.results.ACC, 'incorrect'), Ssubj.Trials));
        
        for ch = 1:nCh
            % --- A. Correct Trials ---
            if ~isempty(correctIdx)
                trial_betas = arrayfun(@(x) x.results.beta_HbO(ch), Ssubj.Trials(correctIdx));
                valid_betas = trial_betas(~isnan(trial_betas));
                if numel(valid_betas) > 1
                    [~, p, ~, stats] = ttest(valid_betas);
                    Ssubj.p_HbO_corr(ch) = p;
                    Ssubj.t_HbO_corr(ch) = stats.tstat;
                end
            end
            
            % --- B. Incorrect Trials ---
            if ~isempty(incorrectIdx)
                trial_betas = arrayfun(@(x) x.results.beta_HbO(ch), Ssubj.Trials(incorrectIdx));
                valid_betas = trial_betas(~isnan(trial_betas));
                if numel(valid_betas) > 1
                    [~, p, ~, stats] = ttest(valid_betas);
                    Ssubj.p_HbO_incor(ch) = p;
                    Ssubj.t_HbO_incor(ch) = stats.tstat;
                end
            end
        end

        %% ---- Add to output ----
        subjCount = subjCount + 1;
        allSubjects(subjCount).results = Ssubj;

    catch ME
        fprintf('❌ Error processing %s: %s\n', mbllFiles(f).name, ME.message);
    end
end


%% Save
fprintf('\nSaving combined file: %s\n', outFile);
save(outFile, 'allSubjects', '-v7.3');
fprintf('Done. Subjects saved: %d\n', numel(allSubjects));

end

%% ==========================================================
%               Helper Functions
%% ==========================================================

function S = initialize_subject_struct(ID, subjKey, grp, mbllPath, Beh, SNR)
S = struct();
S.ID = ID;
S.subjKey = subjKey;
S.Group = grp;
S.SourceFile = mbllPath;
S.BehaviorFile = "";
if ~isempty(Beh), S.BehaviorFile = "Found"; end
S.SNR = SNR;    
S.Trials = struct([]);   
end

function behRow = match_behavior_row(Beh, behUsed, blkNum)
behRow = [];
if isempty(Beh), return; end

vn = lower(Beh.Properties.VariableNames);
blkCol = find(contains(vn, 'block'),1);

if ~isempty(blkCol)
    cand = find(Beh{:,blkCol} == blkNum);
    cand = setdiff(cand, behUsed);
    if ~isempty(cand)
        behRow = cand(1);
        return;
    end
end

% fallback: first unused row
allR = 1:height(Beh);
unused = setdiff(allR, behUsed);
if ~isempty(unused)
    behRow = unused(1);
end
end

function [on_s, rt_s] = extract_onset_rt(Beh, behRow)
on_s = 0; rt_s = 0;
if isempty(Beh) || isempty(behRow), return; end

vn = lower(Beh.Properties.VariableNames);
onCol = find(contains(vn,'onset') & contains(vn,'delay'),1);
rtCol = find(contains(vn,'trial_duration'),1);

if ~isempty(onCol)
    on = Beh{behRow, onCol};
    on = on+60; %60=stiumuls duration
    if iscell(on), on = str2double(on); end
    on_s = on/1000;
end

if ~isempty(rtCol)
    rt = Beh{behRow, rtCol};
    if iscell(rt), rt = str2double(rt); end
    rt_s = rt/1000;
end
end

function Hn = normalize_trial(H)
    Hn = nan(size(H));
    for ch = 1:size(H,2)
        x = H(:,ch);

        % If channel is all NaN or constant → return zeros
        if all(isnan(x)) || range(x,'omitnan') == 0
            Hn(:,ch) = zeros(size(x));
            continue;
        end

        % Normalize to 0–1 range
        xmin = min(x, [], 'omitnan');
        xmax = max(x, [], 'omitnan');
        xn = (x - xmin) / (xmax - xmin);

        % Make first value = 0 by subtracting xn(1)
        xn = xn - xn(1);

        Hn(:,ch) = xn;
    end
end

function [beta, tval, pval] = glm_channelWise(H, pred)

nCh = size(H,2);

beta = nan(nCh,1);
tval = nan(nCh,1);
pval = nan(nCh,1);

% Standardize predictor
predZ = zscore(pred);
X = [predZ ones(size(predZ))];

for ch = 1:nCh
    y = H(:,ch);

    if all(isnan(y)) || std(y,'omitnan') == 0
        continue;
    end

    valid = ~isnan(y) & ~any(isnan(X),2);
    yv = zscore(y(valid));         % also standardize y

    % Robust regression (much better for NIRS)
    [b, stats] = robustfit(predZ(valid), yv, 'bisquare', 50);

    beta(ch) = b(2);               % slope
    tval(ch) = stats.t(2);
    pval(ch) = stats.p(2);
end
end


function T = create_trial_struct(block, ACC, onset, rt, HbO, HbR, t, boxcar, HRF, pred, ...
                                 bO, tO, pO, bR, tR, pR)

T = struct();
T.Block = block;
T.ACC = char(ACC);
T.OnsetDelay = onset;
T.duration = rt;
T.HbO = HbO;
T.HbR = HbR;
T.t = t;
T.Boxcar = boxcar;
T.HRF = HRF;
T.Pred = pred;

T.beta_HbO = bO;
T.t_HbO = tO;
T.p_HbO = pO;

T.beta_HbR = bR;
T.t_HbR = tR;
T.p_HbR = pR;
end

function hrf = spm_hrf(dt)
p=[6 16 1 1 6 0 32];
u = 0:dt:p(7);
hrf = gampdf(u,p(1)/p(3),dt/p(3)) - gampdf(u,p(2)/p(4),dt/p(4))/p(5);
hrf = hrf'/max(hrf);
end
