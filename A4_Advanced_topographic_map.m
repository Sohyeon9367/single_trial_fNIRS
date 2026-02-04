%% ============================================================
%  ADVANCED 3-PANEL STATISTICAL TOPOGRAPHIC MAPPING
%  - Conditions: Correct, Incorrect, Difference (C-I)
%  - Stats: Partial Spearman (Adj: Age, Education)
% ============================================================

infoPath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Raw data\subject_info.xlsx';
s_info = readtable(infoPath);
dataPath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed\Stroop_AllSubjects_NIRS_HRF_TTest_05.mat';
load(dataPath);

nSubjects = numel(allSubjects);
nCh = 48;
snrThreshold = 30;
beta_C_mat = nan(nSubjects, nCh);   % Correct
beta_I_mat = nan(nSubjects, nCh);   % Incorrect
beta_D_mat = nan(nSubjects, nCh);   % Difference (C - I)
group_vec = nan(nSubjects, 1);
covariates = nan(nSubjects, 2); % [Education, Age]

fprintf('Extract the dataset...\n');
for s = 1:nSubjects
    subj = allSubjects(s).results;
    subjID = subj.ID;
    
    mIdx = find(string(s_info.SubjectID) == string(subjID), 1);
    if isempty(mIdx), continue; end
    
    group_vec(s) = subj.Group;
    covariates(s, :) = [s_info.Education(mIdx), s_info.Age(mIdx)];
    
    c_idx = find(arrayfun(@(x) strcmpi(x.results.ACC, 'correct'), subj.Trials));
    i_idx = find(arrayfun(@(x) strcmpi(x.results.ACC, 'incorrect'), subj.Trials));
    
    for ch = 1:nCh
        if subj.SNR(ch) >= snrThreshold
            b_c = mean(arrayfun(@(x) x.results.beta_HbO(ch), subj.Trials(c_idx)), 'omitnan');
            b_i = mean(arrayfun(@(x) x.results.beta_HbO(ch), subj.Trials(i_idx)), 'omitnan');
            
            beta_C_mat(s, ch) = b_c;
            beta_I_mat(s, ch) = b_i;
            beta_D_mat(s, ch) = b_c - b_i;
        end
    end
end

%% --- 3. Partial Spearman ---
topoResults = struct('condition', {'Correct', 'Incorrect', 'Difference'}, ...
                     'rho', {zeros(nCh,1), zeros(nCh,1), zeros(nCh,1)}, ...
                     'p',   {ones(nCh,1),  ones(nCh,1),  ones(nCh,1)});
beta_mats = {beta_C_mat, beta_I_mat, beta_D_mat};

fprintf('Partial Spearman  (Age, Edu control)...\n');
for i = 1:3
    target_mat = beta_mats{i};
    for ch = 1:nCh
        valid_idx = ~isnan(target_mat(:, ch)) & all(~isnan(covariates), 2);
        if sum(valid_idx) > 10
            [r, p] = partialcorr(target_mat(valid_idx, ch), group_vec(valid_idx), ...
                                 covariates(valid_idx, :), 'Type', 'Spearman');
            topoResults(i).rho(ch) = r;
            topoResults(i).p(ch) = p;
        else
            topoResults(i).rho(ch) = 0; topoResults(i).p(ch) = 1;
        end
    end
end
%% --- 4. Visualization ---

chanMap = [0,0,1,2,3,17,18,19,20,33,34,35,0,0;
           4,5,6,7,8,21,22,23,24,36,37,38,39,40;
           9,10,11,12,13,25,26,27,28,41,42,43,44,45;
           0,0,14,15,16,29,30,31,32,46,47,48,0,0];
all_x = zeros(48,1); all_y = zeros(48,1);

for r = 1:4
    for c = 1:14
        chID = chanMap(r,c);
        if chID > 0, all_x(chID) = c * spacing; all_y(chID) = (5-r) * spacing; end
    end
end

pad = 3.0; 
xi_lin = linspace(min(all_x)-pad, max(all_x)+pad, 200);
yi_lin = linspace(min(all_y)-pad, max(all_y)+pad, 200);
[XI, YI] = meshgrid(xi_lin, yi_lin);

k_hull = convhull(all_x, all_y);
cx = mean(all_x); cy = mean(all_y);
scale_f = 1.25; 
ex_x = (all_x(k_hull) - cx) * scale_f + cx;
ex_y = (all_y(k_hull) - cy) * scale_f + cy;
mask = inpolygon(XI, YI, ex_x, ex_y);
hFig = figure('Color', 'w', 'Position', [50, 50, 1800, 300]);
tlo = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, 'Brain-Behavior Coupling: Partial Spearman Correlation Map', ...
      'FontSize', 20, 'FontWeight', 'bold');

commonClim = [-0.6, 0.6];
cmap = [linspace(0,1,128)', linspace(0,1,128)', ones(128,1); 
        ones(128,1), linspace(1,0,128)', linspace(1,0,128)'];

for i = 1:3
    ax = nexttile;
    rho_data = topoResults(i).rho;
    sig_idx = find(topoResults(i).p < 0.05); 
    
    ZI = griddata(all_x, all_y, rho_data, XI, YI, 'v4'); 
    ZI(~mask) = NaN; 
    p_surf = surf(XI, YI, ZI); 
    shading interp; view(2); hold on;
    if ~isempty(sig_idx)
        scatter3(all_x(sig_idx), all_y(sig_idx), ones(size(sig_idx))*50, ...
                 50, 'w', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    end
    
    colormap(ax, cmap); clim(commonClim);
    axis equal; axis off;
    t_sub = title(topoResults(i).condition, 'FontSize', 16, 'FontWeight', 'bold');
    t_sub.VerticalAlignment = 'bottom';
end
cb = colorbar;
cb.Layout.Tile = 'south';
cb.Orientation = 'horizontal';
cb.Label.String = 'Partial Spearman \rho (adjusted for Age and Education)';
cb.Label.FontSize = 14;
cb.Label.FontWeight = 'bold';
cb.Position = [0.3, 0.08, 0.4, 0.03]; 