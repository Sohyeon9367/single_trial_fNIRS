%% ============================================================
%  SNR Filtered Topographic Beta Mapping (Non-biased)
%  - Rejection: Only by SNR (No P-value masking for beta maps)
%  - Analysis: Includes all signal-stable channels for ANCOVA
% ============================================================
clear; clc; close all;

%% ----------------- 1. Load Data & Config -----------------
dataPath = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed\Stroop_AllSubjects_NIRS_HRF_TTest_05.mat';
load(dataPath);
nSubjects = numel(allSubjects);
snrThreshold = 30; 
pThreshold = 0.05; 
Stats = struct();  

total_ch_processed = 0;
total_ch_rejected_snr = 0;
total_ch_rejected_p = 0; % No rejection for statistics
% (1:HC, 2:MCI, 3:AD)
group_names = {'HC', 'MCI', 'AD'};
group_ch_processed = zeros(1, 3);
group_ch_rejected_p = zeros(1, 3);

%% ----------------- 2. Compute Beta Maps (SNR-Only Rejection) -----------------
fprintf('Processing subject beta maps (SNR-Only filtering for Group Analysis)...\n');

for s = 1:nSubjects
    subj_res = allSubjects(s).results;
    trials = subj_res.Trials;
    subjSNR = subj_res.SNR;
    g = subj_res.Group; 
    
    p_corr  = subj_res.p_HbO_corr;
    p_incor = subj_res.p_HbO_incor;
    
    nCh = 48; 
    beta_corr_valid = nan(nCh,1);
    beta_incor_valid = nan(nCh,1);
    
    for ch = 1:nCh
        total_ch_processed = total_ch_processed + 1;
        group_ch_processed(g) = group_ch_processed(g) + 1;
        if subjSNR(ch) < snrThreshold
            total_ch_rejected_snr = total_ch_rejected_snr + 1;
            continue; 
        end
        % Correct
        c_indices = find(arrayfun(@(x) strcmpi(x.results.ACC, 'correct'), trials));
        if ~isempty(c_indices)
            c_betas = arrayfun(@(x) x.results.beta_HbO(ch), trials(c_indices));
            beta_corr_valid(ch) = mean(c_betas, 'omitnan');
        end
        
        % Incorrect
        i_indices = find(arrayfun(@(x) strcmpi(x.results.ACC, 'incorrect'), trials));
        if ~isempty(i_indices)
            i_betas = arrayfun(@(x) x.results.beta_HbO(ch), trials(i_indices));
            beta_incor_valid(ch) = mean(i_betas, 'omitnan');
        end
        if p_corr(ch) >= pThreshold
            total_ch_rejected_p = total_ch_rejected_p + 0.5;
            group_ch_rejected_p(g) = group_ch_rejected_p(g) + 0.5;
        end
        if p_incor(ch) >= pThreshold
            total_ch_rejected_p = total_ch_rejected_p + 0.5;
            group_ch_rejected_p(g) = group_ch_rejected_p(g) + 0.5;
        end
    end
    
    Stats(s).betaCorr = beta_corr_valid;    
    Stats(s).betaIncor = beta_incor_valid;  
    Stats(s).betaDiff = beta_corr_valid - beta_incor_valid; 
    Stats(s).Group = g; 
end

% --- QC---
line_sep = repmat('=', 1, 60);
fprintf('\n%s\n', line_sep);
fprintf(' FINAL DATA QC REPORT\n');
fprintf('%s\n', line_sep);
fprintf(' SNR Rejected (Actual): %.2f%%\n', (total_ch_rejected_snr/total_ch_processed)*100);
fprintf(' P-value "Non-Significant" Rate (Informative): %.2f%%\n', (total_ch_rejected_p/total_ch_processed)*100);
fprintf('%s\n', repmat('-', 1, 60));
for g = 1:3
    if group_ch_processed(g) > 0
        p_rate = (group_ch_rejected_p(g) / group_ch_processed(g)) * 100;
        fprintf(' [%s Group] Non-Significant Channels: %.2f%%\n', group_names{g}, p_rate);
    end
end
fprintf('%s\n\n', line_sep);

%% ----------------- 3. Visualization Setup -----------------
outputFolder = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Plots_Beta_Final';
if ~exist(outputFolder, 'dir'); mkdir(outputFolder); end
chanMap = [0,0,1,2,3,17,18,19,20,33,34,35,0,0;
           4,5,6,7,8,21,22,23,24,36,37,38,39,40;
           9,10,11,12,13,25,26,27,28,41,42,43,44,45;
           0,0,14,15,16,29,30,31,32,46,47,48,0,0];

%% ----------------- 4. GROUP AVERAGED MAPS -----------------
all_maps = cell(1,3);
global_max = 0;
for g = 1:3
    groupIndices = find([Stats.Group] == g);
    if isempty(groupIndices), continue; end
    
    all_maps{g}.Correct = mean([Stats(groupIndices).betaCorr], 2, 'omitnan');
    all_maps{g}.Incorrect = mean([Stats(groupIndices).betaIncor], 2, 'omitnan');
    all_maps{g}.Difference = mean([Stats(groupIndices).betaDiff], 2, 'omitnan');
    
    max_local = max(abs([all_maps{g}.Correct(:); all_maps{g}.Incorrect(:); all_maps{g}.Difference(:)]), [], 'omitnan');
    if max_local > global_max, global_max = max_local; end
end
global_limit = ceil(global_max * 10) / 10;

%% ----------------- 5. FINAL 3x3 INTEGRATED FIGURE -----------------
dlPFC_channels = [1,2,3,5,6,11,17,18,19,20,33,34,35,38,39,43]; 
mapTypes = {'Correct', 'Incorrect', 'Difference'};
colTitles = {'Correct Trials', 'Incorrect Trials', '\Delta (Correct - Incorrect)'};
rowTitles = {'HC', 'MCI', 'AD'};

all_x = zeros(48,1); all_y = zeros(48,1);
spacing = 3.0; 
y_coords_fixed = [4, 3, 2, 1] * spacing; 
for r = 1:4
    for c = 1:14
        chID = chanMap(r,c);
        if chID > 0
            all_x(chID) = c * spacing; all_y(chID) = y_coords_fixed(r);
        end
    end
end

[xi, yi] = meshgrid(min(all_x)-8:0.1:max(all_x)+8, min(all_y)-8:0.1:max(all_y)+8);
k_full = convhull(all_x, all_y);
cx = mean(all_x); cy = mean(all_y);
mask_full = inpolygon(xi, yi, (all_x(k_full)-cx)*1.15+cx, (all_y(k_full)-cy)*1.50+cy);

hFig = figure('Units', 'pixels', 'Position', [50, 50, 1200, 1050], 'Color', 'w');
tlo = tiledlayout(3, 3, 'Padding', 'compact', 'TileSpacing', 'tight'); 
title(tlo, {'Group-wise Cortical Activation Maps', ''}, 'FontSize', 24, 'FontWeight', 'bold');

for g = 1:3 
    for i = 1:3 
        ax = nexttile(tlo); hold on;
        if isempty(all_maps{g}), axis off; continue; end
        
        zi = griddata(all_x, all_y, all_maps{g}.(mapTypes{i}), xi, yi, 'v4');
        zi(~mask_full) = NaN;
        
        surf(xi, yi, zi); shading interp; view(0, 90);
        colormap(ax, 'jet'); clim([-global_limit global_limit]);
        
        scatter3(all_x(dlPFC_channels), all_y(dlPFC_channels), ones(length(dlPFC_channels),1)*10, ...
                 40, 'k', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1.2);
        
        axis equal; axis tight; axis off;
        if g == 1, title(colTitles{i}, 'FontSize', 16, 'FontWeight', 'bold'); end
        if i == 1
            text(min(all_x)-6, mean(all_y), rowTitles{g}, 'FontSize', 18, 'FontWeight', 'bold', ...
                 'Rotation', 90, 'HorizontalAlignment', 'center');
        end
    end
end

cb = colorbar; cb.Layout.Tile = 'south';
cb.Label.String = 'Standardized Beta Weight';
cb.Label.FontSize = 14; cb.Label.FontWeight = 'bold';

export_path = fullfile(outputFolder, 'Figure_Final_SNR_Only_Filtering.png');
exportgraphics(tlo, export_path, 'Resolution', 300);
fprintf('Image saved: %s\n', export_path);

