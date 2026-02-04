clc;
clear;
close all;

%% ==========================
%  Configuration
% ==========================
baseDir = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Raw data';
outputBaseDir = 'D:\1_MPI_CBS\0_Research\0_MCI_classification\Processed';
stroopInfoDir = fullfile(baseDir, 'stroop_data');
infoFile = fullfile(baseDir, 'subject_info.xlsx');
logFile = fullfile(outputBaseDir, 'processing_log.xlsx');

if ~exist(outputBaseDir, 'dir')
    mkdir(outputBaseDir);
end

subjectTable = readtable(infoFile);
subjectFolders = dir(baseDir);
subjectFolders = subjectFolders([subjectFolders.isdir] & ~startsWith({subjectFolders.name}, '.'));

logEntries = {};

fprintf('\n=== NIRS MBLL Batch Processing Started ===\n');

for s = 1:length(subjectFolders)
    folderName = subjectFolders(s).name;
    parts = split(folderName, ' ');
    subjPrefix = strtrim(parts{1});
    subjDir = fullfile(baseDir, folderName);

    idx = strcmpi(subjectTable.SubjectID, subjPrefix);
    if ~any(idx)
        fprintf('⚠️ Skipping %s — no matching SubjectID found in Excel.\n', folderName);
        logEntries(end+1,:) = {subjPrefix, folderName, 'N/A', 'Skipped (no subject info)', datestr(now)};
        continue;
    end

    age = subjectTable.Age(idx);
    subjID = subjPrefix;
    outDir = fullfile(outputBaseDir, subjID);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    fprintf('\n=== Subject: %s | Folder: %s ===\n', subjID, folderName);

    csvFiles = dir(fullfile(subjDir, '*_Raw.csv'));

    %% ---- Stroop Task ----
    stroopFile = find_csv(csvFiles, 'stroop');
    if ~isempty(stroopFile)
        try
            blockInfoFile = find_stroop_info(stroopInfoDir, subjID);
            fprintf('→ Processing Stroop: %s\n', stroopFile);
            process_nirs_with_trials(stroopFile, age, subjID, 'Stroop', blockInfoFile, outDir);
            logEntries(end+1,:) = {subjID, folderName, 'Stroop', 'Success', datestr(now)};
        catch ME
            fprintf('❌ Error in Stroop (%s): %s\n', subjID, ME.message);
            logEntries(end+1,:) = {subjID, folderName, 'Stroop', ['Error: ' ME.message], datestr(now)};
        end
    else
        logEntries(end+1,:) = {subjID, folderName, 'Stroop', 'No file found', datestr(now)};
    end

    %% ---- TwoBack Task ----
    twobackFile = find_csv(csvFiles, 'back');
    if ~isempty(twobackFile)
        try
            fprintf('→ Processing TwoBack: %s\n', twobackFile);
            process_nirs_with_trials(twobackFile, age, subjID, 'TwoBack', '', outDir);
            logEntries(end+1,:) = {subjID, folderName, 'TwoBack', 'Success', datestr(now)};
        catch ME
            fprintf('❌ Error in TwoBack (%s): %s\n', subjID, ME.message);
            logEntries(end+1,:) = {subjID, folderName, 'TwoBack', ['Error: ' ME.message], datestr(now)};
        end
    else
        logEntries(end+1,:) = {subjID, folderName, 'TwoBack', 'No file found', datestr(now)};
    end
end

fprintf('\n=== ✅ Batch Processing Complete ===\n');

%% Save log
logTable = cell2table(logEntries, ...
    'VariableNames', {'SubjectID','Folder','Task','Status','Timestamp'});
writetable(logTable, logFile);
fprintf('📄 Log file saved at: %s\n', logFile);


%% --- Helper functions ---
function fpath = find_csv(csvFiles, keyword)
    fpath = '';
    for i = 1:length(csvFiles)
        if contains(lower(csvFiles(i).name), keyword)
            fpath = fullfile(csvFiles(i).folder, csvFiles(i).name);
            return;
        end
    end
end

function blockInfoPath = find_stroop_info(stroopInfoDir, subjID)
    numPart = regexp(subjID, '\d+', 'match');
    if isempty(numPart)
        blockInfoPath = '';
        return;
    end
    subjNum = sprintf('%02d', str2double(numPart{1}));
    files = dir(fullfile(stroopInfoDir, sprintf('*subj%s_session2*.csv', subjNum)));
    if ~isempty(files)
        blockInfoPath = fullfile(files(1).folder, files(1).name);
    else
        warning('⚠️ No Stroop block info found for %s', subjID);
        blockInfoPath = '';
    end
end
