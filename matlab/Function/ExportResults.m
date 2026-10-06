function Files = ExportResults(Result, Info, OutputDir)
%EXPORTRESULTS Write the results of RunNetPRS to CSV tables, JSON and MAT.
%
%   Files = ExportResults(Result, Info)              Info.Config.OutputDir
%   Files = ExportResults(Result, Info, OutputDir)
%
%   Files written (the Python version writes the same CSV tables and JSON)
%     auc_summary.csv     per level and model: mean/SD of the test AUC over the
%                         cross-validated fits, ensemble AUC, mean theta and mu,
%                         median selected epoch; wPRS baseline
%     cv_models.csv       every cross-validated fit: iteration, folds, selected
%                         epoch, train/validation/test AUC, theta, mu
%     networks.csv        Table III statistics of W_P and W_G for every level
%     snps.csv            every SNP: QC statistics, GWAS result, highest level
%     interpretation.csv  final model per level and SNP: beta, effective weight
%                         Omega, P_X, P_Z, SHAP summaries and rank
%     final_model.csv     final model per level: epochs, theta, mu, validation
%                         AUC, P_X vs P_Z summary, SHAP base value
%     run_info.json       configuration, parameters, data summary, software
%     NetPRS_Result.mat   Result and Info structs (MATLAB only)
%   Numbers are written with '%.10g'; NaN as an empty field.
%
%   Output: Files, cellstr of the written paths.

if nargin < 3 || isempty(OutputDir)
    OutputDir = Info.Config.OutputDir;
end
if exist(OutputDir, 'dir') ~= 7
    [ok, msg] = mkdir(OutputDir);
    if ~ok
        error('NetPRS:ExportResults:Folder', 'Cannot create %s (%s).', OutputDir, msg);
    end
end
Files = {};

%% auc_summary.csv and cv_models.csv
S = EmptyColumns({'level', 'threshold', 'num_snp', 'model', 'effects', 'auc_mean', 'auc_sd', ...
    'ensemble_auc', 'num_models', 'median_best_epoch', 'theta_I', 'theta_P', 'theta_G', 'mu_P', 'mu_G'});
M = EmptyColumns({'level', 'model', 'effects', 'model_index', 'iteration', 'valid_fold', 'test_fold', ...
    'best_epoch', 'auc_train', 'auc_valid', 'auc_test', 'theta_I', 'theta_P', 'theta_G', 'mu_P', 'mu_G'});
for r = 1:numel(Result)
    R = Result(r);
    models = ModelList(R);
    for a = 1:numel(models)
        Sm = models(a).Summary;
        [th, mu] = EffectColumns(Sm.EffectCode, Sm.Theta, Sm.Mu);
        S = AddRow(S, {R.Level, R.Threshold, numel(R.SNP), models(a).Name, Sm.EffectCode, Sm.AUCMean, ...
            Sm.AUCSD, Sm.EnsembleAUC, Sm.NumModel, Sm.MedianBestEpoch, th(1), th(2), th(3), mu(1), mu(2)});
        for k = 1:Sm.NumModel
            [th, mu] = EffectColumns(Sm.EffectCode, Sm.ThetaModel(:, k), Sm.MuModel(:, k));
            M = AddRow(M, {R.Level, models(a).Name, Sm.EffectCode, k, Sm.Iteration(k), Sm.FoldValid(k), ...
                Sm.FoldTest(k), Sm.BestEpoch(k), Sm.AUCTrain(k), Sm.AUCValid(k), Sm.AUCTest(k), ...
                th(1), th(2), th(3), mu(1), mu(2)});
        end
    end
    if Info.Config.RunBaseline
        S = AddRow(S, {R.Level, R.Threshold, numel(R.SNP), 'wPRS', '', R.wPRS.AUC, NaN, NaN, NaN, NaN, ...
            NaN, NaN, NaN, NaN, NaN});
    end
end
Files{end + 1} = WriteTable(fullfile(OutputDir, 'auc_summary.csv'), S);
Files{end + 1} = WriteTable(fullfile(OutputDir, 'cv_models.csv'), M);

%% networks.csv
N = EmptyColumns({'level', 'num_snp', 'edges_P', 'avg_weight_P', 'density_P', 'edges_G', 'avg_weight_G', ...
    'density_G', 'common_edges', 'corr_common', 'ks_p_common'});
for r = 1:numel(Result)
    T = Result(r).NetworkSummary;
    N = AddRow(N, {Result(r).Level, T.NumNode, T.Phe.NumEdge, T.Phe.AvgEdge, T.Phe.Density, T.Gen.NumEdge, ...
        T.Gen.AvgEdge, T.Gen.Density, T.NumCommon, T.Corr, T.KSP});
end
Files{end + 1} = WriteTable(fullfile(OutputDir, 'networks.csv'), N);

%% snps.csv (all SNPs of the data)
numAll = numel(Info.AllSNP);
qcCols = {'CallRate', 'MAF', 'HWEP'};
qcVal = nan(numAll, 3);
for k = 1:3
    if isfield(Info.QC, qcCols{k})
        qcVal(:, k) = Info.QC.(qcCols{k});
    end
end
kept = find(Info.KeepQC);
gw = nan(numAll, 3);
gw(kept, :) = [Info.GWAS.Beta(:), SEOrNaN(Info.GWAS, numel(kept)), Info.GWAS.P(:)];
level = zeros(numAll, 1);
for l = 1:numel(Info.LevelIdx)
    level(kept(Info.LevelIdx{l})) = l;
end
P = struct('Name', {{'snp', 'qc_call_rate', 'qc_maf', 'qc_hwe_p', 'qc_pass', 'gwas_beta', 'gwas_se', ...
    'gwas_p', 'gwas_neglog10p', 'level'}}, 'Data', {{Info.AllSNP(:), qcVal(:, 1), qcVal(:, 2), qcVal(:, 3), ...
    double(Info.KeepQC(:)), gw(:, 1), gw(:, 2), gw(:, 3), -log10(gw(:, 3)), level}});
Files{end + 1} = WriteTable(fullfile(OutputDir, 'snps.csv'), P);

%% interpretation.csv and final_model.csv
I = EmptyColumns({'level', 'snp', 'beta', 'effective_weight', 'p_x', 'p_z', 'mean_abs_shap', ...
    'shap_case', 'shap_control', 'shap_rank'});
F = EmptyColumns({'level', 'effects', 'max_epoch', 'theta_I', 'theta_P', 'theta_G', 'mu_P', 'mu_G', ...
    'auc_validation', 'num_z_more_significant', 'frac_z_more_significant', 'mean_neglog10_p_x', ...
    'mean_neglog10_p_z', 'relative_gain', 'shap_base'});
for r = 1:numel(Result)
    X = Result(r).Interpretation;
    if ~isfield(X, 'SHAP')
        continue;
    end
    G = X.Significance;
    I = AddColumns(I, {repmat(Result(r).Level, numel(Result(r).SNP), 1), Result(r).SNP(:), X.Beta, ...
        X.EffectiveWeight, G.PX, G.PZ, X.MeanAbsSHAP, X.SHAPCase, X.SHAPControl, X.SHAPRank});
    [th, mu] = EffectColumns(X.EffectCode, X.Theta, X.Mu);
    F = AddRow(F, {Result(r).Level, X.EffectCode, X.MaxEpoch, th(1), th(2), th(3), mu(1), mu(2), ...
        X.AUCValidation, G.NumZMoreSignificant, G.FracZMoreSignificant, G.MeanLogPX, G.MeanLogPZ, ...
        G.RelativeGain, X.SHAPBase});
end
Files{end + 1} = WriteTable(fullfile(OutputDir, 'interpretation.csv'), I);
Files{end + 1} = WriteTable(fullfile(OutputDir, 'final_model.csv'), F);

%% run_info.json and MAT file
run = struct();
run.software = Info.Software;
stamp = now;
run.created = [datestr(stamp, 'yyyy-mm-dd'), 'T', datestr(stamp, 'HH:MM:SS')];
run.seconds = Info.Seconds;
run.data = Info.Data;
run.num_snp_qc = numel(Info.SNP);
run.level_num_snp = Info.LevelInfo.NumSNP;
run.analysed_levels = Info.AnalysedLevels;
run.effects = Info.Effects;
run.has_genomic = Info.HasGenomic;
run.genomic_threshold = Info.Network.Genomic.Threshold;
run.config = Info.Config;
run.parameter = Info.Parameter;
file = fullfile(OutputDir, 'run_info.json');
fid = fopen(file, 'w', 'n', 'UTF-8');
if fid < 0
    error('NetPRS:ExportResults:Open', 'Cannot write %s.', file);
end
fprintf(fid, '%s\n', jsonencode(run));
fclose(fid);
Files{end + 1} = file;
file = fullfile(OutputDir, 'NetPRS_Result.mat');
save(file, 'Result', 'Info', '-v7');
Files{end + 1} = file;
Files = Files(:);
end

%% ========================================================================
function models = ModelList(R)
if ~isempty(R.Ablation)
    models = R.Ablation;
else
    models = struct('Name', 'NetPRS', 'Effects', R.NetPRS.EffectCode, 'Summary', R.NetPRS);
end
end

function [th, mu] = EffectColumns(code, theta, mu0)
% theta of I, P, G and mu of P, G (NaN for effects not in the model).
th = nan(1, 3);
th(ismember('IPG', code)) = theta;
mu = nan(1, 2);
mu(ismember('PG', code)) = mu0;
end

function se = SEOrNaN(G, n)
if isfield(G, 'SE')
    se = G.SE(:);
else
    se = nan(n, 1);                                  % PLINK report: no SE column
end
end

function T = EmptyColumns(names)
T = struct('Name', {names}, 'Data', {cell(1, numel(names))});
end

function T = AddRow(T, values)
for k = 1:numel(values)
    v = values{k};
    if ischar(v)
        if isempty(T.Data{k})
            T.Data{k} = {v};
        else
            T.Data{k}{end + 1, 1} = v;
        end
    else
        T.Data{k}(end + 1, 1) = double(v);
    end
end
end

function T = AddColumns(T, cols)
for k = 1:numel(cols)
    c = cols{k};
    if iscell(c)
        if isempty(T.Data{k})
            T.Data{k} = c(:);
        else
            T.Data{k} = [T.Data{k}; c(:)];
        end
    else
        T.Data{k} = [T.Data{k}; double(c(:))];
    end
end
end

function file = WriteTable(file, T)
% Header and rows; CSVRows keeps empty fields (NaN), unlike sprintf in MATLAB.
cols = cell(1, numel(T.Name));
for k = 1:numel(T.Name)
    col = T.Data{k};
    if iscell(col)
        cols{k} = col(:);
    else
        cols{k} = NumText(double(col(:)));
    end
end
fid = fopen(file, 'w', 'n', 'UTF-8');
if fid < 0
    error('NetPRS:ExportResults:Open', 'Cannot write %s.', file);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s', [strjoin(T.Name, ','), char(10), CSVRows(cols)]);
end

function t = NumText(v)
% '%.10g' text; NaN -> empty, +/-Inf -> 'Inf'/'-Inf'.
t = repmat({''}, numel(v), 1);
ok = ~isnan(v);
[u, ~, g] = unique(v(ok));
ut = arrayfun(@(x) sprintf('%.10g', x), u, 'UniformOutput', false);
ut(isinf(u) & u > 0) = {'Inf'};
ut(isinf(u) & u < 0) = {'-Inf'};
t(ok) = ut(g);
end
