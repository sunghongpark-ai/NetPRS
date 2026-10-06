function [Result, Info] = RunNetPRS(Config, Parameter)
%RUNNETPRS Complete NetPRS analysis of one dataset (Stage 1 and Stage 2).
%
%   [Result, Info] = RunNetPRS()                    defaults of NetPRSDefaults
%   [Result, Info] = RunNetPRS(Config, Parameter)   absent fields take the defaults
%
%   Steps (sections of the paper in brackets)
%     1. Data            LoadNetPRSData: 'csv' (dataset/sample.csv by default),
%                        'plink' or 'synthetic'; discovery/validation cohorts
%     2. SNP screening   quality control and mean imputation with the
%                        discovery cohort, additive logistic GWAS, SNP levels
%                        by -log10 P [IV-B]
%     3. Networks        W_P from the epistasis test [II-A] and W_G from GSSL
%                        [II-B], built once on the union of the analysed
%                        levels (a single elbow threshold) and restricted to
%                        every level
%     4. NetPRS          effects Config.Effects ('IPG'); repeated stratified
%                        K-fold CV in the discovery cohort (one fold selects the
%                        epoch), validation cohort = test set [IV-C]
%     5. Ablation        the seven effect sets of Fig. 3(b)
%     6. Baseline        wPRS with the discovery-cohort GWAS log odds ratios
%     7. Interpretation  final model on the whole discovery cohort trained for
%                        the median selected epoch; P_X vs P_Z [Fig. 4(a)];
%                        exact linear SHAP [Fig. 4(b)]
%   The AUC reported for every model is the mean test AUC of its
%   NumIter x NumFold cross-validated fits ("average of AUCs").
%   Without SNP-gene relations in the data, effects containing G are skipped.
%
%   Outputs
%     Result  struct array, one element per analysed level with fields
%             Level, Threshold, SNP, SNPIndex, Wphe, Wgen, NetworkSummary,
%             NetPRS (see SummarizeCV), Ablation (Name, Effects, Summary),
%             wPRS (Score, AUC) and Interpretation
%     Info    Config, Parameter, Data (summary), KeepQC, QC, SNP (QC-passed),
%             GWAS, LevelIdx, LevelInfo, AnalysedLevels, Effects, HasGenomic,
%             Network (networks of the union of the levels), Software, Seconds
%   ExportResults writes the CSV tables, the MAT file and run_info.json.

[defConfig, defParam] = NetPRSDefaults();
if nargin < 1 || isempty(Config)
    Config = struct();
end
if nargin < 2 || isempty(Parameter)
    Parameter = struct();
end
Config = MergeDefaults(Config, defConfig);
Parameter = MergeDefaults(Parameter, defParam);
CheckConfig(Config);
Config.Levels = reshape(Config.Levels, 1, []);
Config.LevelThresholds = reshape(Config.LevelThresholds, 1, []);
tStart = tic;

%% 1. Data
Data = LoadNetPRSData(Config);
isDisc = ~Data.IsValidation(:)';
Y = double(Data.Y(:)');
DataInfo = struct('Source', Data.Source, 'NumSNP', size(Data.Geno, 1), 'NumSubject', size(Data.Geno, 2), ...
    'NumDiscovery', nnz(isDisc), 'NumValidation', nnz(~isDisc), ...
    'NumCaseDiscovery', nnz(Y(isDisc) == 1), 'NumCaseValidation', nnz(Y(~isDisc) == 1), ...
    'NumGene', numel(Data.Gene), 'NumRelation', numel(Data.RelSNP), ...
    'NumGGIEdge', nnz(triu(Data.Wggi, 1)), 'NumMissing', nnz(isnan(Data.Geno)));
Log(Config, '%s data: %d SNPs, %d subjects (discovery %d with %d cases / validation %d with %d cases)\n', ...
    DataInfo.Source, DataInfo.NumSNP, DataInfo.NumSubject, DataInfo.NumDiscovery, DataInfo.NumCaseDiscovery, ...
    DataInfo.NumValidation, DataInfo.NumCaseValidation);
Log(Config, '  %d genes, %d SNP-gene relations, %d gene-gene interactions, %d missing calls\n', ...
    DataInfo.NumGene, DataInfo.NumRelation, DataInfo.NumGGIEdge, DataInfo.NumMissing);
if numel(unique(Y(isDisc))) < 2
    error('NetPRS:RunNetPRS:Data', 'The discovery cohort must contain cases and controls.');
end
if ~any(~isDisc)
    error('NetPRS:RunNetPRS:Data', 'The validation cohort is empty (IsValidation is false for every subject).');
end
if numel(unique(Y(~isDisc))) < 2
    warning('NetPRS:RunNetPRS:Validation', 'The validation cohort lacks cases or controls; test AUCs are NaN.');
end

%% 2. SNP screening (discovery cohort only): QC -> imputation -> GWAS -> levels
if Config.RunQC
    [keepQC, QC] = SNPQualityControl(Data.Geno(:, isDisc), Y(isDisc), Config.QCOptions);
else
    keepQC = true(size(Data.Geno, 1), 1);
    QC = struct();
end
Geno = ImputeGenotype(Data.Geno(keepQC, :), find(isDisc));        % discovery means
SNPID = Data.SNP(keepQC);
SNPID = SNPID(:);
Log(Config, 'Quality control: %d of %d SNPs kept\n', numel(SNPID), numel(keepQC));
if isempty(Config.GWASFile)
    GWAS = GWASLogistic(Geno(:, isDisc), Y(isDisc));
else                                                              % PLINK report
    if isempty(Data.A1)
        A1 = {};
        A2 = {};
    else
        A1 = Data.A1(keepQC);
        A2 = Data.A2(keepQC);
    end
    GWAS = MatchPlinkAssoc(ReadPlinkAssoc(Config.GWASFile), SNPID, A1, A2);
end
[LevelIdx, LevelInfo] = SelectSNPLevel(GWAS.P, Config.LevelThresholds);
for l = 1:numel(LevelIdx)
    Log(Config, '  level %d (-log10 P > %g): %d SNPs\n', l, Config.LevelThresholds(l), LevelInfo.NumSNP(l));
end
analysed = Config.Levels(LevelInfo.NumSNP(Config.Levels) >= Config.MinLevelSNP);
for l = setdiff(Config.Levels, analysed)
    Log(Config, '  level %d skipped (fewer than %d SNPs)\n', l, Config.MinLevelSNP);
end
if isempty(analysed)
    error('NetPRS:RunNetPRS:NoLevel', 'No SNP level has %d or more SNPs; lower Config.LevelThresholds.', ...
        Config.MinLevelSNP);
end

%% 3. SNP interaction networks on the union of the analysed levels
% Both interactions are pairwise, so restricting the networks of the union
% to a level equals building them on the level; the elbow threshold of W_G
% is therefore shared by all levels (the paper reports a single cut-off).
allIdx = unique(vertcat(LevelIdx{analysed}));
XAllDisc = Geno(allIdx, isDisc);
YDisc = Y(isDisc);
if isempty(Config.EpistasisFile)
    Epi = EpistasisTest(XAllDisc, YDisc);                                     % Eq. (1)
else
    Epi = ReadPlinkEpistasis(Config.EpistasisFile, SNPID(allIdx));
end
[WpheAll, InfoP] = PhenotypicNetwork(Epi, struct('PThreshold', Config.EpistasisPThreshold));
hasGenomic = ~isempty(Data.RelSNP);
effects = upper(Config.Effects);
if hasGenomic
    Gmat = SNPGeneMatrix(SNPID(allIdx), Data.Gene, Data.RelSNP, Data.RelGene, Data.RelWeight);
    [WgenAll, InfoG] = GenomicNetwork(Gmat, Data.Wggi, struct('Mu', Config.GenomicMu, ...
        'Sigma', Config.GenomicSigma, 'Threshold', Config.GenomicThreshold));
else
    warning('NetPRS:RunNetPRS:NoGenomicKnowledge', ...
        'No SNP-gene relations: W_G cannot be built and effects with G are skipped.');
    WgenAll = zeros(numel(allIdx));
    InfoG = struct('Threshold', NaN, 'NumEdge', 0);
    effects = effects(effects ~= 'G');
    if isempty(effects)
        error('NetPRS:RunNetPRS:Effects', 'Config.Effects needs I or P when W_G is unavailable.');
    end
end
Log(Config, 'Networks on %d SNPs: W_P %d edges (P < %g), W_G %d edges (threshold %.6g)\n', ...
    numel(allIdx), InfoP.NumEdge, Config.EpistasisPThreshold, InfoG.NumEdge, InfoG.Threshold);

%% 4-7. NetPRS, ablation, baseline and interpretation for every level
Result = struct('Level', {}, 'Threshold', {}, 'SNP', {}, 'SNPIndex', {}, 'Wphe', {}, 'Wgen', {}, ...
    'NetworkSummary', {}, 'NetPRS', {}, 'Ablation', {}, 'wPRS', {}, 'Interpretation', {});
Shared = struct('LevelIdx', {LevelIdx}, 'AllIdx', allIdx, 'WpheAll', WpheAll, 'WgenAll', WgenAll, ...
    'Geno', Geno, 'Y', Y, 'IsDiscovery', isDisc, 'SNPID', {SNPID}, 'GWASBeta', GWAS.Beta, ...
    'Effects', effects, 'HasGenomic', hasGenomic);
for level = analysed
    Result(end + 1) = AnalyseLevel(level, Shared, Config, Parameter); %#ok<AGROW>
end

%% 8. Summary
PrintSummary(Config, Result);
Info = struct();
Info.Config = Config;
Info.Parameter = Parameter;
Info.Data = DataInfo;
Info.KeepQC = keepQC;
Info.QC = QC;
Info.AllSNP = Data.SNP(:);
Info.SNP = SNPID;
Info.GWAS = GWAS;
Info.LevelIdx = LevelIdx;
Info.LevelInfo = LevelInfo;
Info.AnalysedLevels = analysed;
Info.Effects = effects;
Info.HasGenomic = hasGenomic;
Info.Network = struct('SNPIndex', allIdx, 'SNP', {SNPID(allIdx)}, 'Wphe', WpheAll, 'Wgen', WgenAll, ...
    'Phenotypic', InfoP, 'Genomic', InfoG);
Info.Software = SoftwareVersion();
Info.Seconds = toc(tStart);
Log(Config, 'Finished in %.1f s\n', Info.Seconds);
end

%% ========================================================================
function R = AnalyseLevel(level, L, Config, Parameter)
idx = L.LevelIdx{level};
pos = find(ismember(L.AllIdx, idx));                 % idx and AllIdx are sorted
Wphe = L.WpheAll(pos, pos);
Wgen = L.WgenAll(pos, pos);
isDisc = L.IsDiscovery;
XDisc = L.Geno(idx, isDisc);
XVal = L.Geno(idx, ~isDisc);
YDisc = L.Y(isDisc);
YVal = L.Y(~isDisc);
snp = L.SNPID(idx);
Log(Config, '\n================ Level %d: %d SNPs (-log10 P > %g) ================\n', ...
    level, numel(idx), Config.LevelThresholds(level));
label = '';
if Config.Verbose
    fprintf('%-28s %6s | %8s %7s %9s | %8s %7s %9s | %6s %8s %10s\n', 'Network (Table III)', 'Nodes', ...
        'Edges_P', 'Avg_P', 'Dens_P', 'Edges_G', 'Avg_G', 'Dens_G', 'Common', 'Corr', 'KS P');
    label = sprintf('Level %d', level);
end
NetSum = NetworkSummary(Wphe, Wgen, label);

%% 4. NetPRS
Dataset = struct('XData', XDisc, 'YData', YDisc, 'XTest', XVal, 'YTest', YVal, 'Wphe', Wphe, 'Wgen', Wgen);
Dataset.SNP = snp;
codes = {'I', 'P', 'G', 'IP', 'IG', 'PG', 'IPG'};
names = {'Phi_I', 'Phi_P', 'Phi_G', 'Phi_I+P', 'Phi_I+G', 'Phi_P+G', 'NetPRS'};
mainName = names{strcmp(codes, L.Effects)};
P = Parameter;
P.Effects = L.Effects;
ModelInit = ModelInitialize(Dataset, P);
SummaryMain = SummarizeCV(ModelInit, RunCrossValidation(ModelInit, Config.NumWorkers), ...
    sprintf('%s (%s)', mainName, L.Effects), Config.Verbose);

%% 5. Ablation study (Fig. 3b)
Ablation = struct('Name', {}, 'Effects', {}, 'Summary', {});
if Config.RunAblation
    for a = 1:numel(codes)
        if ~L.HasGenomic && any(codes{a} == 'G')
            continue;
        end
        if strcmp(codes{a}, L.Effects)
            S = SummaryMain;
        else
            P.Effects = codes{a};
            MI = ModelInitialize(Dataset, P);
            S = SummarizeCV(MI, RunCrossValidation(MI, Config.NumWorkers), ...
                sprintf('%s (%s)', names{a}, codes{a}), Config.Verbose);
        end
        Ablation(end + 1) = struct('Name', names{a}, 'Effects', codes{a}, 'Summary', S); %#ok<AGROW>
    end
end

%% 6. Baseline: weighted PRS with the discovery-cohort GWAS effect sizes
wPRS = struct('Score', [], 'AUC', NaN);
if Config.RunBaseline
    wPRS.Score = WeightedPRS(XVal, L.GWASBeta(idx));
    wPRS.AUC = ComputeAUC(wPRS.Score, YVal);
    Log(Config, '%-24s AUC %.4f (single score)\n', 'wPRS', wPRS.AUC);
end

%% 7. Interpretation (Fig. 4) with a final model fitted on the discovery cohort
Interp = struct();
if Config.RunInterpretation
    Interp = Interpretation(ModelInit, SummaryMain, XDisc, XVal, YVal, snp, Config);
end

R = struct('Level', level, 'Threshold', Config.LevelThresholds(level), 'SNP', {snp}, 'SNPIndex', idx, ...
    'Wphe', Wphe, 'Wgen', Wgen, 'NetworkSummary', NetSum, 'NetPRS', SummaryMain, ...
    'Ablation', Ablation, 'wPRS', wPRS, 'Interpretation', Interp);
end

%% ------------------------------------------------------------------------
function Interp = Interpretation(ModelInit, Summary, XDisc, XVal, YVal, snp, Config)
% Final model: all discovery subjects, median selected epoch (half away from
% zero), same initialization stream as iteration 1.
ModelFinal = ModelInit;
ModelFinal.Param.MaxEpoch = round(median(Summary.BestEpoch));
split = struct('IdxTrain', 1:ModelInit.NumDisc, 'IdxValid', [], ...
    'IdxTest', ModelInit.NumDisc + (1:ModelInit.NumExt), 'IdxIter', 1);
[~, Final] = TrainNetPRS(ModelFinal, split);
EffVal = EffectExtraction(Final, XVal);                % validation cohort
EffDisc = EffectExtraction(Final, XDisc);              % SHAP background
Interp = struct();
Interp.MaxEpoch = ModelFinal.Param.MaxEpoch;
Interp.Significance = EffectSignificance(XVal, EffVal.Z, YVal);
[Interp.SHAP, Interp.SHAPBase] = LinearSHAP(Final.Beta, EffVal.Z, EffDisc.Z);
Interp.MeanAbsSHAP = mean(abs(Interp.SHAP), 2);
Interp.SHAPCase = mean(Interp.SHAP(:, YVal == 1), 2);
Interp.SHAPControl = mean(Interp.SHAP(:, YVal == 0), 2);
[~, order] = sort(Interp.MeanAbsSHAP, 'descend');
rank = zeros(numel(order), 1);
rank(order) = 1:numel(order);
Interp.SHAPRank = rank;
top = order(1:min(Config.NumKeySNP, numel(order)));
Interp.KeySNP = snp(top);
Interp.Beta = Final.Beta;
Interp.Theta = Final.Theta;
Interp.Mu = Final.Mu;
Interp.EffectCode = Final.EffectCode;
Interp.EffectiveWeight = Final.Omega;                  % beta' Z = Omega' X
Interp.AUCValidation = ComputeAUC(RiskPredict(Final, XVal), YVal);
S = Interp.Significance;
Log(Config, 'Final model (%d epochs): theta %s | mu %s | validation AUC %.4f\n', Interp.MaxEpoch, ...
    VectorText(Final.Theta), VectorText(Final.Mu), Interp.AUCValidation);
Log(Config, 'P_Z < P_X for %d / %d SNPs (%.1f%%); mean -log10 P: X %.3f, Z %.3f (%+.1f%%)\n', ...
    S.NumZMoreSignificant, numel(S.PX), 100 * S.FracZMoreSignificant, S.MeanLogPX, S.MeanLogPZ, ...
    100 * S.RelativeGain);
Log(Config, 'Key SNPs by mean |SHAP| (mean SHAP in cases / controls):\n');
for t = 1:numel(top)
    Log(Config, '  %-12s %+7.3f / %+7.3f\n', snp{top(t)}, Interp.SHAPCase(top(t)), Interp.SHAPControl(top(t)));
end
end

%% ------------------------------------------------------------------------
function PrintSummary(Config, Result)
if ~Config.Verbose || isempty(Result) || isempty(Result(1).Ablation)
    return;
end
fprintf('\nMean test AUC of the cross-validated models (validation cohort)\n');
fprintf('%-10s', 'Model');
fprintf('   Level %d', [Result.Level]);
fprintf('\n');
for a = 1:numel(Result(1).Ablation)
    fprintf('%-10s', strrep(Result(1).Ablation(a).Name, 'Phi_', ''));
    for r = 1:numel(Result)
        fprintf('   %.4f ', Result(r).Ablation(a).Summary.AUCMean);
    end
    fprintf('\n');
end
if Config.RunBaseline
    fprintf('%-10s', 'wPRS');
    for r = 1:numel(Result)
        fprintf('   %.4f ', Result(r).wPRS.AUC);
    end
    fprintf('\n');
end
end

function S = MergeDefaults(S, Default)
if ~isstruct(S) || ~isscalar(S)
    error('NetPRS:RunNetPRS:Config', 'Config and Parameter must be scalar structs.');
end
names = fieldnames(Default);
for k = 1:numel(names)
    S = SetDefaultField(S, names{k}, Default.(names{k}));
end
end

function CheckConfig(C)
if ~ischar(C.DataSource) || ~any(strcmpi(C.DataSource, {'csv', 'plink', 'synthetic'}))
    error('NetPRS:RunNetPRS:Config', 'Config.DataSource must be ''csv'', ''plink'' or ''synthetic''.');
end
t = C.LevelThresholds;
if isempty(t) || ~isnumeric(t) || ~isvector(t) || any(~isfinite(t))
    error('NetPRS:RunNetPRS:Config', 'Config.LevelThresholds must be a vector of finite -log10(P) cut-offs.');
end
v = C.Levels;
if isempty(v) || any(v < 1 | v > numel(t) | v ~= fix(v)) || numel(unique(v)) ~= numel(v)
    error('NetPRS:RunNetPRS:Config', 'Config.Levels must list distinct levels in 1..%d.', numel(t));
end
if ~(isscalar(C.MinLevelSNP) && C.MinLevelSNP >= 2 && C.MinLevelSNP == fix(C.MinLevelSNP))
    error('NetPRS:RunNetPRS:Config', 'Config.MinLevelSNP must be an integer >= 2.');
end
if ~(isscalar(C.EpistasisPThreshold) && C.EpistasisPThreshold > 0 && C.EpistasisPThreshold <= 1)
    error('NetPRS:RunNetPRS:Config', 'Config.EpistasisPThreshold must lie in (0, 1].');
end
codes = {'I', 'P', 'G', 'IP', 'IG', 'PG', 'IPG'};
if ~ischar(C.Effects) || ~any(strcmp(upper(C.Effects), codes))
    error('NetPRS:RunNetPRS:Config', 'Config.Effects must be one of I, P, G, IP, IG, PG, IPG.');
end
if ~(isscalar(C.NumKeySNP) && C.NumKeySNP >= 0 && C.NumKeySNP == fix(C.NumKeySNP))
    error('NetPRS:RunNetPRS:Config', 'Config.NumKeySNP must be a non-negative integer.');
end
if ~(isscalar(C.NumWorkers) && C.NumWorkers >= 0)
    error('NetPRS:RunNetPRS:Config', 'Config.NumWorkers must be >= 0.');
end
end

function Log(Config, varargin)
if Config.Verbose
    fprintf(varargin{:});
end
end

function t = VectorText(v)
% '[a b c]' or '-' (never empty: MATLAB's fprintf skips empty arguments).
if isempty(v)
    t = '-';
else
    t = ['[', strtrim(sprintf('%.3g ', v)), ']'];
end
end

function v = SoftwareVersion()
if exist('OCTAVE_VERSION', 'builtin') ~= 0
    v = ['GNU Octave ', OCTAVE_VERSION];
else
    v = ['MATLAB ', version];
end
end
