function [Config, Parameter] = NetPRSDefaults()
%NETPRSDEFAULTS Default configuration of the NetPRS analysis (RunNetPRS).
%
%   [Config, Parameter] = NetPRSDefaults()
%
%   Config     data source, Stage 1 settings, experiments and output
%   Parameter  NetPRS model and training settings (see ModelInitialize)
%
%   Paths are absolute and derived from the location of this file, so the
%   defaults work from any current folder: the data file is
%   <repository>/dataset/sample.csv and results go to <repository>/matlab/result.
%   The Python version (netprs.pipeline.netprs_defaults) has the same defaults.

here = fileparts(mfilename('fullpath'));            % .../matlab/Function
matlabRoot = fileparts(here);                       % .../matlab
repoRoot = fileparts(matlabRoot);

%% Data
Config = struct();
Config.DataSource = 'csv';                          % 'csv' | 'plink' | 'synthetic'
Config.CSVFile = fullfile(repoRoot, 'dataset', 'sample.csv');
% 'plink' input (PLINK QC and LD pruning first; see README.md)
Config.PlinkPrefix = '';                            % .bed/.bim/.fam without extension
Config.ValidationIIDFile = '';                      % IIDs of the validation cohort
Config.SNPGeneFile = '';                            % 'SNP GENE' per line (dbSNP)
Config.GGIFile = '';                                % 'GENE1 GENE2 SCORE' per line (STRING)
Config.GGIWeightScale = 1 / 1000;                   % STRING combined_score -> [0, 1]
Config.GWASFile = '';                               % optional PLINK --logistic report
Config.EpistasisFile = '';                          % optional PLINK --epistasis report
% 'synthetic' input
Config.SyntheticOptions = struct();                 % options of GenerateSyntheticData

%% Stage 1: SNP screening and SNP interaction networks
Config.RunQC = true;                                % call rate, MAF, HWE (Sec. IV-B)
Config.QCOptions = struct();                        % see SNPQualityControl
Config.LevelThresholds = [1 1.5 2];                 % -log10(P) cut-offs (paper: ADNI
                                                    % [3 4 5], BICWALZS [4 5 6]; the
                                                    % sample data have 300 SNPs only)
Config.Levels = 1:3;                                % levels to analyse
Config.MinLevelSNP = 3;                             % smaller levels are skipped
Config.EpistasisPThreshold = 0.05;                  % W_P: epistasis P < 0.05 (paper)
Config.GenomicMu = 1;                               % mu of Eq. (2)-(3) (paper)
Config.GenomicSigma = 1;                            % sigma of T_G (not reported)
Config.GenomicThreshold = 'elbow';                  % 'elbow' (paper) or a number

%% Experiments
Config.Effects = 'IPG';                             % NetPRS effects (Fig. 2)
Config.RunAblation = true;                          % Fig. 3(b): seven effect sets
Config.RunBaseline = true;                          % wPRS (Fig. 3(a))
Config.RunInterpretation = true;                    % Fig. 4: P_X vs P_Z, SHAP
Config.NumKeySNP = 10;                              % SNPs listed by mean |SHAP|
Config.NumWorkers = 0;                              % > 0: parfor (Parallel Computing Toolbox)

%% Output
Config.OutputDir = fullfile(matlabRoot, 'result');  % CSV tables, MAT file, run_info.json
Config.Verbose = true;                              % progress messages

%% NetPRS model and training (values not reported in the paper: see README)
Parameter = struct();
Parameter.NumIter = 10;                             % repetitions of stratified K-fold CV
Parameter.NumFold = 5;                              % K (one fold selects the epoch)
Parameter.MaxEpoch = 500;                           % ADAM steps
Parameter.LearnRate = 0.01;                         % paper
Parameter.RegCoeff = 0.005;                         % delta of Eq. (6)
Parameter.MuInit = 1;                               % paper
Parameter.AlphaInit = 0;                            % paper
Parameter.MuMin = 0;                                % projection mu >= 0
Parameter.Laplacian = 'unnormalized';               % L = D - W (paper)
Parameter.Solver = 'eig';                           % 'eig' (fast) | 'chol' (reference)
Parameter.XScaling = 'center';                      % 'center' | 'none' | 'zscore'
Parameter.Seed = 0;                                 % folds: RngStream(Seed + iter, 1);
                                                    % beta:  RngStream(Seed + iter, 2)
Parameter.Verbose = false;                          % per-epoch messages
end
