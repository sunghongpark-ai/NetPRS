%% NetPRS: SNP interaction aware network-based polygenic risk score
%
%   Park S, Lee D, Kim J, Kim D, Hong CH, Son SJ, Roh HW, Park K, Kim D,
%   Shin H, Woo HG. "NetPRS: SNP interaction aware network-based polygenic
%   risk score for Alzheimer's disease." 2024 IEEE EMBS International
%   Conference on Biomedical and Health Informatics (BHI), pp. 1-8, 2024.
%   doi:10.1109/BHI62660.2024.10913658
%
%   Stage 1  SNP interaction network construction (Sec. II)
%            phenotypic network W_P : epistasis test, Eq. (1), T_P, P < 0.05
%            genomic network    W_G : SNP-gene relations x GGI by GSSL,
%                                     Eq. (2)-(3), T_G, elbow threshold
%   Stage 2  NetPRS (Sec. III)
%            network propagation  F_* = (I + mu_* L_*)^{-1} X        Eq. (4)
%            effect integration   Z = sum_* theta_* F_*, theta = softmax(alpha)
%            risk prediction      P = 1 / (1 + exp(-beta' Z))
%            training             min CE + delta * L2, ADAM (lr 0.01)  Eq. (6)
%
%   By default the analysis runs on ../dataset/sample.csv, a SYNTHETIC
%   dataset in the single-file CSV format (see ReadNetPRSCSV and
%   ../dataset/README.md); the cohorts of the paper cannot be redistributed.
%   To analyse your own data, write it in the same CSV format and set
%   Config.CSVFile, or use PLINK files (Config.DataSource = 'plink').
%
%   MATLAB R2016b or later; no toolbox is required (Parallel Computing
%   Toolbox only for Config.NumWorkers > 0). Tested with GNU Octave 8.4.
%   The Python version in ../python gives the same results.

clc; clear;
here = fileparts(mfilename('fullpath'));
if isempty(here)                     % evaluated section by section
    here = pwd;
end
addpath(fullfile(here, 'Function'));

%% 0. Configuration (defaults: NetPRSDefaults)
[Config, Parameter] = NetPRSDefaults();
Config.CSVFile = fullfile(here, '..', 'dataset', 'sample.csv');
Config.OutputDir = fullfile(here, 'result');
% Config.LevelThresholds = [1 1.5 2];   % -log10(P) cut-offs of the SNP levels
% Config.RunAblation = false;           % NetPRS only (7x faster)
% Parameter.NumIter = 10;               % repetitions of stratified 5-fold CV

%% 1-8. Stage 1 and Stage 2 (data, screening, networks, NetPRS, ablation,
%       wPRS baseline, interpretation)
[Result, Info] = RunNetPRS(Config, Parameter);

%% 9. Export (CSV tables, run_info.json, NetPRS_Result.mat)
Files = ExportResults(Result, Info);
fprintf('\nResults written to %s\n', Config.OutputDir);
