function [Config, Parameter] = NetPRSDefaults()

here = fileparts(mfilename('fullpath'));
matlabRoot = fileparts(here);
repoRoot = fileparts(matlabRoot);

Config = struct();
Config.DataSource = 'csv';
Config.CSVFile = fullfile(repoRoot, 'dataset', 'sample.csv');

Config.PlinkPrefix = '';
Config.ValidationIIDFile = '';
Config.SNPGeneFile = '';
Config.GGIFile = '';
Config.GGIWeightScale = 1 / 1000;
Config.GWASFile = '';
Config.EpistasisFile = '';

Config.SyntheticOptions = struct();

Config.RunQC = true;
Config.QCOptions = struct();
Config.LevelThresholds = [1 1.5 2];

Config.Levels = 1:3;
Config.MinLevelSNP = 3;
Config.EpistasisPThreshold = 0.05;
Config.GenomicMu = 1;
Config.GenomicSigma = 1;
Config.GenomicThreshold = 'elbow';

Config.Effects = 'IPG';
Config.RunAblation = true;
Config.RunBaseline = true;
Config.RunInterpretation = true;
Config.NumKeySNP = 10;
Config.NumWorkers = 0;

Config.OutputDir = fullfile(tempdir, 'NetPRS_matlab_results');
Config.Verbose = true;

Parameter = struct();
Parameter.NumIter = 10;
Parameter.NumFold = 5;
Parameter.MaxEpoch = 500;
Parameter.LearnRate = 0.01;
Parameter.RegCoeff = 0.005;
Parameter.MuInit = 1;
Parameter.AlphaInit = 0;
Parameter.MuMin = 0;
Parameter.Laplacian = 'unnormalized';
Parameter.Solver = 'eig';
Parameter.XScaling = 'center';
Parameter.Seed = 0;

Parameter.Verbose = false;
end
