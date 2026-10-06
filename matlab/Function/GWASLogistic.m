function Res = GWASLogistic(X, Y, Cov, Opts)
%GWASLOGISTIC Single-SNP association test (additive logistic regression).
%
%   Res = GWASLogistic(X, Y)
%   Res = GWASLogistic(X, Y, Cov, Opts)
%
%   For every SNP j the model logit Pr(Y=1) = b0 + b1 * x_j (+ Cov * c) is
%   fitted on complete cases and b1 is tested with the Wald statistic,
%   which is the additive test of PLINK --logistic used for the SNP
%   screening of the NetPRS experiments (Sec. IV-B: SNPs are divided into
%   levels according to their GWAS P-values).
%
%   Inputs
%     X     [s x n] allele dosage (0/1/2), NaN = missing genotype
%     Y     [1 x n] binary phenotype (1 = case, 0 = control)
%     Cov   [n x c] optional covariates (no intercept column; [] = none)
%     Opts  .ChunkSize  SNPs solved simultaneously (default 5000)
%           .MaxIter/.Tol/.PivotTol  passed to LogisticBatch
%
%   Output (s x 1 vectors)
%     Res.Beta   log odds ratio b1      Res.SE    standard error of b1
%     Res.OR     exp(b1)                Res.Stat  Wald z = b1/SE
%     Res.P      two-sided P-value      Res.NumObs complete cases
%     Res.Converged  logical (false -> NaN results, e.g. monomorphic SNP)

if nargin < 3
    Cov = [];
end
if nargin < 4 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'ChunkSize', 5000);

[s, n] = size(X);
Y = double(Y(:));
if numel(Y) ~= n
    error('NetPRS:GWASLogistic:SizeMismatch', 'Y must have one entry per column of X.');
end
if isempty(Cov)
    Cov = zeros(n, 0);
end
if size(Cov, 1) ~= n
    error('NetPRS:GWASLogistic:SizeMismatch', 'Cov must have n rows.');
end
C = [ones(n, 1), double(Cov)];

Res.Beta = nan(s, 1);
Res.SE = nan(s, 1);
Res.NumObs = zeros(s, 1);
Res.Converged = false(s, 1);
for first = 1:Opts.ChunkSize:s
    idx = first:min(first + Opts.ChunkSize - 1, s);
    V = double(X(idx, :))';                       % n x B
    M = isfinite(V);
    Out = LogisticBatch(Y, C, reshape(V, n, numel(idx), 1), M, Opts);
    Res.Beta(idx) = Out.Coef(end, :)';
    Res.SE(idx) = Out.SE(end, :)';
    Res.NumObs(idx) = Out.NumObs';
    Res.Converged(idx) = Out.Converged';
end
Res.OR = exp(Res.Beta);
Res.Stat = Res.Beta ./ Res.SE;
Res.P = erfc(abs(Res.Stat) / sqrt(2));
end
