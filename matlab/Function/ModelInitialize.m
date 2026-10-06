function model = ModelInitialize(Dataset, Param)
%MODELINITIALIZE Build the NetPRS model structure (data, networks, CV design).
%
%   model = ModelInitialize(Dataset, Param)
%
%   Dataset (struct)
%     .XData  [s x n] SNP independent effect of the discovery cohort, i.e.
%                     the genotype dosage matrix X in R^{s x n} (Sec. III-A).
%     .YData  [1 x n] diagnosis of the discovery cohort (1 = AD, 0 = non-AD).
%     .XTest  [s x t] (optional) genotype of an external validation cohort.
%     .YTest  [1 x t] (optional) diagnosis of the external validation cohort.
%     .Wphe   [s x s] phenotypic SNP interaction network W_P (Sec. II-A);
%                     required when Param.Effects contains 'P'.
%     .Wgen   [s x s] genomic SNP interaction network W_G (Sec. II-B);
%                     required when Param.Effects contains 'G'.
%     .SNP    {s x 1} (optional) SNP identifiers.
%
%   Param (struct; absent fields take the defaults below)
%     .Effects    'IPG'           effects combined in Z, any non-empty subset
%                                 of 'I','P','G' (ablation models of Fig. 3b)
%     .NumIter    10              repetitions of stratified K-fold splitting
%     .NumFold    5               number of folds K
%     .MaxEpoch   500             number of gradient (Adam) steps
%     .LearnRate  0.01            Adam learning rate (paper, Sec. IV-C)
%     .RegCoeff   0.005           L2 coefficient delta of Eq. (6)
%     .MuInit     1               initial smoothness mu_* (paper, Sec. IV-C)
%     .AlphaInit  0               initial combining coefficient alpha_* (paper)
%     .MuMin      0               lower bound projected onto mu_* after each
%                                 update; -Inf disables the projection
%     .Laplacian  'unnormalized'  L = D - W (paper) | 'normalized'
%     .Solver     'eig'           'eig' | 'chol' (see PropagationSolve)
%     .XScaling   'center'        'center' | 'none' | 'zscore'; statistics are
%                                 estimated on the training fold only.
%                                 The risk P = 1/(1+exp(-beta'Z)) of the paper
%                                 has no intercept; with raw 0/1/2 dosages
%                                 (all >= 0) beta must then absorb the base
%                                 rate, which can invert the risk ranking for
%                                 small SNP sets. Centring each SNP removes
%                                 this without changing the six parameters
%                                 (beta stays in per-allele units).
%     .Seed       0               iteration i draws its folds from the
%                                 portable stream RngStream(Seed + i, 1) and the
%                                 initial beta from RngStream(Seed + i, 2), so
%                                 MATLAB, GNU Octave and the Python version use
%                                 identical folds and initial values
%     .Verbose    false           print the loss every 100 epochs
%
%   Cross-validation design
%     With an external validation cohort (XTest/YTest given), every model
%     is trained on K-1 folds of the discovery cohort, the remaining fold is
%     used to select the epoch with the minimum validation loss, and the
%     external cohort is the test set (CVlist rows: [iter, validFold]).
%     Without an external cohort, a BIGPN-style nested design is used
%     (CVlist rows: [iter, testFold, validFold]). Folds are stratified by Y.
%
%   Values not reported in the paper (MaxEpoch, RegCoeff, beta
%   initialization, CV design, XScaling, MuMin) follow NeuroFANN/BIGPN/PPIxGPN
%   or are declared choices/safeguards; see README.md, "Implementation notes".

if nargin < 2 || isempty(Param)
    Param = struct();
end
Param = SetDefaultField(Param, 'Effects', 'IPG');
Param = SetDefaultField(Param, 'NumIter', 10);
Param = SetDefaultField(Param, 'NumFold', 5);
Param = SetDefaultField(Param, 'MaxEpoch', 500);
Param = SetDefaultField(Param, 'LearnRate', 0.01);
Param = SetDefaultField(Param, 'RegCoeff', 0.005);
Param = SetDefaultField(Param, 'MuInit', 1);
Param = SetDefaultField(Param, 'AlphaInit', 0);
Param = SetDefaultField(Param, 'MuMin', 0);
Param = SetDefaultField(Param, 'Laplacian', 'unnormalized');
Param = SetDefaultField(Param, 'Solver', 'eig');
Param = SetDefaultField(Param, 'XScaling', 'center');
Param = SetDefaultField(Param, 'Seed', 0);
Param = SetDefaultField(Param, 'Verbose', false);
CheckParam(Param);

%% Data
X = double(Dataset.XData);
Y = double(Dataset.YData(:)');
[s, n] = size(X);
CheckData(X, Y, 'XData', 'YData');
hasTest = isfield(Dataset, 'XTest') && ~isempty(Dataset.XTest);
if hasTest
    XT = double(Dataset.XTest);
    YT = double(Dataset.YTest(:)');
    CheckData(XT, YT, 'XTest', 'YTest');
    if size(XT, 1) ~= s
        error('NetPRS:ModelInitialize:SizeMismatch', 'XTest must have the same number of SNPs (rows) as XData.');
    end
else
    XT = zeros(s, 0);
    YT = zeros(1, 0);
    if Param.NumFold < 3
        error('NetPRS:ModelInitialize:Param', ...
            'Without an external validation cohort Param.NumFold must be at least 3 (test, validation and training folds).');
    end
end

model = struct();
model.Param = Param;
model.NumSNP = s;
model.NumDisc = n;
model.NumExt = size(XT, 2);
model.NumAll = n + model.NumExt;
model.HasExternalTest = hasTest;
model.XAll = [X, XT];
model.YAll = [Y, YT];
if isfield(Dataset, 'SNP') && ~isempty(Dataset.SNP)
    model.SNP = Dataset.SNP(:);
else
    model.SNP = arrayfun(@(i) sprintf('SNP%d', i), (1:s)', 'UniformOutput', false);
end

%% Effects (I: independent, P: phenotypic, G: genomic)
[useEffect, effectCode] = ParseEffects(Param.Effects);
model.EffectCode = effectCode;
model.EffectList = find(useEffect);          % indices into {I, P, G}
model.NumEffect = numel(model.EffectList);
model.UseIndependent = useEffect(1);

%% SNP interaction networks: Laplacian (and eigendecomposition)
netNames = {'P', 'G'};
netFields = {'Wphe', 'Wgen'};
Net = struct('Name', {}, 'Effect', {}, 'W', {}, 'L', {}, 'V', {}, 'Lambda', {}, 'D', {}, 'R', {});
for k = 1:2
    if ~useEffect(k + 1)
        continue;
    end
    if ~isfield(Dataset, netFields{k}) || isempty(Dataset.(netFields{k}))
        error('NetPRS:ModelInitialize:MissingNetwork', ...
            'Dataset.%s is required for effect ''%s''.', netFields{k}, netNames{k});
    end
    W = Dataset.(netFields{k});
    if ~isequal(size(W), [s, s])
        error('NetPRS:ModelInitialize:NetworkSize', 'Dataset.%s must be %d x %d.', netFields{k}, s, s);
    end
    W = full(double(W));
    L = GraphLaplacian(W, Param.Laplacian);
    item = struct('Name', netNames{k}, 'Effect', k + 1, 'W', W, 'L', L, ...
        'V', [], 'Lambda', [], 'D', [], 'R', []);
    if strcmpi(Param.Solver, 'eig')
        [V, Lam] = eig(L);
        lambda = diag(Lam);
        % L is positive semi-definite; clip rounding-level negative eigenvalues.
        tol = 1e-12 * max(1, max(abs(lambda)));
        if min(lambda) < -tol * s
            error('NetPRS:ModelInitialize:NotPSD', 'Laplacian of %s is not positive semi-definite.', netFields{k});
        end
        lambda(lambda < 0) = 0;
        item.V = V;
        item.Lambda = lambda;
    end
    Net(end + 1) = item; %#ok<AGROW>
end
model.Net = Net;
model.NumNet = numel(Net);

%% Stratified cross-validation design (BIGPN style: mod(randperm(n),K)+1)
% Iteration i permutes the controls and then the cases with the portable
% stream RngStream(Seed + i, 1) (see RngPermutation); fold = mod(p, K) + 1.
K = Param.NumFold;
classes = [0, 1];
CVdata = zeros(Param.NumIter, n);
for iter = 1:Param.NumIter
    S = RngStream(Param.Seed + iter, 1);
    for c = classes
        idx = find(Y == c);
        [perm, S] = RngPermutation(S, numel(idx));
        CVdata(iter, idx) = mod(perm(:)', K) + 1;
    end
end
if any(CVdata(:) == 0)
    error('NetPRS:ModelInitialize:CVIndex', 'Cross-validation index setting failed.');
end
counts = zeros(K, 2);
for c = 1:2
    for f = 1:K
        counts(f, c) = sum(CVdata(1, :) == f & Y == classes(c));
    end
end
if any(counts(:, 2) == 0)
    warning('NetPRS:ModelInitialize:EmptyClassFold', ...
        'At least one fold contains no case; reduce NumFold.');
end

if hasTest
    [ff, ii] = ndgrid(1:K, 1:Param.NumIter);
    CVlist = [ii(:), ff(:)];
    model.CVMode = 'external';
else
    pairs = zeros(K * (K - 1), 2);
    row = 0;
    for ft = 1:K
        for fv = 1:K
            if fv ~= ft
                row = row + 1;
                pairs(row, :) = [ft, fv];
            end
        end
    end
    CVlist = zeros(Param.NumIter * size(pairs, 1), 3);
    for iter = 1:Param.NumIter
        rows = (iter - 1) * size(pairs, 1) + (1:size(pairs, 1));
        CVlist(rows, :) = [repmat(iter, size(pairs, 1), 1), pairs];
    end
    model.CVMode = 'nested';
end
model.CVdata = CVdata;
model.CVlist = CVlist;
model.NumModel = size(CVlist, 1);
end

%% ------------------------------------------------------------------------
function [useEffect, code] = ParseEffects(Effects)
if ~ischar(Effects)
    error('NetPRS:ModelInitialize:Effects', 'Param.Effects must be a character vector such as ''IPG''.');
end
letters = upper(Effects(isletter(Effects)));
if isempty(letters) || any(~ismember(letters, 'IPG')) || numel(unique(letters)) ~= numel(letters)
    error('NetPRS:ModelInitialize:Effects', ...
        'Param.Effects must be a non-empty subset of ''I'', ''P'', ''G'' without repetition.');
end
useEffect = ismember('IPG', letters);
code = 'IPG';
code = code(useEffect);
end

function CheckData(X, Y, nameX, nameY)
if ~isnumeric(X) || ndims(X) ~= 2 || isempty(X)
    error('NetPRS:ModelInitialize:InvalidX', '%s must be a non-empty numeric s x n matrix.', nameX);
end
if any(~isfinite(X(:)))
    error('NetPRS:ModelInitialize:MissingGenotype', ...
        '%s contains NaN/Inf; impute missing genotypes first (see ImputeGenotype).', nameX);
end
if numel(Y) ~= size(X, 2)
    error('NetPRS:ModelInitialize:InvalidY', '%s must have one label per column of %s.', nameY, nameX);
end
if any(Y ~= 0 & Y ~= 1)
    error('NetPRS:ModelInitialize:InvalidY', '%s must be binary (0/1).', nameY);
end
end

function CheckParam(P)
mustPosInt = {'NumIter', 'NumFold', 'MaxEpoch'};
for i = 1:numel(mustPosInt)
    v = P.(mustPosInt{i});
    if ~isscalar(v) || ~isnumeric(v) || v < 1 || v ~= fix(v)
        error('NetPRS:ModelInitialize:Param', 'Param.%s must be a positive integer.', mustPosInt{i});
    end
end
if P.NumFold < 2
    error('NetPRS:ModelInitialize:Param', 'Param.NumFold must be at least 2.');
end
mustNonNeg = {'LearnRate', 'RegCoeff'};
for i = 1:numel(mustNonNeg)
    v = P.(mustNonNeg{i});
    if ~isscalar(v) || ~isnumeric(v) || ~isfinite(v) || v < 0
        error('NetPRS:ModelInitialize:Param', 'Param.%s must be a finite non-negative scalar.', mustNonNeg{i});
    end
end
if ~isscalar(P.Seed) || ~isnumeric(P.Seed) || P.Seed < 0 || P.Seed ~= fix(P.Seed) || P.Seed + P.NumIter >= 2^32
    error('NetPRS:ModelInitialize:Param', 'Param.Seed must be a non-negative integer with Seed + NumIter < 2^32.');
end
if ~isscalar(P.MuInit) || ~isfinite(P.MuInit) || P.MuInit < P.MuMin
    error('NetPRS:ModelInitialize:Param', 'Param.MuInit must be finite and not below Param.MuMin.');
end
if ~any(strcmpi(P.Solver, {'eig', 'chol'}))
    error('NetPRS:ModelInitialize:Param', 'Param.Solver must be ''eig'' or ''chol''.');
end
if ~any(strcmpi(P.XScaling, {'none', 'center', 'zscore'}))
    error('NetPRS:ModelInitialize:Param', 'Param.XScaling must be ''none'', ''center'' or ''zscore''.');
end
end
