function [Wg, Info] = GenomicNetwork(G, Wggi, Opts)
%GENOMICNETWORK Genomic SNP interaction network W_G (Sec. II-B).
%
%   [Wg, Info] = GenomicNetwork(G, Wggi)
%   [Wg, Info] = GenomicNetwork(G, Wggi, Opts)
%
%   SNP-gene relations g in R^{m x s} are combined with the gene-gene
%   interaction (GGI) network W in R^{m x m} by graph-based semi-supervised
%   learning (GSSL). For SNP j the effect of its related genes on the GGI,
%       f^j = argmin_f (f - g^j)'(f - g^j) + mu f' L f,  L = D - W     (2)
%           = (I + mu L)^{-1} g^j,                                     (3)
%   gives the genomic interaction h_G^{jk} = f^j' g^k, transformed by
%       T_G(x) = exp(ln(x) / sigma^2),     W_G^{jk} = T_G(h_G^{jk}),
%   and W_G is obtained by thresholding (elbow point in the paper).
%   Because I + mu L is symmetric, h_G^{jk} = h_G^{kj}: W_G is undirected.
%
%   Inputs
%     G     [m x s] SNP-gene relation (non-negative; binary from dbSNP),
%                   full or sparse (see SNPGeneMatrix)
%     Wggi  [m x m] symmetric non-negative GGI adjacency (e.g. STRING
%                   combined_score/1000), full or sparse (see ReadEdgeList)
%     Opts  .Mu          smoothness mu of Eq. (2)-(3)       (default 1, paper)
%           .Sigma       scaling parameter sigma of T_G     (default 1; not
%                        reported in the paper; sigma = 1 makes T_G the identity)
%           .Threshold   'elbow' (default) | numeric value | 'none'
%                        (the paper used the elbow point 5.35e-5)
%           .ThresholdOn 'transformed' (W_G values, default) | 'raw' (h_G)
%           .ElbowScale  'linear' (default) | 'log', see ElbowThreshold
%           .TieTol      relative tolerance of the threshold comparison
%                        (default 1e-9): values within TieTol*|tau| below the
%                        threshold tau are kept. Binary SNP-gene relations give
%                        many mathematically equal h_G values that may differ
%                        in the last bits; the tolerance keeps or drops such
%                        ties together, independently of rounding (and of the
%                        BLAS library or language)
%           .Laplacian   'unnormalized' (default, L = D - W) | 'normalized'
%           .Solver      'auto' (default) | 'dense' | 'sparse' | 'pcg'
%           .DenseMax    'auto' uses 'dense' up to this many genes, otherwise
%                        'pcg' (default 12000)
%           .PcgTol      relative residual tolerance of 'pcg' (default 1e-10)
%           .PcgMaxIter  (default 2000)
%
%   Outputs
%     Wg    [s x s] symmetric, non-negative, zero diagonal
%     Info  .H (h_G, s x s), .Tg (T_G(h_G)), .Threshold, .NumEdge, .Density,
%           .NumGeneUsed, .Solver, .MaxRelResidual
%
%   Exactness of the gene restriction: (I + mu L)^{-1} is block diagonal
%   over the connected components of the GGI network, so components that
%   contain no SNP-related gene contribute nothing to h_G and are dropped.

if nargin < 3 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'Mu', 1);
Opts = SetDefaultField(Opts, 'Sigma', 1);
Opts = SetDefaultField(Opts, 'Threshold', 'elbow');
Opts = SetDefaultField(Opts, 'ThresholdOn', 'transformed');
Opts = SetDefaultField(Opts, 'ElbowScale', 'linear');
Opts = SetDefaultField(Opts, 'TieTol', 1e-9);
Opts = SetDefaultField(Opts, 'Laplacian', 'unnormalized');
Opts = SetDefaultField(Opts, 'Solver', 'auto');
Opts = SetDefaultField(Opts, 'DenseMax', 12000);
Opts = SetDefaultField(Opts, 'PcgTol', 1e-10);
Opts = SetDefaultField(Opts, 'PcgMaxIter', 2000);

[m, s] = size(G);
if ~isequal(size(Wggi), [m, m])
    error('NetPRS:GenomicNetwork:SizeMismatch', 'Wggi must be m x m with m = size(G,1).');
end
if any(nonzeros(G) < 0) || any(~isfinite(nonzeros(G)))
    error('NetPRS:GenomicNetwork:InvalidG', 'G must be finite and non-negative.');
end
if ~(Opts.Mu >= 0) || ~(Opts.Sigma > 0)
    error('NetPRS:GenomicNetwork:Param', 'Opts.Mu must be >= 0 and Opts.Sigma > 0.');
end

%% Keep the GGI components that contain at least one SNP-related gene (exact)
Wggi = sparse(double(Wggi));
G = sparse(double(G));
comp = ConnectedComponents(Wggi);
related = any(G ~= 0, 2);
keepGene = ismember(comp, unique(comp(related)));
Gk = G(keepGene, :);
Wk = Wggi(keepGene, keepGene);
mk = nnz(keepGene);

%% Propagation F = (I + mu L)^{-1} G
L = GraphLaplacian(Wk, Opts.Laplacian);
Q = speye(mk) + Opts.Mu * L;
solver = lower(Opts.Solver);
if strcmp(solver, 'auto')
    if mk <= Opts.DenseMax
        solver = 'dense';
    else
        solver = 'pcg';
    end
end
cols = find(any(Gk ~= 0, 1));
F = zeros(mk, s);
if isempty(cols)
    solver = 'none';                                 % no analysed SNP has a related gene
end
switch solver
    case 'none'
    case 'dense'
        [R, p] = chol(full(Q));
        if p ~= 0
            error('NetPRS:GenomicNetwork:NotPositiveDefinite', 'I + mu*L is not positive definite.');
        end
        F(:, cols) = R \ (R' \ full(Gk(:, cols)));
    case 'sparse'
        F(:, cols) = Q \ full(Gk(:, cols));
    case 'pcg'
        dq = full(diag(Q));
        precond = @(v) v ./ dq;                       % Jacobi preconditioner
        for j = cols
            [f, flag] = pcg(Q, full(Gk(:, j)), Opts.PcgTol, Opts.PcgMaxIter, precond);
            if flag ~= 0
                error('NetPRS:GenomicNetwork:PcgFailed', 'pcg did not converge for SNP %d (flag %d).', j, flag);
            end
            F(:, j) = f;
        end
    otherwise
        error('NetPRS:GenomicNetwork:Solver', 'Unknown solver ''%s''.', Opts.Solver);
end
resid = full(Q * F(:, cols) - Gk(:, cols));
colNorm = sqrt(full(sum(Gk(:, cols) .^ 2, 1)));
maxRelRes = 0;
if ~isempty(cols)
    maxRelRes = max(sqrt(sum(resid .^ 2, 1)) ./ colNorm);
end

%% h_G^{jk} = f^j' g^k and T_G
H = full(F' * Gk);
H = (H + H') / 2;                                    % rounding-level symmetrization
H(1:s+1:end) = 0;                                    % self-interactions excluded
scale = max(abs(H(:)));
if any(H(:) < -1e-8 * max(scale, eps))
    error('NetPRS:GenomicNetwork:Negative', 'Negative h_G beyond rounding error (check the inputs).');
end
H(H < 0) = 0;
Tg = exp(log(H) / Opts.Sigma ^ 2);                   % T_G(0) = 0

%% Thresholding
switch lower(Opts.ThresholdOn)
    case 'transformed'
        V = Tg;
    case 'raw'
        V = H;
    otherwise
        error('NetPRS:GenomicNetwork:ThresholdOn', 'ThresholdOn must be ''transformed'' or ''raw''.');
end
upperMask = triu(true(s), 1);
vals = V(upperMask);
positive = vals(vals > 0);
if ischar(Opts.Threshold)
    switch lower(Opts.Threshold)
        case 'elbow'
            if isempty(positive)
                tau = NaN;
            else
                tau = ElbowThreshold(positive, struct('Scale', Opts.ElbowScale));
            end
        case 'none'
            tau = 0;
        otherwise
            error('NetPRS:GenomicNetwork:Threshold', 'Unknown threshold ''%s''.', Opts.Threshold);
    end
else
    tau = Opts.Threshold;
end
if isempty(positive)
    warning('NetPRS:GenomicNetwork:NoInteraction', ...
        'No SNP pair has a positive genomic interaction (no shared GGI component); W_G is empty.');
end
keep = upperMask & (V >= tau - Opts.TieTol * abs(tau)) & (V > 0);   % NaN tau keeps nothing
Wu = zeros(s, s);
Wu(keep) = Tg(keep);
Wg = Wu + Wu';

Info.H = H;
Info.Tg = Tg;
Info.Threshold = tau;
Info.ThresholdOn = lower(Opts.ThresholdOn);
Info.NumEdge = nnz(keep);
Info.Density = Info.NumEdge / (s * (s - 1) / 2);
Info.NumGeneUsed = mk;
Info.Solver = solver;
Info.MaxRelResidual = maxRelRes;
end

function comp = ConnectedComponents(W)
% Connected-component label of every node (Dulmage-Mendelsohn blocks of the
% symmetric pattern with a zero-free diagonal).
m = size(W, 1);
A = spones(W) + speye(m);
[p, ~, r] = dmperm(A);
blk = zeros(m, 1);
blk(r(1:end-1)) = 1;
blk = cumsum(blk);
comp = zeros(m, 1);
comp(p) = blk;
end
