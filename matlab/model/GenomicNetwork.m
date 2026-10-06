function [Wg, Info] = GenomicNetwork(G, Wggi, Opts)

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

Wggi = sparse(double(Wggi));
G = sparse(double(G));
comp = ConnectedComponents(Wggi);
related = any(G ~= 0, 2);
keepGene = ismember(comp, unique(comp(related)));
Gk = G(keepGene, :);
Wk = Wggi(keepGene, keepGene);
mk = nnz(keepGene);

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
    solver = 'none';
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
        precond = @(v) v ./ dq;
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

H = full(F' * Gk);
H = (H + H') / 2;
H(1:s+1:end) = 0;
scale = max(abs(H(:)));
if any(H(:) < -1e-8 * max(scale, eps))
    error('NetPRS:GenomicNetwork:Negative', 'Negative h_G beyond rounding error (check the inputs).');
end
H(H < 0) = 0;
Tg = exp(log(H) / Opts.Sigma ^ 2);

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
keep = upperMask & (V >= tau - Opts.TieTol * abs(tau)) & (V > 0);
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

m = size(W, 1);
A = spones(W) + speye(m);
[p, ~, r] = dmperm(A);
blk = zeros(m, 1);
blk(r(1:end-1)) = 1;
blk = cumsum(blk);
comp = zeros(m, 1);
comp(p) = blk;
end
