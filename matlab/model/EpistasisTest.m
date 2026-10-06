function Epi = EpistasisTest(X, Y, Opts)

if nargin < 3 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'Pairs', []);
Opts = SetDefaultField(Opts, 'Cov', []);
Opts = SetDefaultField(Opts, 'ChunkSize', 2000);
Opts = SetDefaultField(Opts, 'Verbose', false);

[s, n] = size(X);
Y = double(Y(:));
if numel(Y) ~= n
    error('NetPRS:EpistasisTest:SizeMismatch', 'Y must have one entry per column of X.');
end
if isempty(Opts.Pairs)
    [kk, jj] = find(triu(true(s), 1)');
    pairs = [jj, kk];
else
    pairs = Opts.Pairs;
    if size(pairs, 2) ~= 2 || any(pairs(:) < 1 | pairs(:) > s | pairs(:) ~= fix(pairs(:))) || any(pairs(:, 1) == pairs(:, 2))
        error('NetPRS:EpistasisTest:Pairs', 'Opts.Pairs must be a B x 2 list of distinct SNP indices.');
    end
end
cov = Opts.Cov;
if isempty(cov)
    cov = zeros(n, 0);
end
C = [ones(n, 1), double(cov)];
Xt = double(X)';

numPair = size(pairs, 1);
b3 = nan(numPair, 1);
se3 = nan(numPair, 1);
nobs = zeros(numPair, 1);
for first = 1:Opts.ChunkSize:numPair
    idx = first:min(first + Opts.ChunkSize - 1, numPair);
    A = Xt(:, pairs(idx, 1));
    Bm = Xt(:, pairs(idx, 2));
    V = cat(3, A, Bm, A .* Bm);
    Mask = isfinite(A) & isfinite(Bm);
    Out = LogisticBatch(Y, C, V, Mask, Opts);
    b3(idx) = Out.Coef(end, :)';
    se3(idx) = Out.SE(end, :)';
    nobs(idx) = Out.NumObs';
    if Opts.Verbose
        fprintf('  epistasis: %d / %d pairs\n', idx(end), numPair);
    end
end
stat = (b3 ./ se3) .^ 2;

Epi.Pairs = pairs;
Epi.PairBeta3 = b3;
Epi.PairSE3 = se3;
Epi.PairORint = exp(b3);
Epi.PairStat = stat;
Epi.PairP = erfc(sqrt(stat / 2));
Epi.Beta3 = PairsToMatrix(pairs, b3, s);
Epi.SE3 = PairsToMatrix(pairs, se3, s);
Epi.ORint = PairsToMatrix(pairs, Epi.PairORint, s);
Epi.Stat = PairsToMatrix(pairs, stat, s);
Epi.P = PairsToMatrix(pairs, Epi.PairP, s);
Epi.NumObs = PairsToMatrix(pairs, nobs, s);
end

function A = PairsToMatrix(pairs, v, s)
A = nan(s, s);
A(sub2ind([s, s], pairs(:, 1), pairs(:, 2))) = v;
A(sub2ind([s, s], pairs(:, 2), pairs(:, 1))) = v;
end
