function Out = LogisticBatch(y, C, V, M, Opts)

if nargin < 5 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'MaxIter', 25);
Opts = SetDefaultField(Opts, 'Tol', 1e-8);
Opts = SetDefaultField(Opts, 'PivotTol', 1e-10);

y = double(y(:));
n = numel(y);
if size(C, 1) ~= n || size(V, 1) ~= n
    error('NetPRS:LogisticBatch:SizeMismatch', 'y, C and V must have the same number of rows.');
end
q = size(C, 2);
B = size(V, 2);
r = size(V, 3);
p = q + r;
if nargin < 4 || isempty(M)
    M = true(n, B);
end
if ~isequal(size(M), [n, B])
    error('NetPRS:LogisticBatch:MaskSize', 'M must be n x B.');
end

rowOK = isfinite(y) & all(isfinite(C), 2);
M = logical(M) & repmat(rowOK, 1, B) & all(isfinite(V), 3);
y(~rowOK) = 0;
C(~rowOK, :) = 0;
V(~isfinite(V)) = 0;
V(repmat(~M, [1, 1, r])) = 0;
Mf = double(M);

coef = zeros(p, B);
converged = false(1, B);
failed = false(1, B);
numIter = zeros(1, B);
active = 1:B;
for iter = 1:Opts.MaxIter
    if isempty(active)
        break;
    end
    [g, H] = ScoreHessian(y, C, V(:, active, :), Mf(:, active), coef(:, active));
    [L, ok] = BatchChol(H, Opts.PivotTol);
    newtonStep = zeros(p, numel(active));
    if any(ok)
        newtonStep(:, ok) = BatchSolve(L(:, :, ok), g(:, ok));
    end
    coef(:, active) = coef(:, active) + newtonStep;
    numIter(active) = iter;
    tolAbs = Opts.Tol * (1 + max(abs(coef(:, active)), [], 1));
    done = ok & (max(abs(newtonStep), [], 1) <= tolAbs) & all(isfinite(coef(:, active)), 1);
    converged(active(done)) = true;
    failed(active(~ok)) = true;
    active = active(~done & ok);
end

SE = nan(p, B);
idx = find(converged);
if ~isempty(idx)
    [~, H] = ScoreHessian(y, C, V(:, idx, :), Mf(:, idx), coef(:, idx));
    [L, ok] = BatchChol(H, Opts.PivotTol);
    se = nan(p, numel(idx));
    if any(ok)
        se(:, ok) = sqrt(BatchInvDiag(L(:, :, ok)));
    end
    SE(:, idx) = se;
    converged(idx(~ok)) = false;
end
coef(:, ~converged) = NaN;
SE(:, ~converged) = NaN;

Out.Coef = coef;
Out.SE = SE;
Out.Converged = converged;
Out.NumObs = sum(M, 1);
Out.NumIter = numIter;
Out.Failed = failed | ~converged;
end

function [g, H] = ScoreHessian(y, C, V, Mf, coef)

q = size(C, 2);
r = size(V, 3);
p = q + r;
B = size(V, 2);
eta = zeros(size(Mf));
for a = 1:p
    eta = eta + Column(C, V, a, q) .* coef(a, :);
end
mu = 1 ./ (1 + exp(-eta));
w = mu .* (1 - mu) .* Mf;
res = (y - mu) .* Mf;
g = zeros(p, B);
H = zeros(p, p, B);
for a = 1:p
    xa = Column(C, V, a, q);
    g(a, :) = sum(xa .* res, 1);
    wxa = w .* xa;
    for b = 1:a
        hab = sum(wxa .* Column(C, V, b, q), 1);
        H(a, b, :) = reshape(hab, 1, 1, B);
        H(b, a, :) = reshape(hab, 1, 1, B);
    end
end
end

function x = Column(C, V, a, q)
if a <= q
    x = C(:, a);
else
    x = V(:, :, a - q);
end
end

function [L, ok] = BatchChol(H, pivotTol)

p = size(H, 1);
B = size(H, 3);
L = zeros(p, p, B);
ok = true(1, B);
for j = 1:p
    d = H(j, j, :) - sum(L(j, 1:j-1, :) .^ 2, 2);
    bad = ~(d > pivotTol * max(abs(H(j, j, :)), realmin));
    ok = ok & ~reshape(bad, 1, B);
    d(bad) = 1;
    L(j, j, :) = sqrt(d);
    for i = j+1:p
        L(i, j, :) = (H(i, j, :) - sum(L(i, 1:j-1, :) .* L(j, 1:j-1, :), 2)) ./ L(j, j, :);
    end
end
end

function x = BatchSolve(L, b)

p = size(L, 1);
B = size(L, 3);
z = zeros(p, B);
for i = 1:p
    acc = b(i, :);
    for k = 1:i-1
        acc = acc - reshape(L(i, k, :), 1, B) .* z(k, :);
    end
    z(i, :) = acc ./ reshape(L(i, i, :), 1, B);
end
x = zeros(p, B);
for i = p:-1:1
    acc = z(i, :);
    for k = i+1:p
        acc = acc - reshape(L(k, i, :), 1, B) .* x(k, :);
    end
    x(i, :) = acc ./ reshape(L(i, i, :), 1, B);
end
end

function v = BatchInvDiag(L)

p = size(L, 1);
B = size(L, 3);
Linv = zeros(p, p, B);
for j = 1:p
    Linv(j, j, :) = 1 ./ L(j, j, :);
    for i = j+1:p
        acc = zeros(1, 1, B);
        for k = j:i-1
            acc = acc + L(i, k, :) .* Linv(k, j, :);
        end
        Linv(i, j, :) = -acc ./ L(i, i, :);
    end
end
v = reshape(sum(Linv .^ 2, 1), p, B);
end
