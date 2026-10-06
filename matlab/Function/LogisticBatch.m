function Out = LogisticBatch(y, C, V, M, Opts)
%LOGISTICBATCH Many independent logistic regressions sharing one response.
%
%   Out = LogisticBatch(y, C, V, M, Opts) fits, for b = 1..B,
%       logit Pr(y = 1) = C * a_b + squeeze(V(:, b, :)) * c_b
%   by Newton-Raphson (IRLS) and returns Wald standard errors. All B
%   problems are solved simultaneously with vectorized arithmetic and a
%   batched Cholesky factorization of the p x p Hessians (p = q + r).
%   It is the engine of GWASLogistic (r = 1: dosage) and EpistasisTest
%   (r = 3: S^j, S^k, S^j S^k; Eq. (1)).
%
%   Inputs
%     y     [n x 1] binary response (0/1)
%     C     [n x q] covariates shared by all problems; include the
%                   intercept column ones(n,1) explicitly
%     V     [n x B x r] problem-specific covariates
%     M     [n x B] logical mask of the observations used by each problem
%                   ([] = all). Observations with a non-finite value in y,
%                   C or V(:,b,:) are always excluded (complete cases).
%     Opts  .MaxIter  maximum Newton iterations       (default 25)
%           .Tol      convergence: max|step| <= Tol*(1 + max|coef|)
%                                                     (default 1e-8)
%           .PivotTol relative pivot tolerance of the Cholesky factor;
%                     a smaller pivot marks the problem as singular
%                     (non-identifiable coefficient)  (default 1e-10)
%
%   Output (p = q + r; coefficients ordered as [C, V(:,b,1..r)])
%     Out.Coef      [p x B]  maximum-likelihood estimates
%     Out.SE        [p x B]  sqrt(diag(inv(Hessian))) at the estimate
%     Out.Converged [1 x B]  logical
%     Out.NumObs    [1 x B]  complete observations used
%     Out.NumIter   [1 x B]  Newton iterations performed
%   Problems that are singular or do not converge (e.g. separation) are
%   reported with Converged = false and NaN estimates, the analogue of the
%   'NA' entries written by PLINK.

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

% Complete cases, then zero-fill excluded entries so that 0*NaN never occurs.
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

% Standard errors from the Hessian at the final estimates.
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

%% ------------------------------------------------------------------------
function [g, H] = ScoreHessian(y, C, V, Mf, coef)
% Score vector g [p x B] and observed information H [p x p x B].
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
    x = C(:, a);             % n x 1, broadcast over problems
else
    x = V(:, :, a - q);      % n x B
end
end

function [L, ok] = BatchChol(H, pivotTol)
% Lower Cholesky factors of B symmetric p x p matrices (H = L L').
p = size(H, 1);
B = size(H, 3);
L = zeros(p, p, B);
ok = true(1, B);
for j = 1:p
    d = H(j, j, :) - sum(L(j, 1:j-1, :) .^ 2, 2);
    bad = ~(d > pivotTol * max(abs(H(j, j, :)), realmin));
    ok = ok & ~reshape(bad, 1, B);
    d(bad) = 1;                                   % placeholder, problem flagged
    L(j, j, :) = sqrt(d);
    for i = j+1:p
        L(i, j, :) = (H(i, j, :) - sum(L(i, 1:j-1, :) .* L(j, 1:j-1, :), 2)) ./ L(j, j, :);
    end
end
end

function x = BatchSolve(L, b)
% Solve (L L') x = b for every problem; L [p x p x B], b [p x B].
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
% diag((L L')^{-1}) = column sums of (L^{-1}).^2 for every problem.
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
