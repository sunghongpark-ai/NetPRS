function [P, D] = KSTest2Asymptotic(X1, X2)

x1 = X1(:);
x2 = X2(:);
x1 = x1(~isnan(x1));
x2 = x2(~isnan(x2));
n1 = numel(x1);
n2 = numel(x2);
if n1 == 0 || n2 == 0
    P = NaN;
    D = NaN;
    return;
end
v = [x1; x2];
lab = [ones(n1, 1); zeros(n2, 1)];
[vs, order] = sort(v);
ls = lab(order);
cdfDiff = cumsum(ls / n1 - (1 - ls) / n2);
lastOfTie = [find(diff(vs) ~= 0); numel(vs)];
D = max(abs(cdfDiff(lastOfTie)));
ne = n1 * n2 / (n1 + n2);
lambda = max((sqrt(ne) + 0.12 + 0.11 / sqrt(ne)) * D, 0);
j = (1:101)';
P = 2 * sum((-1) .^ (j - 1) .* exp(-2 * lambda ^ 2 * j .^ 2));
P = min(max(P, 0), 1);
end
