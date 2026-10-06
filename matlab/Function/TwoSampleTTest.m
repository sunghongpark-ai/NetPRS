function [P, T, DF] = TwoSampleTTest(A, B, Type)
%TWOSAMPLETTEST Row-wise two-sample t-test (two-sided), no toolbox required.
%
%   [P, T, DF] = TwoSampleTTest(A, B)            pooled variance (default;
%                                                as MATLAB ttest2)
%   [P, T, DF] = TwoSampleTTest(A, B, 'welch')   unequal variances
%
%   Row i of A (group 1) is compared with row i of B (group 2).
%   The two-sided P-value of a t statistic with DF degrees of freedom is
%       P = betainc(DF / (DF + T^2), DF/2, 1/2).
%
%   Inputs
%     A  [s x n1], B [s x n2]
%     Type  'pooled' | 'welch'
%   Outputs (s x 1)
%     P  P-values, T  t statistics (group 1 - group 2), DF  degrees of freedom
%   Rows with zero variance in both groups give T = NaN (equal means) or
%   +/-Inf (different means, P = 0).

if nargin < 3 || isempty(Type)
    Type = 'pooled';
end
A = double(A);
B = double(B);
n1 = size(A, 2);
n2 = size(B, 2);
if size(A, 1) ~= size(B, 1)
    error('NetPRS:TwoSampleTTest:SizeMismatch', 'A and B must have the same number of rows.');
end
if n1 < 2 || n2 < 2
    error('NetPRS:TwoSampleTTest:TooFew', 'Each group needs at least two observations.');
end
m1 = mean(A, 2);
m2 = mean(B, 2);
v1 = var(A, 0, 2);
v2 = var(B, 0, 2);
switch lower(Type)
    case 'pooled'
        DF = (n1 + n2 - 2) * ones(size(m1));
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2);
        se = sqrt(sp2 * (1 / n1 + 1 / n2));
    case 'welch'
        a1 = v1 / n1;
        a2 = v2 / n2;
        se = sqrt(a1 + a2);
        DF = (a1 + a2) .^ 2 ./ (a1 .^ 2 / (n1 - 1) + a2 .^ 2 / (n2 - 1));
    otherwise
        error('NetPRS:TwoSampleTTest:Type', 'Type must be ''pooled'' or ''welch''.');
end
T = (m1 - m2) ./ se;
P = nan(size(T));
ok = ~isnan(T) & ~isnan(DF);
P(ok) = betainc(DF(ok) ./ (DF(ok) + T(ok) .^ 2), DF(ok) / 2, 0.5);
P(isinf(T)) = 0;
end
