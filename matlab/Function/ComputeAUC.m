function A = ComputeAUC(Score, Label)
%COMPUTEAUC Area under the ROC curve (Mann-Whitney statistic, ties = 1/2).
%
%   A = ComputeAUC(Score, Label)
%
%   A = P(Score_case > Score_control) + 0.5 * P(Score_case = Score_control),
%   computed from mid-ranks in O(N log N). This equals the trapezoidal area
%   under the empirical ROC curve (e.g. perfcurve) without any toolbox.
%
%   Inputs
%     Score  [1 x N] or [N x 1] real scores (higher = more likely a case)
%     Label  [same size] binary labels (1 = case, 0 = control)
%   Output
%     A      scalar in [0, 1]; NaN if one of the classes is absent

Score = double(Score(:));
Label = double(Label(:));
if numel(Score) ~= numel(Label)
    error('NetPRS:ComputeAUC:SizeMismatch', 'Score and Label must have the same number of elements.');
end
if any(Label ~= 0 & Label ~= 1)
    error('NetPRS:ComputeAUC:InvalidLabel', 'Label must be binary (0/1).');
end
if any(isnan(Score))
    error('NetPRS:ComputeAUC:NaNScore', 'Score contains NaN.');
end
nPos = sum(Label == 1);
nNeg = sum(Label == 0);
if nPos == 0 || nNeg == 0
    A = NaN;
    return;
end
midRank = MidRank(Score);
A = (sum(midRank(Label == 1)) - nPos * (nPos + 1) / 2) / (nPos * nNeg);
end

function r = MidRank(x)
% Mid-ranks (average rank within ties), equivalent to tiedrank.
[xs, order] = sort(x);
N = numel(x);
grp = cumsum([1; diff(xs) ~= 0]);                 % tie group in sorted order
avg = accumarray(grp, (1:N)') ./ accumarray(grp, 1);
r = zeros(N, 1);
r(order) = avg(grp);
end
