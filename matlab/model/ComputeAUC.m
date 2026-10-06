function A = ComputeAUC(Score, Label)

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

[xs, order] = sort(x);
N = numel(x);
grp = cumsum([1; diff(xs) ~= 0]);
avg = accumarray(grp, (1:N)') ./ accumarray(grp, 1);
r = zeros(N, 1);
r(order) = avg(grp);
end
