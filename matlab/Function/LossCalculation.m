function model = LossCalculation(model)
%LOSSCALCULATION Cross-entropy loss and the NetPRS objective of Eq. (6).
%
%   L = -(1/n) * ( Y log P' + (1 - Y) log(1 - P)' )              (Sec. III-D)
%   Objective = L_train + delta * R,
%   R = ||beta||^2 + sum alpha_*^2 + sum mu_*^2  (L2 regularizer whose
%   gradient 2*delta*(.) appears in every gradient of Sec. III-D).
%
%   The cross-entropy is evaluated in the algebraically identical form
%       -[y log s(t) + (1-y) log(1-s(t))] = softplus(t) - y t,  t = beta' Z,
%   softplus(t) = max(t,0) + log(1 + exp(-|t|)), which is exact and does not
%   overflow or take log(0) for large |t|.
%
%   Outputs stored in model: LossTrain, LossValid (NaN when no validation
%   subjects), Penalty (R), Objective.

t = model.Logit;
y = model.YAll;
model.LossTrain = CrossEntropy(t(model.IdxTrain), y(model.IdxTrain));
if model.NumValid > 0
    model.LossValid = CrossEntropy(t(model.IdxValid), y(model.IdxValid));
else
    model.LossValid = NaN;
end
model.Penalty = sum(model.Beta .^ 2) + sum(model.Alpha .^ 2) + sum(model.Mu .^ 2);
model.Objective = model.LossTrain + model.Param.RegCoeff * model.Penalty;
end

function L = CrossEntropy(t, y)
if isempty(t)
    L = NaN;
    return;
end
softplus = max(t, 0) + log1p(exp(-abs(t)));
L = mean(softplus - y .* t);
end
