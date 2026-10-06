function model = LossCalculation(model)

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
