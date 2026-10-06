function [Result, model] = TrainNetPRS(ModelInit, Selector)

model = DataIndexing(ModelInit, Selector);
model = ParamInitialize(model);
model = ParamTraining(model);

Result = struct();
Result.IdxModel = model.IdxModel;
Result.IdxIter = model.IdxIter;
Result.FoldValid = model.FoldValid;
Result.FoldTest = model.FoldTest;
Result.BestEpoch = model.BestEpoch;
Result.EffectCode = model.EffectCode;
Result.Mu = model.Mu;
Result.Alpha = model.Alpha;
Result.Theta = model.Theta;
Result.Beta = model.Beta;
Result.Omega = model.Omega;
Result.IdxTest = model.IdxTest;
Result.ProbTest = model.Prob(model.IdxTest);
Result.AUCTrain = SafeAUC(model.Prob(model.IdxTrain), model.YAll(model.IdxTrain));
Result.AUCValid = SafeAUC(model.Prob(model.IdxValid), model.YAll(model.IdxValid));
Result.AUCTest = SafeAUC(model.Prob(model.IdxTest), model.YAll(model.IdxTest));
Result.LossTrainCurve = model.LossTrainCurve;
Result.LossValidCurve = model.LossValidCurve;
end

function A = SafeAUC(score, label)
if isempty(score)
    A = NaN;
else
    A = ComputeAUC(score, label);
end
end
