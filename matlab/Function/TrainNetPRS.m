function [Result, model] = TrainNetPRS(ModelInit, Selector)
%TRAINNETPRS Train and evaluate one NetPRS model.
%
%   [Result, model] = TrainNetPRS(ModelInit, IdxModel)
%       trains the model defined by row IdxModel of ModelInit.CVlist.
%   [Result, model] = TrainNetPRS(ModelInit, Custom)
%       uses a custom split (struct with IdxTrain, IdxValid, IdxTest, IdxIter;
%       see DataIndexing).
%
%   Result (compact struct suitable for parfor collection)
%     .IdxModel .IdxIter .FoldValid .FoldTest .BestEpoch
%     .Mu .Alpha .Theta .Beta .Omega          trained parameters
%     .IdxTest  [1 x t]  test subjects (columns of ModelInit.XAll)
%     .ProbTest [1 x t]  predicted risk of the test subjects
%     .AUCTrain .AUCValid .AUCTest
%     .LossTrainCurve .LossValidCurve
%   model: full trained model (for RiskPredict / EffectExtraction).

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
