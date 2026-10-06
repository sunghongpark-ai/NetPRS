function model = ParamTraining(model)
%PARAMTRAINING Train NetPRS by gradient descent (ADAM) on Eq. (6).
%
%   The weights after 0, 1, ..., Param.MaxEpoch updates are evaluated; when
%   validation subjects exist, the weights with the minimum validation
%   cross-entropy are kept (first minimum on ties), which is the epoch
%   selection used in NeuroFANN/BIGPN/PPIxGPN. Without validation subjects
%   the weights after exactly Param.MaxEpoch updates are kept.
%
%   Outputs stored in model
%     LossTrainCurve, LossValidCurve, ObjectiveCurve  [(MaxEpoch+1) x 1]
%     BestEpoch     number of updates of the selected weights
%     WeightParam   selected weights; the forward pass is then evaluated
%                   for all subjects (model.Prob, model.Logit)

T = model.Param.MaxEpoch;
model.LossTrainCurve = nan(T + 1, 1);
model.LossValidCurve = nan(T + 1, 1);
model.ObjectiveCurve = nan(T + 1, 1);
hasValid = model.NumValid > 0;

bestLoss = Inf;
bestWeight = model.WeightParam;
bestEpoch = 0;
for epoch = 0:T
    model = ParamReshape(model);
    model = ForwardPropagate(model, 'fit');
    model = LossCalculation(model);
    model.LossTrainCurve(epoch + 1) = model.LossTrain;
    model.LossValidCurve(epoch + 1) = model.LossValid;
    model.ObjectiveCurve(epoch + 1) = model.Objective;
    if ~isfinite(model.Objective)
        error('NetPRS:ParamTraining:NonFinite', ...
            'Non-finite objective at epoch %d (reduce Param.LearnRate).', epoch);
    end
    if hasValid && model.LossValid < bestLoss
        bestLoss = model.LossValid;
        bestWeight = model.WeightParam;
        bestEpoch = epoch;
    end
    if model.Param.Verbose && mod(epoch, 100) == 0
        fprintf('    epoch %4d | train %.4f | valid %.4f | mu %s | theta %s\n', epoch, ...
            model.LossTrain, model.LossValid, sprintf('%.3g ', model.Mu), sprintf('%.3g ', model.Theta));
    end
    if epoch == T
        break;
    end
    model = BackwardPropagate(model);
    model = ParameterUpdate(model);
end
if ~hasValid
    bestWeight = model.WeightParam;
    bestEpoch = T;
end

model.BestEpoch = bestEpoch;
model.WeightParam = bestWeight;
model = ParamReshape(model);
model = ForwardPropagate(model, 'all');
model = LossCalculation(model);
end
