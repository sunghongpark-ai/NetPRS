function [P, Logit] = RiskPredict(model, X)
%RISKPREDICT Individual AD risk of new subjects with a trained NetPRS model.
%
%   [P, Logit] = RiskPredict(model, X)
%
%   Inputs
%     model  trained model (output of ParamTraining, or the second output of
%            TrainNetPRS)
%     X      [s x m] genotype dosage of m subjects, same SNP order as XData
%   Outputs
%     P      [1 x m] predicted risk 1 ./ (1 + exp(-beta' Z))
%     Logit  [1 x m] beta' Z
%
%   The scaling fitted on the training subjects (Param.XScaling) is applied
%   first; beta' Z is evaluated as Omega' X (see ForwardPropagate).

if ~isfield(model, 'Omega') || isempty(model.Omega)
    error('NetPRS:RiskPredict:NotTrained', 'The model has not been trained (Omega is missing).');
end
if size(X, 1) ~= model.NumSNP
    error('NetPRS:RiskPredict:SizeMismatch', 'X must have %d rows (SNPs).', model.NumSNP);
end
if any(~isfinite(X(:)))
    error('NetPRS:RiskPredict:MissingGenotype', 'X contains NaN/Inf; impute missing genotypes first.');
end
Xs = (double(X) - model.XShift) ./ model.XScale;
Logit = model.Omega' * Xs;
P = 1 ./ (1 + exp(-Logit));
end
