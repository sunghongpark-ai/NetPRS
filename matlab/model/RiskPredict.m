function [P, Logit] = RiskPredict(model, X)

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
