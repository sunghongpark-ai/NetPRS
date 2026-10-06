function Effect = EffectExtraction(model, X)
%EFFECTEXTRACTION Independent, interactive and integrated SNP effects.
%
%   Effect = EffectExtraction(model, X) returns, for the subjects in X
%   (columns), the matrices of Sec. III-A/B for the current parameters:
%     Effect.FI  [s x m]  independent effect F_I = X (after Param.XScaling)
%     Effect.FP  [s x m]  phenotypic interactive effect (I + mu_P L_P)^{-1} X
%     Effect.FG  [s x m]  genomic interactive effect    (I + mu_G L_G)^{-1} X
%     Effect.Z   [s x m]  integrated effect Z = sum_* theta_* F_*  (Eq. (5))
%     Effect.Theta, Effect.Mu, Effect.EffectCode
%   Effects not included in model.EffectCode are returned empty.
%
%   If X is omitted, all subjects of the model (model.XAll) are used.
%   Cost O(s^2 m); used for interpretation (Fig. 4), not for training.

if nargin < 2 || isempty(X)
    X = model.XAll;
end
if size(X, 1) ~= model.NumSNP
    error('NetPRS:EffectExtraction:SizeMismatch', 'X must have %d rows (SNPs).', model.NumSNP);
end
model = ParamReshape(model);
Xs = (double(X) - model.XShift) ./ model.XScale;

Effect = struct('FI', [], 'FP', [], 'FG', [], 'Z', zeros(size(Xs)), ...
    'Theta', model.Theta, 'Mu', model.Mu, 'EffectCode', model.EffectCode);
names = {'FI', 'FP', 'FG'};
netEffect = [model.Net.Effect];
for e = 1:model.NumEffect
    eff = model.EffectList(e);
    if eff == 1
        F = Xs;
    else
        k = find(netEffect == eff, 1);
        F = PropagationSolve(model.Net(k), Xs);
    end
    Effect.(names{eff}) = F;
    Effect.Z = Effect.Z + model.Theta(e) * F;
end
end
