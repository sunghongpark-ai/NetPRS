function Effect = EffectExtraction(model, X)

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
