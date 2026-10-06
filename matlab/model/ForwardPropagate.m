function model = ForwardPropagate(model, Scope)

if nargin < 2 || isempty(Scope)
    Scope = 'fit';
end
switch lower(Scope)
    case 'fit'
        cols = model.IdxFit;
    case 'all'
        cols = 1:model.NumAll;
    otherwise
        error('NetPRS:ForwardPropagate:Scope', 'Scope must be ''fit'' or ''all''.');
end

E = model.NumEffect;
Weff = zeros(model.NumSNP, E);
netEffect = [model.Net.Effect];
for e = 1:E
    eff = model.EffectList(e);
    if eff == 1
        Weff(:, e) = model.Beta;
    else
        k = find(netEffect == eff, 1);
        Weff(:, e) = PropagationSolve(model.Net(k), model.Beta);
    end
end
model.Weff = Weff;
model.Omega = Weff * model.Theta;

model.Logit = nan(1, model.NumAll);
model.Logit(cols) = model.Omega' * model.XUse(:, cols);
model.Prob = 1 ./ (1 + exp(-model.Logit));
end
