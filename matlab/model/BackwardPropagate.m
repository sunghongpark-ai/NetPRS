function model = BackwardPropagate(model)

n = model.NumTrain;
delta = model.Param.RegCoeff;
r = model.Prob(model.IdxTrain) - model.YAll(model.IdxTrain);
c = model.XUse(:, model.IdxTrain) * r';

E = model.NumEffect;
netEffect = [model.Net.Effect];
QinvC = zeros(model.NumSNP, E);
for e = 1:E
    eff = model.EffectList(e);
    if eff == 1
        QinvC(:, e) = c;
    else
        k = find(netEffect == eff, 1);
        QinvC(:, e) = PropagationSolve(model.Net(k), c);
    end
end

gBeta = (QinvC * model.Theta) / n + 2 * delta * model.Beta;
gAlpha = model.Theta .* ((model.Weff - model.Omega)' * c) / n + 2 * delta * model.Alpha;

gMu = zeros(model.NumNet, 1);
for k = 1:model.NumNet
    e = find(model.EffectList == model.Net(k).Effect, 1);
    quadForm = QinvC(:, e)' * (model.Net(k).L * model.Weff(:, e));
    gMu(k) = -model.Theta(e) / n * quadForm + 2 * delta * model.Mu(k);
end

model.Gradient = [gMu; gAlpha; gBeta];
end
