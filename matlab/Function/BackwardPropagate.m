function model = BackwardPropagate(model)
%BACKWARDPROPAGATE Analytic gradients of the NetPRS objective (Sec. III-D).
%
%   With n training subjects, P-Y = r (1 x n) and Q_* = I + mu_* L_*:
%
%   beta   : grad = (1/n) Z (P - Y)' + 2 delta beta
%   alpha_*: grad = (theta_*/n) Tr((F_* - Z)' beta (P - Y)) + 2 delta alpha_*
%            using Eq. (7)-(8): dZ/dalpha_i = theta_i (F_i - Z)
%   mu_*   : grad = -(theta_*/n) Tr((Q_*^{-1} L_* Q_*^{-1} X)' beta (P - Y))
%                   + 2 delta mu_*,   using Eq. (9)-(10)
%
%   Exact low-cost evaluation (Q_* and L_* symmetric), with c = X (P - Y)':
%     Z (P - Y)'                        = sum_e theta_e Q_e^{-1} c
%     Tr((F_e - Z)' beta (P - Y))       = (Weff_e - Omega)' c
%     Tr((Q^{-1} L Q^{-1} X)' beta (P-Y)) = (Q^{-1} c)' L (Q^{-1} beta)
%   where Weff_e = Q_e^{-1} beta and Omega = sum_e theta_e Weff_e come from
%   ForwardPropagate. Each identity is checked against the literal trace
%   formulas and against finite differences in Test/RunAllTests.m.
%
%   Output: model.Gradient = [grad_mu; grad_alpha; grad_beta]

n = model.NumTrain;
delta = model.Param.RegCoeff;
r = model.Prob(model.IdxTrain) - model.YAll(model.IdxTrain);   % 1 x n
c = model.XUse(:, model.IdxTrain) * r';                         % s x 1

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
