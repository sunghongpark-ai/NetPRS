function model = ForwardPropagate(model, Scope)
%FORWARDPROPAGATE NetPRS forward pass: propagation, integration, risk.
%
%   model = ForwardPropagate(model)          subjects used for fitting
%   model = ForwardPropagate(model, 'all')   all subjects (incl. test)
%
%   Paper equations
%     (4)  F_* = (I + mu_* L_*)^{-1} X                     (* = P, G)
%     (5)  Z   = theta_I F_I + theta_P F_P + theta_G F_G,  F_I = X
%          P   = 1 ./ (1 + exp(-beta' Z))
%
%   Implementation. Because Q_* = I + mu_* L_* is symmetric,
%       beta' F_* = beta' Q_*^{-1} X = (Q_*^{-1} beta)' X,
%   hence beta' Z = Omega' X with the effective SNP weight vector
%       Omega = theta_I beta + theta_P Q_P^{-1} beta + theta_G Q_G^{-1} beta.
%   The risk is therefore evaluated exactly without forming the s x n
%   matrices F_* and Z (O(s^2 + s n) instead of O(s^2 n) per epoch).
%   EffectExtraction returns F_* and Z explicitly when they are needed.
%
%   Outputs stored in model
%     Weff   [s x E]   column e = Q_e^{-1} beta (beta for the effect I)
%     Omega  [s x 1]   effective SNP weights, Weff * Theta
%     Logit  [1 x NumAll] beta' Z (NaN outside Scope)
%     Prob   [1 x NumAll] predicted risk P (NaN outside Scope)

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
