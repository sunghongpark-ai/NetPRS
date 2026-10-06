function model = ParameterUpdate(model)
%PARAMETERUPDATE One ADAM step on model.WeightParam (Kingma & Ba, 2014).
%
%   m = b1 m + (1-b1) g,  v = b2 v + (1-b2) g.^2,
%   w = w - lr * (m/(1-b1^t)) ./ (sqrt(v/(1-b2^t)) + eps)
%   (identical to ParameterUpdate of BIGPN / WeightUpdate of PPIxGPN).
%
%   Safeguard (declared, not stated in the paper): afterwards every mu_* is
%   projected onto [Param.MuMin, Inf) (default MuMin = 0). mu_* is the
%   smoothness weight of the GSSL objective min (F-X)'(F-X) + mu F'LF, which
%   is a smoothness penalty only for mu >= 0; the projection also keeps
%   I + mu L positive definite. Set Param.MuMin = -Inf to disable it.

A = model.AdamParam;
g = model.Gradient;
A.t = A.t + 1;
A.m = A.beta1 * A.m + (1 - A.beta1) * g;
A.v = A.beta2 * A.v + (1 - A.beta2) * (g .^ 2);
mHat = A.m / (1 - A.beta1 ^ A.t);
vHat = A.v / (1 - A.beta2 ^ A.t);
model.WeightParam = model.WeightParam - A.alpha * mHat ./ (sqrt(vHat) + A.epsilon);
model.AdamParam = A;

muMin = model.Param.MuMin;
if model.NumNet > 0 && isfinite(muMin)
    model.WeightParam(model.IdxMu) = max(model.WeightParam(model.IdxMu), muMin);
end
end
