function model = ParameterUpdate(model)

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
