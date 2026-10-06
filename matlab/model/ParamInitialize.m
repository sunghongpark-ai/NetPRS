function model = ParamInitialize(model)

K = model.NumNet;
E = model.NumEffect;
s = model.NumSNP;

Mu = model.Param.MuInit * ones(K, 1);
Alpha = model.Param.AlphaInit * ones(E, 1);
sizeBeta = [s, 1];
u = RngUniform(RngStream(model.Param.Seed + model.IdxIter, 2), s);
Beta = (2 * u - 1) * sqrt(6 / sum(sizeBeta));

model.IdxMu = 1:K;
model.IdxAlpha = K + (1:E);
model.IdxBeta = K + E + (1:s);
model.SizeParam = [K, 1; E, 1; sizeBeta];
model.NumParam = K + E + s;
model.WeightParam = [Mu; Alpha; Beta];
model.AdamParam = AdamInitialize(model.NumParam, model.Param.LearnRate);
end
