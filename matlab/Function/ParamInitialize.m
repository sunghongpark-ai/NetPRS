function model = ParamInitialize(model)
%PARAMINITIALIZE Initialize the NetPRS parameters and the ADAM state.
%
%   Parameters (Fig. 2; Sec. III-D)
%     Mu    [K x 1] smoothness mu_* of the K included networks (P, G),
%                   initialized to Param.MuInit (= 1 in the paper)
%     Alpha [E x 1] combining coefficients alpha_* of the E included
%                   effects (I, P, G), initialized to Param.AlphaInit (= 0)
%     Beta  [s x 1] SNP effect size, Glorot-uniform initialization
%                   (2*u-1)*sqrt(6/(s+1)), u ~ U(0,1), as in NeuroFANN/BIGPN/
%                   PPIxGPN (the paper does not report the initialization)
%
%   The parameters are packed into model.WeightParam = [Mu; Alpha; Beta].
%   u is drawn from the portable stream RngStream(Param.Seed + IdxIter, 2),
%   so all folds of one iteration share the same initial beta (as in
%   NeuroFANN/BIGPN) and MATLAB, GNU Octave and Python start from the same
%   values. No global random state is used or changed.

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
