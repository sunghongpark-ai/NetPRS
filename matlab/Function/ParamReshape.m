function model = ParamReshape(model)
%PARAMRESHAPE Unpack model.WeightParam and prepare the propagation operators.
%
%   Mu, Alpha and Beta are read from model.WeightParam = [Mu; Alpha; Beta].
%   The combining ratio is the softmax of Alpha (Sec. III-B),
%       theta_* = exp(alpha_*) / sum_k exp(alpha_k),
%   evaluated after subtracting max(Alpha) (an exact, overflow-free form).
%
%   For every included network the operator Q_*^{-1} = (I + mu_* L_*)^{-1}
%   of Eq. (4) is prepared:
%     Solver 'eig'  : D_* = 1 ./ (1 + mu_* * lambda_*), with L_* = V diag(lambda) V'
%     Solver 'chol' : R_* = chol(I + mu_* L_*)
%   Q_* must be positive definite; this holds for every mu_* >= 0 and fails
%   only when mu_* <= -1/lambda_max (prevented by the Param.MuMin projection).

w = model.WeightParam;
model.Mu = w(model.IdxMu);
model.Alpha = w(model.IdxAlpha);
model.Beta = w(model.IdxBeta);

a = model.Alpha - max(model.Alpha);
ea = exp(a);
model.Theta = ea / sum(ea);

s = model.NumSNP;
useEig = strcmpi(model.Param.Solver, 'eig');
for k = 1:model.NumNet
    mu = model.Mu(k);
    if useEig
        den = 1 + mu * model.Net(k).Lambda;
        if any(den <= 0)
            error('NetPRS:ParamReshape:NotPositiveDefinite', ...
                'I + mu*L of network %s is not positive definite (mu = %g).', model.Net(k).Name, mu);
        end
        model.Net(k).D = 1 ./ den;
        model.Net(k).R = [];
    else
        [R, p] = chol(eye(s) + mu * model.Net(k).L);
        if p ~= 0
            error('NetPRS:ParamReshape:NotPositiveDefinite', ...
                'I + mu*L of network %s is not positive definite (mu = %g).', model.Net(k).Name, mu);
        end
        model.Net(k).R = R;
        model.Net(k).D = [];
    end
end
end
