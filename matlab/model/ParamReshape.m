function model = ParamReshape(model)

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
