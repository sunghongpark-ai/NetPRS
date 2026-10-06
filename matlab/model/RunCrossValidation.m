function Results = RunCrossValidation(ModelInit, NumWorkers)

if nargin < 2 || isempty(NumWorkers)
    NumWorkers = 0;
end
N = ModelInit.NumModel;
Results = cell(N, 1);
if NumWorkers > 0 && exist('gcp', 'file') == 2 && exist('parpool', 'file') == 2
    if isempty(gcp('nocreate'))
        parpool(NumWorkers);
    end
end
verbose = ModelInit.Param.Verbose;
parfor (IdxModel = 1:N, NumWorkers)
    r = TrainNetPRS(ModelInit, IdxModel);
    if verbose
        fprintf('  model %3d/%d (iter %d): best epoch %4d | valid AUC %.4f | test AUC %.4f\n', ...
            IdxModel, N, r.IdxIter, r.BestEpoch, r.AUCValid, r.AUCTest);
    end
    Results{IdxModel} = r;
end
end
