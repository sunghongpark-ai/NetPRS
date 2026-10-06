function Results = RunCrossValidation(ModelInit, NumWorkers)
%RUNCROSSVALIDATION Train every NetPRS model listed in ModelInit.CVlist.
%
%   Results = RunCrossValidation(ModelInit)
%   Results = RunCrossValidation(ModelInit, NumWorkers)
%
%   NumWorkers = 0 (default) runs the models one after another. With
%   NumWorkers > 0 the models are trained in a parfor loop (MATLAB Parallel
%   Computing Toolbox; a pool with NumWorkers workers is opened if none is
%   running). Each model is independent and seeded by its iteration, so the
%   results do not depend on NumWorkers.
%
%   Output: Results {NumModel x 1} cell of TrainNetPRS result structs.

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
