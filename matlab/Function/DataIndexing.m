function model = DataIndexing(model, Selector)
%DATAINDEXING Training/validation/test subject indices for one NetPRS model.
%
%   model = DataIndexing(model, IdxModel) selects row IdxModel of
%   model.CVlist and sets the column indices (into model.XAll) of the
%   training, validation and test subjects, together with the (optionally
%   scaled) genotype matrix model.XUse used by the model.
%
%     External mode : CVlist row = [iter, validFold]
%                     train = discovery subjects outside validFold
%                     valid = discovery subjects in validFold
%                     test  = external validation cohort
%     Nested mode   : CVlist row = [iter, testFold, validFold]
%                     train = discovery subjects outside both folds
%
%   model = DataIndexing(model, Custom) uses a user-defined split given as a
%   struct with fields IdxTrain, IdxValid (may be empty), IdxTest (may be
%   empty) and IdxIter (seed offset for the parameter initialization). An
%   empty IdxValid trains for exactly Param.MaxEpoch steps (no epoch
%   selection), e.g. for the final model fitted on the whole discovery cohort.
%
%   Scaling statistics ('center'/'zscore') are estimated on the training
%   subjects only, so no information leaks from validation/test subjects.

if isstruct(Selector)
    req = {'IdxTrain', 'IdxValid', 'IdxTest', 'IdxIter'};
    for i = 1:numel(req)
        if ~isfield(Selector, req{i})
            error('NetPRS:DataIndexing:Custom', 'Custom split requires the field %s.', req{i});
        end
    end
    allIdx = [Selector.IdxTrain(:); Selector.IdxValid(:); Selector.IdxTest(:)];
    if any(allIdx < 1 | allIdx > model.NumAll | allIdx ~= fix(allIdx))
        error('NetPRS:DataIndexing:Custom', 'Custom indices must be integers in [1, %d].', model.NumAll);
    end
    fitIdx = [Selector.IdxTrain(:); Selector.IdxValid(:)];
    if numel(unique(fitIdx)) ~= numel(fitIdx) || any(ismember(Selector.IdxTest(:), fitIdx))
        error('NetPRS:DataIndexing:Custom', 'Training, validation and test subjects must be disjoint.');
    end
    model.IdxModel = 0;
    model.IdxIter = Selector.IdxIter;
    model.FoldValid = [];
    model.FoldTest = [];
    model = SetSubjects(model, Selector.IdxTrain, Selector.IdxValid, Selector.IdxTest);
    return;
end

IdxModel = Selector;
if ~isscalar(IdxModel) || IdxModel < 1 || IdxModel > model.NumModel || IdxModel ~= fix(IdxModel)
    error('NetPRS:DataIndexing:InvalidIndex', 'IdxModel must be an integer in [1, %d].', model.NumModel);
end
row = model.CVlist(IdxModel, :);
model.IdxModel = IdxModel;
model.IdxIter = row(1);
fold = model.CVdata(model.IdxIter, :);
switch model.CVMode
    case 'external'
        model.FoldValid = row(2);
        model.FoldTest = [];
        idxTrain = find(fold ~= model.FoldValid);
        idxValid = find(fold == model.FoldValid);
        idxTest = model.NumDisc + (1:model.NumExt);
    case 'nested'
        model.FoldTest = row(2);
        model.FoldValid = row(3);
        idxTrain = find(fold ~= model.FoldTest & fold ~= model.FoldValid);
        idxValid = find(fold == model.FoldValid);
        idxTest = find(fold == model.FoldTest);
    otherwise
        error('NetPRS:DataIndexing:CVMode', 'Unknown CV mode ''%s''.', model.CVMode);
end
model = SetSubjects(model, idxTrain, idxValid, idxTest);
end

%% ------------------------------------------------------------------------
function model = SetSubjects(model, IdxTrain, IdxValid, IdxTest)
model.IdxTrain = reshape(IdxTrain, 1, []);
model.IdxValid = reshape(IdxValid, 1, []);
model.IdxTest = reshape(IdxTest, 1, []);
model.IdxFit = [model.IdxTrain, model.IdxValid];
model.NumTrain = numel(model.IdxTrain);
model.NumValid = numel(model.IdxValid);
model.NumTest = numel(model.IdxTest);
if model.NumTrain == 0
    error('NetPRS:DataIndexing:EmptyTraining', 'The training set is empty.');
end
Xtr = model.XAll(:, model.IdxTrain);
switch lower(model.Param.XScaling)
    case 'none'
        model.XShift = zeros(model.NumSNP, 1);
        model.XScale = ones(model.NumSNP, 1);
        model.XUse = model.XAll;
    case 'center'
        model.XShift = mean(Xtr, 2);
        model.XScale = ones(model.NumSNP, 1);
        model.XUse = model.XAll - model.XShift;
    case 'zscore'
        model.XShift = mean(Xtr, 2);
        sd = std(Xtr, 0, 2);
        sd(sd == 0) = 1;                 % monomorphic in training: centred only
        model.XScale = sd;
        model.XUse = (model.XAll - model.XShift) ./ model.XScale;
end
end
