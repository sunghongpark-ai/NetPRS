function Summary = SummarizeCV(ModelInit, Results, Label, Verbose)

numModel = numel(Results);
numIter = ModelInit.Param.NumIter;
Summary.Label = '';
if nargin >= 3 && ~isempty(Label)
    Summary.Label = Label;
end
if nargin < 4 || isempty(Verbose)
    Verbose = ~isempty(Summary.Label);
end
Summary.EffectCode = ModelInit.EffectCode;
Summary.CVMode = ModelInit.CVMode;
Summary.NumModel = numModel;
Summary.Iteration = cellfun(@(r) r.IdxIter, Results(:));
Summary.FoldValid = cellfun(@ScalarOrNaN, cellfun(@(r) r.FoldValid, Results(:), 'UniformOutput', false));
Summary.FoldTest = cellfun(@ScalarOrNaN, cellfun(@(r) r.FoldTest, Results(:), 'UniformOutput', false));
Summary.BestEpoch = cellfun(@(r) r.BestEpoch, Results(:));
Summary.AUCTrain = cellfun(@(r) r.AUCTrain, Results(:));
Summary.AUCValid = cellfun(@(r) r.AUCValid, Results(:));
Summary.AUCTest = cellfun(@(r) r.AUCTest, Results(:));
ok = ~isnan(Summary.AUCTest);
Summary.AUCMean = mean(Summary.AUCTest(ok));
Summary.AUCSD = std(Summary.AUCTest(ok));
Summary.ThetaModel = cell2mat(cellfun(@(r) r.Theta(:), Results(:)', 'UniformOutput', false));
Summary.Theta = mean(Summary.ThetaModel, 2);
if ModelInit.NumNet > 0
    Summary.MuModel = cell2mat(cellfun(@(r) r.Mu(:), Results(:)', 'UniformOutput', false));
    Summary.Mu = mean(Summary.MuModel, 2);
else
    Summary.MuModel = zeros(0, numModel);
    Summary.Mu = zeros(0, 1);
end
Summary.MedianBestEpoch = median(Summary.BestEpoch);

iters = cellfun(@(r) r.IdxIter, Results);
switch ModelInit.CVMode
    case 'external'
        yTest = ModelInit.YAll(ModelInit.NumDisc + (1:ModelInit.NumExt));
        prob = zeros(1, ModelInit.NumExt);
        aucIter = nan(numIter, 1);
        for it = 1:numIter
            sel = find(iters == it);
            if isempty(sel)
                continue;
            end
            pIt = zeros(1, ModelInit.NumExt);
            for k = sel(:)'
                pIt = pIt + Results{k}.ProbTest;
            end
            aucIter(it) = ComputeAUC(pIt / numel(sel), yTest);
            prob = prob + pIt;
        end
        Summary.AUCIterEnsemble = aucIter;
        Summary.EnsembleProb = prob / numModel;
        Summary.EnsembleAUC = ComputeAUC(Summary.EnsembleProb, yTest);
    case 'nested'
        yDisc = ModelInit.YAll(1:ModelInit.NumDisc);
        aucIter = nan(numIter, 1);
        accAll = zeros(1, ModelInit.NumDisc);
        cntAll = zeros(1, ModelInit.NumDisc);
        for it = 1:numIter
            acc = zeros(1, ModelInit.NumDisc);
            cnt = zeros(1, ModelInit.NumDisc);
            for k = find(iters == it)'
                acc(Results{k}.IdxTest) = acc(Results{k}.IdxTest) + Results{k}.ProbTest;
                cnt(Results{k}.IdxTest) = cnt(Results{k}.IdxTest) + 1;
            end
            if all(cnt > 0)
                aucIter(it) = ComputeAUC(acc ./ cnt, yDisc);
            end
            accAll = accAll + acc;
            cntAll = cntAll + cnt;
        end
        Summary.AUCIterOOF = aucIter;
        Summary.EnsembleProb = accAll ./ max(cntAll, 1);
        if all(cntAll > 0)
            Summary.EnsembleAUC = ComputeAUC(Summary.EnsembleProb, yDisc);
        else
            Summary.EnsembleAUC = NaN;
        end
    otherwise
        error('NetPRS:SummarizeCV:CVMode', 'Unknown CV mode.');
end

if Verbose
    label = Summary.Label;
    if isempty(label)
        label = Summary.EffectCode;
    end
    fprintf('%-24s AUC %.4f +/- %.4f (%d models; ensemble %.4f) | theta %s | mu %s | epoch %g\n', ...
        label, Summary.AUCMean, Summary.AUCSD, nnz(ok), Summary.EnsembleAUC, ...
        VectorText(Summary.Theta), VectorText(Summary.Mu), Summary.MedianBestEpoch);
end
end

function v = ScalarOrNaN(x)
if isempty(x)
    v = NaN;
else
    v = x;
end
end

function t = VectorText(v)
if isempty(v)
    t = '-';
else
    t = ['[', strtrim(sprintf('%.3g ', v)), ']'];
end
end
