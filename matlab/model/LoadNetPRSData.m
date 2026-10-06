function Data = LoadNetPRSData(Config)

if ~isfield(Config, 'DataSource')
    error('NetPRS:LoadNetPRSData:Config', 'Config.DataSource is required.');
end
switch lower(Config.DataSource)
    case 'csv'
        if ~isfield(Config, 'CSVFile') || isempty(Config.CSVFile)
            error('NetPRS:LoadNetPRSData:Config', 'Config.CSVFile is required for CSV input.');
        end
        Data = ReadNetPRSCSV(Config.CSVFile);
    case 'synthetic'
        opts = struct();
        if isfield(Config, 'SyntheticOptions') && ~isempty(Config.SyntheticOptions)
            opts = Config.SyntheticOptions;
        end
        Data = GenerateSyntheticData(opts);
    case 'plink'
        req = {'PlinkPrefix', 'ValidationIIDFile', 'SNPGeneFile', 'GGIFile'};
        for i = 1:numel(req)
            if ~isfield(Config, req{i}) || isempty(Config.(req{i}))
                error('NetPRS:LoadNetPRSData:Config', 'Config.%s is required for PLINK input.', req{i});
            end
        end
        if ~isfield(Config, 'GGIWeightScale') || isempty(Config.GGIWeightScale)
            Config.GGIWeightScale = 1;
        end
        [Geno, Bim, Fam] = ReadPlinkBed(Config.PlinkPrefix);
        valText = fileread(Config.ValidationIIDFile);
        valIID = regexp(valText, '\S+', 'match');
        isVal = ismember(Fam.IID, valIID)';
        if ~any(isVal)
            warning('NetPRS:LoadNetPRSData:NoValidation', 'No subject of %s was found in the fileset.', ...
                Config.ValidationIIDFile);
        end
        hasY = ~isnan(Fam.Y');
        Data = struct();
        Data.Geno = Geno(:, hasY);
        Data.Y = Fam.Y(hasY)';
        Data.IsValidation = isVal(hasY);
        Data.SNP = Bim.SNP;
        Data.A1 = Bim.A1;
        Data.A2 = Bim.A2;
        Data.IID = Fam.IID(hasY);
        [Data.RelSNP, Data.RelGene] = ReadSNPGeneRelation(Config.SNPGeneFile);
        Data.RelWeight = ones(numel(Data.RelSNP), 1);
        opts = struct('WeightScale', Config.GGIWeightScale);
        [~, ggiGene] = ReadEdgeList(Config.GGIFile, [], opts);
        Data.Gene = unique([ggiGene(:); Data.RelGene(:)]);
        Data.Wggi = ReadEdgeList(Config.GGIFile, Data.Gene, opts);
        Data.Source = 'plink';
    otherwise
        error('NetPRS:LoadNetPRSData:Config', 'Config.DataSource must be ''csv'', ''plink'' or ''synthetic''.');
end
end
