function Data = LoadNetPRSData(Config)
%LOADNETPRSDATA Load the inputs of the NetPRS pipeline.
%
%   Data = LoadNetPRSData(Config)
%
%   Config.DataSource = 'csv' (default of NetPRSDefaults)
%       Config.CSVFile            single-file CSV (format: ReadNetPRSCSV),
%                                 e.g. dataset/sample.csv
%   Config.DataSource = 'plink'
%       Config.PlinkPrefix        PLINK 1 binary fileset (.bed/.bim/.fam)
%                                 after quality control and LD pruning
%       Config.ValidationIIDFile  IIDs of the validation cohort, one per line
%                                 (all other subjects form the discovery cohort)
%       Config.SNPGeneFile        SNP-gene relations, 'SNP GENE' per line
%       Config.GGIFile            gene-gene interactions, 'GENE1 GENE2 SCORE'
%       Config.GGIWeightScale     multiplier of SCORE (1/1000 for STRING)
%     Subjects without a case/control status (PLINK phenotype not 1/2) are
%     removed.
%   Config.DataSource = 'synthetic'
%       data of GenerateSyntheticData(Config.SyntheticOptions); with the
%       default options these are exactly the data of dataset/sample.csv
%
%   Genes (CSV and PLINK input): the sorted union of the genes of the
%   gene-gene interactions and of the SNP-gene relations. A related gene
%   without any interaction is an isolated node of the GGI network, so SNPs
%   that share it still interact through it (Eq. (2)-(3) without smoothing
%   at that gene).
%
%   Output Data
%     .Geno [s x n] dosage (NaN = missing), .Y [1 x n], .IsValidation [1 x n]
%     .SNP {s x 1}, .A1/.A2 {s x 1} (counted / other allele; empty unless
%     PLINK), .IID {n x 1}, .Gene {m x 1}, .Wggi [m x m] sparse,
%     .RelSNP/.RelGene {R x 1}, .RelWeight [R x 1], .Source

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
