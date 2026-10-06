function [RelSNP, RelGene] = ReadSNPGeneRelation(File, Opts)
%READSNPGENERELATION Read SNP-gene relations ('SNP GENE [...]' per line).
%
%   [RelSNP, RelGene] = ReadSNPGeneRelation(File)
%   [RelSNP, RelGene] = ReadSNPGeneRelation(File, Opts)
%
%   Each non-empty line holds a SNP identifier and a gene identifier
%   separated by white space (e.g. 'rs429358 APOE'), as extracted from dbSNP;
%   further fields are ignored and lines may have different numbers of
%   fields. Windows line endings are accepted.
%
%   Opts.Header  'auto' (default) | true | false; 'auto' treats the first
%                line as a header when its first field contains no digit
%                (e.g. 'SNP GENE'), since SNP identifiers contain digits.
%
%   Outputs: RelSNP, RelGene {R x 1} cellstr (input of SNPGeneMatrix)

if nargin < 2 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'Header', 'auto');
if exist(File, 'file') ~= 2
    error('NetPRS:ReadSNPGeneRelation:Open', 'Cannot open %s.', File);
end
text = fileread(File);
tok = regexp(text, '^[ \t]*(\S+)[ \t]+(\S+)', 'tokens', 'lineanchors');
if isempty(tok)
    error('NetPRS:ReadSNPGeneRelation:Empty', 'No relation found in %s.', File);
end
tok = vertcat(tok{:});
if ischar(Opts.Header)
    hasHeader = isempty(regexp(tok{1, 1}, '\d', 'once'));
else
    hasHeader = logical(Opts.Header);
end
if hasHeader
    tok = tok(2:end, :);
end
RelSNP = tok(:, 1);
RelGene = tok(:, 2);
end
