function [RelSNP, RelGene] = ReadSNPGeneRelation(File, Opts)

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
