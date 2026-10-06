function Text = CSVRows(Columns)
%CSVROWS Comma-separated text of rows given column-wise (empty fields kept).
%
%   Text = CSVRows(Columns)
%
%   Columns is a cell array {c1, ..., ck} of cellstr columns with N elements
%   each; Text is the char row 'c1{1},...,ck{1}\n...c1{N},...,ck{N}\n' (LF
%   line ends). The text is assembled by one concatenation instead of
%   sprintf('%s,%s\n', C{:}) because MATLAB's sprintf skips empty arguments
%   (an empty field would shift the following fields), whereas GNU Octave
%   keeps them; concatenation behaves the same in both.

k = numel(Columns);
if k == 0
    Text = '';
    return;
end
N = numel(Columns{1});
if any(cellfun(@numel, Columns) ~= N)
    error('NetPRS:CSVRows:Size', 'All columns must have the same number of elements.');
end
if N == 0
    Text = '';
    return;
end
T = repmat({','}, 2 * k, N);
for c = 1:k
    T(2 * c - 1, :) = reshape(Columns{c}, 1, N);
end
T(2 * k, :) = {char(10)};
Text = [T{:}];
end
