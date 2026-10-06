function Text = CSVRows(Columns)

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
