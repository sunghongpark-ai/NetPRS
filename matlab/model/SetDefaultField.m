function S = SetDefaultField(S, Name, Value)

if ~isstruct(S) || ~isscalar(S)
    error('NetPRS:SetDefaultField:InvalidInput', 'S must be a scalar struct.');
end
if ~ischar(Name)
    error('NetPRS:SetDefaultField:InvalidName', 'Name must be a character vector.');
end
if ~isfield(S, Name)
    S.(Name) = Value;
end
end
