function S = SetDefaultField(S, Name, Value)
%SETDEFAULTFIELD Assign a default value to a struct field that is absent.
%
%   S = SetDefaultField(S, Name, Value) returns S with S.(Name) = Value when
%   the field Name does not exist. An existing field is never overwritten,
%   even when it is empty, so that an explicit [] supplied by the user keeps
%   its meaning (e.g. "no fixed epoch").
%
%   Inputs
%     S      scalar struct (use struct() for an empty option set)
%     Name   char, field name
%     Value  any MATLAB value
%
%   Output
%     S      struct with the field guaranteed to exist

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
