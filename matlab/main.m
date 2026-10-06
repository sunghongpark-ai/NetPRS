function [Result, Info, Files] = main(ConfigOverride, ParameterOverride)
here = fileparts(mfilename('fullpath'));
previousPath = path;
cleanup = onCleanup(@() path(previousPath));
addpath(fullfile(here, 'model'));
[Config, Parameter] = NetPRSDefaults();
if nargin >= 1
    names = fieldnames(ConfigOverride);
    for k = 1:numel(names)
        if ~isfield(Config, names{k})
            error('NetPRS:Config', 'Unknown configuration field: %s', names{k});
        end
        Config.(names{k}) = ConfigOverride.(names{k});
    end
end
if nargin >= 2
    names = fieldnames(ParameterOverride);
    for k = 1:numel(names)
        if ~isfield(Parameter, names{k})
            error('NetPRS:Parameter', 'Unknown parameter field: %s', names{k});
        end
        Parameter.(names{k}) = ParameterOverride.(names{k});
    end
end
[Result, Info] = RunNetPRS(Config, Parameter);
Files = ExportResults(Result, Info);
if Config.Verbose
    fprintf('Results written to %s\n', Config.OutputDir);
end
end
