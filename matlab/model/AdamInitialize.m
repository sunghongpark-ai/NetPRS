function param = AdamInitialize(NumVar, Alpha, Beta1, Beta2, Epsilon)

if nargin < 2 || isempty(Alpha),   Alpha = 1e-4;   end
if nargin < 3 || isempty(Beta1),   Beta1 = 0.9;    end
if nargin < 4 || isempty(Beta2),   Beta2 = 0.999;  end
if nargin < 5 || isempty(Epsilon), Epsilon = 1e-8; end

param.alpha = Alpha;
param.beta1 = Beta1;
param.beta2 = Beta2;
param.epsilon = Epsilon;
param.t = 0;
param.m = zeros(NumVar, 1);
param.v = zeros(NumVar, 1);
end
