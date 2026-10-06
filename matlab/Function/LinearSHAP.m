function [Phi, Base] = LinearSHAP(Beta, Zexplain, Zbackground)
%LINEARSHAP Exact SHAP values of the NetPRS risk score (log-odds scale).
%
%   [Phi, Base] = LinearSHAP(Beta, Zexplain, Zbackground)
%
%   NetPRS predicts logit(P) = beta' Z, which is linear in the integrated
%   effect Z. For a linear model with independent (interventional) features
%   the SHAP values (Lundberg & Lee, 2017) are exact and given by
%       Phi(j, i) = beta_j * (Z(j, i) - E_background[Z(j, :)]),
%   and satisfy local accuracy: sum_j Phi(:, i) = beta' Z(:, i) - Base,
%   with Base = beta' E_background[Z]. This is the explanatory analysis of
%   Fig. 4(b) (SHAP values of the integrated effect); the paper does not
%   state the SHAP variant, the exact linear form is used here.
%
%   Inputs
%     Beta         [s x 1] SNP effect size of a trained model
%     Zexplain     [s x n] integrated effect of the subjects to explain
%     Zbackground  [s x b] integrated effect of the background subjects
%                  (e.g. the training subjects)
%   Outputs
%     Phi   [s x n] SHAP values (log-odds)
%     Base  expected log-odds over the background

Beta = Beta(:);
if size(Zexplain, 1) ~= numel(Beta) || size(Zbackground, 1) ~= numel(Beta)
    error('NetPRS:LinearSHAP:SizeMismatch', 'Z matrices must have numel(Beta) rows.');
end
mu = mean(Zbackground, 2);
Phi = Beta .* (Zexplain - mu);
Base = Beta' * mu;
end
