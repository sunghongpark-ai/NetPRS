"""Stage 2 of NetPRS: the network-based polygenic risk score model (Sec. III).

Paper equations
---------------
* (4) network propagation  ``F_* = (I + mu_* L_*)^{-1} X``  (``* = P, G``)
* (5) effect integration   ``Z = theta_I F_I + theta_P F_P + theta_G F_G``,
  ``F_I = X``, ``theta = softmax(alpha)``
* risk                     ``P = 1 / (1 + exp(-beta' Z))``
* (6) training             ``min CE + delta (||beta||^2 + sum alpha^2 + sum mu^2)``
  by ADAM (learning rate 0.01) from ``mu_* = 1``, ``alpha_* = 0``, with the
  analytic gradients of Sec. III-D.

Implementation
--------------
Because ``Q_* = I + mu_* L_*`` is symmetric, ``beta' F_* = (Q_*^{-1} beta)' X`` and
``beta' Z = Omega' X`` with ``Omega = sum_e theta_e Q_e^{-1} beta``: risk and
gradients are evaluated exactly with matrix-vector products after one
eigendecomposition per network (``solver='eig'``) instead of forming the
``s x n`` matrices ``F_*`` and ``Z``. ``solver='chol'`` is the reference path.

The classes mirror the MATLAB functions: :func:`model_initialize`
(``ModelInitialize``), :class:`NetPRSModel` with ``data_indexing``,
``param_initialize``, ``param_reshape``, ``forward_propagate``,
``loss_calculation``, ``backward_propagate``, ``parameter_update``,
``param_training``, ``risk_predict``, ``effect_extraction``, and
:func:`train_netprs` (``TrainNetPRS``). Folds and the initial ``beta`` come from
the portable streams ``RandomStream(seed + iteration, 1)`` and
``RandomStream(seed + iteration, 2)``, exactly as in MATLAB/Octave.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import scipy.linalg as la

from .network import graph_laplacian
from .rng import RandomStream
from .stats import compute_auc

__all__ = [
    "EFFECT_CODES", "Parameters", "Network", "NetPRSDesign", "model_initialize", "Split", "AdamState",
    "Effects", "NetPRSModel", "TrainResult", "train_netprs",
]

EFFECT_CODES = ("I", "P", "G", "IP", "IG", "PG", "IPG")


@dataclass
class Parameters:
    """NetPRS model and training settings (``Param`` of ``ModelInitialize``).

    Values not reported in the paper (``max_epoch``, ``reg_coeff``, the
    initialization of beta, the CV design, ``x_scaling``, ``mu_min``) follow
    NeuroFANN/BIGPN/PPIxGPN or are declared safeguards (see README).
    """

    effects: str = "IPG"            # any non-empty subset of I, P, G (ablation: Fig. 3b)
    num_iter: int = 10              # repetitions of stratified K-fold splitting
    num_fold: int = 5               # K
    max_epoch: int = 500            # ADAM steps
    learn_rate: float = 0.01        # paper
    reg_coeff: float = 0.005        # delta of Eq. (6)
    mu_init: float = 1.0            # paper
    alpha_init: float = 0.0         # paper
    mu_min: float = 0.0             # projection mu >= mu_min after each step (-inf disables)
    laplacian: str = "unnormalized" # L = D - W (paper) | "normalized"
    solver: str = "eig"             # "eig" (fast) | "chol" (reference)
    x_scaling: str = "center"       # "center" | "none" | "zscore" (training-fold statistics)
    seed: int = 0                   # folds: RandomStream(seed + i, 1); beta: RandomStream(seed + i, 2)
    verbose: bool = False           # loss every 100 epochs

    def validate(self) -> None:
        for name in ("num_iter", "num_fold", "max_epoch", "seed"):
            v = getattr(self, name)
            if not (np.isscalar(v) and np.isfinite(v) and float(v).is_integer()):
                raise ValueError(f"{name} must be an integer.")
            setattr(self, name, int(v))
        for name in ("num_iter", "num_fold", "max_epoch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.num_fold < 2:
            raise ValueError("num_fold must be at least 2.")
        for name in ("learn_rate", "reg_coeff"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be a finite non-negative scalar.")
        if not 0 <= self.seed < 2 ** 32 - self.num_iter:
            raise ValueError("seed must be a non-negative integer with seed + num_iter < 2**32.")
        if not (np.isfinite(self.mu_init) and self.mu_init >= self.mu_min):
            raise ValueError("mu_init must be finite and not below mu_min.")
        if self.solver not in ("eig", "chol"):
            raise ValueError("solver must be 'eig' or 'chol'.")
        if self.x_scaling not in ("none", "center", "zscore"):
            raise ValueError("x_scaling must be 'none', 'center' or 'zscore'.")
        if self.laplacian not in ("unnormalized", "normalized"):
            raise ValueError("laplacian must be 'unnormalized' or 'normalized'.")


@dataclass
class Network:
    """One SNP interaction network with its Laplacian (and eigendecomposition)."""

    name: str                       # "P" or "G"
    effect: int                     # position in "IPG" (1 = P, 2 = G)
    W: np.ndarray
    L: np.ndarray
    V: Optional[np.ndarray] = None  # eigenvectors (solver "eig")
    lam: Optional[np.ndarray] = None


@dataclass
class NetPRSDesign:
    """Data, networks and cross-validation design (output of :func:`model_initialize`).

    ``cv_folds[i, j]`` is the fold label (1..K) of discovery subject j in
    iteration i + 1. ``cv_list`` rows are ``(iteration, valid_fold)`` in the
    external mode and ``(iteration, test_fold, valid_fold)`` in the nested mode
    (1-based labels, as in MATLAB).
    """

    params: Parameters
    num_snp: int
    num_disc: int
    num_ext: int
    has_external_test: bool
    x_all: np.ndarray               # (s, num_disc + num_ext)
    y_all: np.ndarray
    snp: List[str]
    effect_code: str
    effect_list: np.ndarray         # positions in "IPG"
    nets: List[Network]
    cv_mode: str
    cv_folds: np.ndarray
    cv_list: np.ndarray

    @property
    def num_all(self) -> int:
        return self.num_disc + self.num_ext

    @property
    def num_effect(self) -> int:
        return int(self.effect_list.size)

    @property
    def num_net(self) -> int:
        return len(self.nets)

    @property
    def num_model(self) -> int:
        return int(self.cv_list.shape[0])


def model_initialize(x_data: np.ndarray, y_data: np.ndarray, x_test: Optional[np.ndarray] = None,
                     y_test: Optional[np.ndarray] = None, w_phe: Optional[np.ndarray] = None,
                     w_gen: Optional[np.ndarray] = None, snp: Optional[Sequence[str]] = None,
                     params: Optional[Parameters] = None) -> NetPRSDesign:
    """Build the NetPRS design (``ModelInitialize``).

    Parameters
    ----------
    x_data, y_data : (s, n) genotype dosage and (n,) diagnosis of the discovery cohort.
    x_test, y_test : optional external validation cohort (test set of every model).
    w_phe, w_gen : (s, s) networks W_P / W_G, required when ``params.effects``
        contains P / G.
    snp : optional SNP identifiers.

    Cross-validation: with an external cohort every model is trained on K-1
    discovery folds and the remaining fold selects the epoch; without one, a
    nested design (test fold, validation fold) is used. Folds are stratified by
    the diagnosis.
    """
    params = params or Parameters()
    params.validate()
    X = np.asarray(x_data, dtype=float)
    Y = np.asarray(y_data, dtype=float).ravel()
    _check_data(X, Y, "x_data", "y_data")
    s, n = X.shape
    has_test = x_test is not None and np.size(x_test) > 0
    if has_test:
        XT = np.asarray(x_test, dtype=float)
        YT = np.asarray(y_test, dtype=float).ravel()
        _check_data(XT, YT, "x_test", "y_test")
        if XT.shape[0] != s:
            raise ValueError("x_test must have the same number of SNPs (rows) as x_data.")
    else:
        XT, YT = np.zeros((s, 0)), np.zeros(0)
        if params.num_fold < 3:
            raise ValueError("without an external validation cohort num_fold must be at least 3.")

    letters = [c for c in params.effects.upper() if c.isalpha()]
    if not letters or any(c not in "IPG" for c in letters) or len(set(letters)) != len(letters):
        raise ValueError("effects must be a non-empty subset of 'I', 'P', 'G' without repetition.")
    use = np.array([c in letters for c in "IPG"])
    effect_list = np.flatnonzero(use)

    nets: List[Network] = []
    for k, (name, W) in enumerate((("P", w_phe), ("G", w_gen))):
        if not use[k + 1]:
            continue
        if W is None or np.size(W) == 0:
            raise ValueError(f"network W_{name} is required for effect '{name}'.")
        W = np.asarray(W.toarray() if hasattr(W, "toarray") else W, dtype=float)
        if W.shape != (s, s):
            raise ValueError(f"W_{name} must be {s} x {s}.")
        L = graph_laplacian(W, params.laplacian)
        net = Network(name=name, effect=k + 1, W=W, L=L)
        if params.solver == "eig":
            lam, V = np.linalg.eigh(L)
            tol = 1e-12 * max(1.0, float(np.max(np.abs(lam))))
            if lam.min() < -tol * s:
                raise ValueError(f"Laplacian of W_{name} is not positive semi-definite.")
            net.lam = np.maximum(lam, 0.0)
            net.V = V
        nets.append(net)

    # stratified folds: controls then cases permuted with RandomStream(seed + i, 1)
    K = params.num_fold
    cv_folds = np.zeros((params.num_iter, n), dtype=np.int64)
    for it in range(1, params.num_iter + 1):
        rs = RandomStream(params.seed + it, 1)
        for c in (0, 1):
            idx = np.flatnonzero(Y == c)
            perm = rs.permutation(idx.size) + 1                 # 1-based, as MATLAB
            cv_folds[it - 1, idx] = perm % K + 1
    counts = np.array([[np.sum((cv_folds[0] == f) & (Y == c)) for c in (0, 1)] for f in range(1, K + 1)])
    if np.any(counts[:, 1] == 0):
        warnings.warn("At least one fold contains no case; reduce num_fold.", RuntimeWarning, stacklevel=2)
    if has_test:
        cv_list = np.array([(it, f) for it in range(1, params.num_iter + 1) for f in range(1, K + 1)])
        mode = "external"
    else:
        cv_list = np.array([(it, ft, fv) for it in range(1, params.num_iter + 1)
                            for ft in range(1, K + 1) for fv in range(1, K + 1) if fv != ft])
        mode = "nested"
    return NetPRSDesign(
        params=params, num_snp=s, num_disc=n, num_ext=XT.shape[1], has_external_test=has_test,
        x_all=np.hstack([X, XT]), y_all=np.concatenate([Y, YT]),
        snp=list(snp) if snp is not None else [f"SNP{j + 1}" for j in range(s)],
        effect_code="".join(c for c, u in zip("IPG", use) if u), effect_list=effect_list, nets=nets,
        cv_mode=mode, cv_folds=cv_folds, cv_list=cv_list)


def _check_data(X: np.ndarray, Y: np.ndarray, name_x: str, name_y: str) -> None:
    if X.ndim != 2 or X.size == 0:
        raise ValueError(f"{name_x} must be a non-empty s x n matrix.")
    if np.any(~np.isfinite(X)):
        raise ValueError(f"{name_x} contains NaN/Inf; impute missing genotypes first (impute_genotype).")
    if Y.size != X.shape[1]:
        raise ValueError(f"{name_y} must have one label per column of {name_x}.")
    if np.any((Y != 0) & (Y != 1)):
        raise ValueError(f"{name_y} must be binary (0/1).")


@dataclass
class Split:
    """User-defined split (0-based column indices into ``x_all``)."""

    idx_train: Sequence[int]
    idx_valid: Sequence[int] = ()
    idx_test: Sequence[int] = ()
    idx_iter: int = 1               # iteration label: beta from RandomStream(seed + idx_iter, 2)


@dataclass
class AdamState:
    """ADAM optimizer state (Kingma & Ba, 2014), as ``AdamInitialize``."""

    alpha: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    t: int = 0
    m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def initialize(cls, num_var: int, alpha: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999,
                   epsilon: float = 1e-8) -> "AdamState":
        return cls(alpha=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon, t=0,
                   m=np.zeros(num_var), v=np.zeros(num_var))


@dataclass
class Effects:
    """Explicit effects of :meth:`NetPRSModel.effect_extraction` (absent effects are None)."""

    FI: Optional[np.ndarray]
    FP: Optional[np.ndarray]
    FG: Optional[np.ndarray]
    Z: np.ndarray
    theta: np.ndarray
    mu: np.ndarray
    effect_code: str


class NetPRSModel:
    """One NetPRS model on one split of a :class:`NetPRSDesign`.

    ``selector`` is a 0-based row of ``design.cv_list`` or a :class:`Split`.
    """

    def __init__(self, design: NetPRSDesign, selector: Union[int, Split]) -> None:
        self.design = design
        self.params = design.params
        self.data_indexing(selector)
        self.weights = np.zeros(0)
        self.adam = AdamState()
        self.gradient = np.zeros(0)

    # -- data ---------------------------------------------------------------
    def data_indexing(self, selector: Union[int, Split]) -> None:
        """Training/validation/test subjects and the scaled genotypes (``DataIndexing``)."""
        d = self.design
        if isinstance(selector, Split):
            tr = np.asarray(selector.idx_train, dtype=np.int64).ravel()
            va = np.asarray(selector.idx_valid, dtype=np.int64).ravel()
            te = np.asarray(selector.idx_test, dtype=np.int64).ravel()
            all_idx = np.concatenate([tr, va, te])
            if np.any((all_idx < 0) | (all_idx >= d.num_all)):
                raise ValueError(f"custom indices must lie in [0, {d.num_all}).")
            fit = np.concatenate([tr, va])
            if np.unique(fit).size != fit.size or np.isin(te, fit).any():
                raise ValueError("training, validation and test subjects must be disjoint.")
            self.idx_model, self.idx_iter = -1, int(selector.idx_iter)
            self.fold_valid = self.fold_test = None
        else:
            idx_model = int(selector)
            if not 0 <= idx_model < d.num_model:
                raise ValueError(f"idx_model must be an integer in [0, {d.num_model}).")
            row = d.cv_list[idx_model]
            self.idx_model, self.idx_iter = idx_model, int(row[0])
            fold = d.cv_folds[self.idx_iter - 1]
            if d.cv_mode == "external":
                self.fold_valid, self.fold_test = int(row[1]), None
                tr = np.flatnonzero(fold != self.fold_valid)
                va = np.flatnonzero(fold == self.fold_valid)
                te = d.num_disc + np.arange(d.num_ext)
            else:
                self.fold_test, self.fold_valid = int(row[1]), int(row[2])
                tr = np.flatnonzero((fold != self.fold_test) & (fold != self.fold_valid))
                va = np.flatnonzero(fold == self.fold_valid)
                te = np.flatnonzero(fold == self.fold_test)
        if tr.size == 0:
            raise ValueError("the training set is empty.")
        self.idx_train, self.idx_valid, self.idx_test = tr, va, te
        self.idx_fit = np.concatenate([tr, va])
        Xtr = d.x_all[:, tr]
        s = d.num_snp
        if self.params.x_scaling == "none":
            self.x_shift, self.x_scale = np.zeros(s), np.ones(s)
        elif self.params.x_scaling == "center":
            self.x_shift, self.x_scale = Xtr.mean(axis=1), np.ones(s)
        else:
            sd = Xtr.std(axis=1, ddof=1) if tr.size > 1 else np.zeros(s)
            self.x_shift, self.x_scale = Xtr.mean(axis=1), np.where(sd == 0, 1.0, sd)
        self.x_use = (d.x_all - self.x_shift[:, None]) / self.x_scale[:, None]

    @property
    def num_train(self) -> int:
        return int(self.idx_train.size)

    @property
    def num_valid(self) -> int:
        return int(self.idx_valid.size)

    # -- parameters -----------------------------------------------------------
    def param_initialize(self) -> None:
        """``mu = mu_init``, ``alpha = alpha_init``, Glorot-uniform beta; ADAM state."""
        d, p = self.design, self.params
        K, E, s = d.num_net, d.num_effect, d.num_snp
        u = RandomStream(p.seed + self.idx_iter, 2).uniform(s)
        beta = (2.0 * u - 1.0) * np.sqrt(6.0 / (s + 1))
        self.sl_mu = slice(0, K)
        self.sl_alpha = slice(K, K + E)
        self.sl_beta = slice(K + E, K + E + s)
        self.weights = np.concatenate([np.full(K, p.mu_init), np.full(E, p.alpha_init), beta])
        self.adam = AdamState.initialize(self.weights.size, p.learn_rate)

    def param_reshape(self) -> None:
        """Unpack the weights; ``theta = softmax(alpha)``; prepare ``(I + mu L)^{-1}``."""
        w = self.weights
        self.mu, self.alpha, self.beta = w[self.sl_mu], w[self.sl_alpha], w[self.sl_beta]
        a = np.exp(self.alpha - self.alpha.max())
        self.theta = a / a.sum()
        s = self.design.num_snp
        self._ops = []
        for k, net in enumerate(self.design.nets):
            mu = self.mu[k]
            if self.params.solver == "eig":
                den = 1.0 + mu * net.lam
                if np.any(den <= 0):
                    raise ValueError(f"I + mu*L of network {net.name} is not positive definite (mu = {mu:g}).")
                self._ops.append(("eig", 1.0 / den))
            else:
                try:
                    R = la.cholesky(np.eye(s) + mu * net.L, lower=False)
                except la.LinAlgError:
                    raise ValueError(f"I + mu*L of network {net.name} is not positive definite "
                                     f"(mu = {mu:g}).") from None
                self._ops.append(("chol", R))

    def propagation_solve(self, k: int, B: np.ndarray) -> np.ndarray:
        """``(I + mu_k L_k)^{-1} B`` for network k (``PropagationSolve``)."""
        kind, op = self._ops[k]
        if kind == "eig":
            V = self.design.nets[k].V
            coef = V.T @ B
            return V @ (op[:, None] * coef if coef.ndim == 2 else op * coef)
        return la.cho_solve((op, False), B)

    def _net_index(self, effect: int) -> int:
        return next(k for k, net in enumerate(self.design.nets) if net.effect == effect)

    # -- forward / loss / backward -----------------------------------------------
    def forward_propagate(self, scope: str = "fit") -> None:
        """Risk of the fitting subjects (``'fit'``) or of all subjects (``'all'``)."""
        d = self.design
        if scope == "fit":
            cols = self.idx_fit
        elif scope == "all":
            cols = np.arange(d.num_all)
        else:
            raise ValueError("scope must be 'fit' or 'all'.")
        weff = np.zeros((d.num_snp, d.num_effect))
        for e, eff in enumerate(d.effect_list):
            weff[:, e] = self.beta if eff == 0 else self.propagation_solve(self._net_index(eff), self.beta)
        self.weff = weff
        self.omega = weff @ self.theta                       # beta' Z = omega' X
        self.logit = np.full(d.num_all, np.nan)
        self.logit[cols] = self.omega @ self.x_use[:, cols]
        self.prob = 1.0 / (1.0 + np.exp(-self.logit))

    def loss_calculation(self) -> None:
        """Cross-entropy (stable softplus form) and the objective of Eq. (6)."""
        t, y = self.logit, self.design.y_all
        self.loss_train = _cross_entropy(t[self.idx_train], y[self.idx_train])
        self.loss_valid = _cross_entropy(t[self.idx_valid], y[self.idx_valid]) if self.num_valid else float("nan")
        self.penalty = float(np.sum(self.beta ** 2) + np.sum(self.alpha ** 2) + np.sum(self.mu ** 2))
        self.objective = self.loss_train + self.params.reg_coeff * self.penalty

    def backward_propagate(self) -> None:
        """Analytic gradients of Sec. III-D (exact low-cost form; see module docstring)."""
        d = self.design
        n = self.num_train
        delta = self.params.reg_coeff
        r = self.prob[self.idx_train] - d.y_all[self.idx_train]
        c = self.x_use[:, self.idx_train] @ r
        qinv_c = np.zeros((d.num_snp, d.num_effect))
        for e, eff in enumerate(d.effect_list):
            qinv_c[:, e] = c if eff == 0 else self.propagation_solve(self._net_index(eff), c)
        g_beta = qinv_c @ self.theta / n + 2 * delta * self.beta
        g_alpha = self.theta * ((self.weff - self.omega[:, None]).T @ c) / n + 2 * delta * self.alpha
        g_mu = np.zeros(d.num_net)
        for k, net in enumerate(d.nets):
            e = int(np.flatnonzero(d.effect_list == net.effect)[0])
            quad = qinv_c[:, e] @ (net.L @ self.weff[:, e])
            g_mu[k] = -self.theta[e] / n * quad + 2 * delta * self.mu[k]
        self.gradient = np.concatenate([g_mu, g_alpha, g_beta])

    def parameter_update(self) -> None:
        """One ADAM step, then the projection ``mu >= mu_min``."""
        A, g = self.adam, self.gradient
        A.t += 1
        A.m = A.beta1 * A.m + (1 - A.beta1) * g
        A.v = A.beta2 * A.v + (1 - A.beta2) * g ** 2
        m_hat = A.m / (1 - A.beta1 ** A.t)
        v_hat = A.v / (1 - A.beta2 ** A.t)
        self.weights = self.weights - A.alpha * m_hat / (np.sqrt(v_hat) + A.epsilon)
        if self.design.num_net > 0 and np.isfinite(self.params.mu_min):
            self.weights[self.sl_mu] = np.maximum(self.weights[self.sl_mu], self.params.mu_min)

    def param_training(self) -> None:
        """ADAM training; the weights with the minimum validation loss are kept.

        The weights after 0, 1, ..., ``max_epoch`` updates are evaluated (first
        minimum on ties). Without validation subjects the weights after exactly
        ``max_epoch`` updates are kept.
        """
        T = self.params.max_epoch
        self.loss_train_curve = np.full(T + 1, np.nan)
        self.loss_valid_curve = np.full(T + 1, np.nan)
        self.objective_curve = np.full(T + 1, np.nan)
        has_valid = self.num_valid > 0
        best_loss, best_weight, best_epoch = np.inf, self.weights.copy(), 0
        for epoch in range(T + 1):
            self.param_reshape()
            self.forward_propagate("fit")
            self.loss_calculation()
            self.loss_train_curve[epoch] = self.loss_train
            self.loss_valid_curve[epoch] = self.loss_valid
            self.objective_curve[epoch] = self.objective
            if not np.isfinite(self.objective):
                raise FloatingPointError(f"non-finite objective at epoch {epoch} (reduce learn_rate).")
            if has_valid and self.loss_valid < best_loss:
                best_loss, best_weight, best_epoch = self.loss_valid, self.weights.copy(), epoch
            if self.params.verbose and epoch % 100 == 0:
                print(f"    epoch {epoch:4d} | train {self.loss_train:.4f} | valid {self.loss_valid:.4f} | "
                      f"mu {np.array2string(self.mu, precision=3)} | theta {np.array2string(self.theta, precision=3)}")
            if epoch == T:
                break
            self.backward_propagate()
            self.parameter_update()
        if not has_valid:
            best_weight, best_epoch = self.weights.copy(), T
        self.best_epoch = best_epoch
        self.weights = best_weight
        self.param_reshape()
        self.forward_propagate("all")
        self.loss_calculation()

    # -- prediction and interpretation --------------------------------------------
    def risk_predict(self, X: np.ndarray):
        """Risk ``P`` and logit ``beta' Z`` of new subjects (``X``: s x m, same SNP order)."""
        X = np.asarray(X, dtype=float)
        if X.shape[0] != self.design.num_snp:
            raise ValueError(f"X must have {self.design.num_snp} rows (SNPs).")
        if np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN/Inf; impute missing genotypes first.")
        logit = self.omega @ ((X - self.x_shift[:, None]) / self.x_scale[:, None])
        return 1.0 / (1.0 + np.exp(-logit)), logit

    def effect_extraction(self, X: Optional[np.ndarray] = None) -> Effects:
        """Independent, interactive and integrated effects ``F_I, F_P, F_G, Z`` of ``X``."""
        d = self.design
        X = d.x_all if X is None else np.asarray(X, dtype=float)
        if X.shape[0] != d.num_snp:
            raise ValueError(f"X must have {d.num_snp} rows (SNPs).")
        self.param_reshape()
        Xs = (X - self.x_shift[:, None]) / self.x_scale[:, None]
        F = {0: None, 1: None, 2: None}
        Z = np.zeros(Xs.shape)
        for e, eff in enumerate(d.effect_list):
            Fe = Xs if eff == 0 else self.propagation_solve(self._net_index(eff), Xs)
            F[int(eff)] = Fe
            Z = Z + self.theta[e] * Fe
        return Effects(FI=F[0], FP=F[1], FG=F[2], Z=Z, theta=self.theta.copy(), mu=self.mu.copy(),
                       effect_code=d.effect_code)


def _cross_entropy(t: np.ndarray, y: np.ndarray) -> float:
    if t.size == 0:
        return float("nan")
    softplus = np.maximum(t, 0) + np.log1p(np.exp(-np.abs(t)))
    return float(np.mean(softplus - y * t))


@dataclass
class TrainResult:
    """Compact result of one trained model (``TrainNetPRS``).

    ``idx_model`` is the 0-based row of ``cv_list`` (-1 for a custom split);
    iteration and fold labels are 1-based; ``idx_test`` are 0-based columns of
    ``x_all``.
    """

    idx_model: int
    idx_iter: int
    fold_valid: Optional[int]
    fold_test: Optional[int]
    best_epoch: int
    effect_code: str
    mu: np.ndarray
    alpha: np.ndarray
    theta: np.ndarray
    beta: np.ndarray
    omega: np.ndarray
    idx_test: np.ndarray
    prob_test: np.ndarray
    auc_train: float
    auc_valid: float
    auc_test: float
    loss_train_curve: np.ndarray
    loss_valid_curve: np.ndarray


def train_netprs(design: NetPRSDesign, selector: Union[int, Split]):
    """Train and evaluate one model; returns ``(TrainResult, NetPRSModel)``."""
    model = NetPRSModel(design, selector)
    model.param_initialize()
    model.param_training()
    y = design.y_all

    def auc(idx: np.ndarray) -> float:
        return compute_auc(model.prob[idx], y[idx]) if idx.size else float("nan")

    result = TrainResult(
        idx_model=model.idx_model, idx_iter=model.idx_iter, fold_valid=model.fold_valid,
        fold_test=model.fold_test, best_epoch=model.best_epoch, effect_code=design.effect_code,
        mu=model.mu.copy(), alpha=model.alpha.copy(), theta=model.theta.copy(), beta=model.beta.copy(),
        omega=model.omega.copy(), idx_test=model.idx_test.copy(), prob_test=model.prob[model.idx_test].copy(),
        auc_train=auc(model.idx_train), auc_valid=auc(model.idx_valid), auc_test=auc(model.idx_test),
        loss_train_curve=model.loss_train_curve, loss_valid_curve=model.loss_valid_curve)
    return result, model
