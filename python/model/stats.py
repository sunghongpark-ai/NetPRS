from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import special

__all__ = [
    "LogisticBatchResult", "logistic_batch", "GWASResult", "gwas_logistic", "EpistasisResult",
    "epistasis_test", "hwe_exact_test", "two_sample_ttest", "ks_test2_asymptotic", "compute_auc",
    "pairs_to_matrix",
]


@dataclass
class LogisticBatchResult:


    coef: np.ndarray
    se: np.ndarray
    converged: np.ndarray
    num_obs: np.ndarray
    num_iter: np.ndarray
    failed: np.ndarray


def logistic_batch(y: np.ndarray, C: np.ndarray, V: np.ndarray, M: Optional[np.ndarray] = None,
                   max_iter: int = 25, tol: float = 1e-8, pivot_tol: float = 1e-10) -> LogisticBatchResult:


    y = np.asarray(y, dtype=float).ravel().copy()
    C = np.asarray(C, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    n = y.size
    if C.shape[0] != n or V.shape[0] != n:
        raise ValueError("y, C and V must have the same number of rows.")
    q, B, r = C.shape[1], V.shape[1], V.shape[2]
    p = q + r
    M = np.ones((n, B), dtype=bool) if M is None else np.asarray(M, dtype=bool).copy()
    if M.shape != (n, B):
        raise ValueError("M must be n x B.")

    row_ok = np.isfinite(y) & np.all(np.isfinite(C), axis=1)
    M &= row_ok[:, None] & np.all(np.isfinite(V), axis=2)
    y[~row_ok] = 0.0
    C[~row_ok, :] = 0.0
    V[~np.isfinite(V)] = 0.0
    V[~M] = 0.0
    Mf = M.astype(float)

    coef = np.zeros((p, B))
    converged = np.zeros(B, dtype=bool)
    failed = np.zeros(B, dtype=bool)
    num_iter = np.zeros(B, dtype=np.int64)
    active = np.arange(B)
    for it in range(1, max_iter + 1):
        if active.size == 0:
            break
        g, H = _score_hessian(y, C, V[:, active, :], Mf[:, active], coef[:, active])
        L, ok = _batch_chol(H, pivot_tol)
        step = np.zeros((p, active.size))
        if ok.any():
            step[:, ok] = _batch_solve(L[:, :, ok], g[:, ok])
        coef[:, active] += step
        num_iter[active] = it
        tol_abs = tol * (1.0 + np.max(np.abs(coef[:, active]), axis=0))
        done = ok & (np.max(np.abs(step), axis=0) <= tol_abs) & np.all(np.isfinite(coef[:, active]), axis=0)
        converged[active[done]] = True
        failed[active[~ok]] = True
        active = active[~done & ok]

    se = np.full((p, B), np.nan)
    idx = np.flatnonzero(converged)
    if idx.size:
        _, H = _score_hessian(y, C, V[:, idx, :], Mf[:, idx], coef[:, idx])
        L, ok = _batch_chol(H, pivot_tol)
        se_idx = np.full((p, idx.size), np.nan)
        if ok.any():
            se_idx[:, ok] = np.sqrt(_batch_inv_diag(L[:, :, ok]))
        se[:, idx] = se_idx
        converged[idx[~ok]] = False
    coef[:, ~converged] = np.nan
    se[:, ~converged] = np.nan
    return LogisticBatchResult(coef=coef, se=se, converged=converged, num_obs=M.sum(axis=0),
                               num_iter=num_iter, failed=failed | ~converged)


def _column(C: np.ndarray, V: np.ndarray, a: int) -> np.ndarray:
    q = C.shape[1]
    return C[:, a:a + 1] if a < q else V[:, :, a - q]


def _score_hessian(y, C, V, Mf, coef) -> Tuple[np.ndarray, np.ndarray]:

    p = C.shape[1] + V.shape[2]
    B = V.shape[1]
    eta = np.zeros(Mf.shape)
    for a in range(p):
        eta = eta + _column(C, V, a) * coef[a, :]
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = mu * (1.0 - mu) * Mf
    res = (y[:, None] - mu) * Mf
    g = np.zeros((p, B))
    H = np.zeros((p, p, B))
    for a in range(p):
        xa = _column(C, V, a)
        g[a] = np.sum(xa * res, axis=0)
        wxa = w * xa
        for b in range(a + 1):
            hab = np.sum(wxa * _column(C, V, b), axis=0)
            H[a, b] = hab
            H[b, a] = hab
    return g, H


def _batch_chol(H: np.ndarray, pivot_tol: float) -> Tuple[np.ndarray, np.ndarray]:

    p, _, B = H.shape
    L = np.zeros((p, p, B))
    ok = np.ones(B, dtype=bool)
    tiny = np.finfo(float).tiny
    for j in range(p):
        d = H[j, j] - np.sum(L[j, :j] ** 2, axis=0)
        bad = ~(d > pivot_tol * np.maximum(np.abs(H[j, j]), tiny))
        ok &= ~bad
        d = np.where(bad, 1.0, d)
        L[j, j] = np.sqrt(d)
        for i in range(j + 1, p):
            L[i, j] = (H[i, j] - np.sum(L[i, :j] * L[j, :j], axis=0)) / L[j, j]
    return L, ok


def _batch_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:

    p, _, B = L.shape
    z = np.zeros((p, B))
    for i in range(p):
        acc = b[i].copy()
        for k in range(i):
            acc -= L[i, k] * z[k]
        z[i] = acc / L[i, i]
    x = np.zeros((p, B))
    for i in range(p - 1, -1, -1):
        acc = z[i].copy()
        for k in range(i + 1, p):
            acc -= L[k, i] * x[k]
        x[i] = acc / L[i, i]
    return x


def _batch_inv_diag(L: np.ndarray) -> np.ndarray:

    p, _, B = L.shape
    Linv = np.zeros((p, p, B))
    for j in range(p):
        Linv[j, j] = 1.0 / L[j, j]
        for i in range(j + 1, p):
            acc = np.zeros(B)
            for k in range(j, i):
                acc += L[i, k] * Linv[k, j]
            Linv[i, j] = -acc / L[i, i]
    return np.sum(Linv ** 2, axis=0)


@dataclass
class GWASResult:


    beta: np.ndarray
    se: np.ndarray
    odds_ratio: np.ndarray
    stat: np.ndarray
    p: np.ndarray
    num_obs: np.ndarray
    converged: np.ndarray


def gwas_logistic(X: np.ndarray, y: np.ndarray, cov: Optional[np.ndarray] = None,
                  chunk_size: int = 5000, **opts) -> GWASResult:


    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    s, n = X.shape
    if y.size != n:
        raise ValueError("y must have one entry per column of X.")
    cov = np.zeros((n, 0)) if cov is None else np.asarray(cov, dtype=float)
    if cov.shape[0] != n:
        raise ValueError("cov must have n rows.")
    C = np.column_stack([np.ones(n), cov])
    beta = np.full(s, np.nan)
    se = np.full(s, np.nan)
    nobs = np.zeros(s, dtype=np.int64)
    conv = np.zeros(s, dtype=bool)
    for first in range(0, s, chunk_size):
        idx = np.arange(first, min(first + chunk_size, s))
        V = X[idx].T
        out = logistic_batch(y, C, V[:, :, None], np.isfinite(V), **opts)
        beta[idx] = out.coef[-1]
        se[idx] = out.se[-1]
        nobs[idx] = out.num_obs
        conv[idx] = out.converged
    stat = beta / se
    return GWASResult(beta=beta, se=se, odds_ratio=np.exp(beta), stat=stat,
                      p=special.erfc(np.abs(stat) / np.sqrt(2.0)), num_obs=nobs, converged=conv)


@dataclass
class EpistasisResult:


    pairs: np.ndarray
    pair_beta3: np.ndarray
    pair_se3: np.ndarray
    pair_or_int: np.ndarray
    pair_stat: np.ndarray
    pair_p: np.ndarray
    beta3: np.ndarray
    se3: np.ndarray
    or_int: np.ndarray
    stat: np.ndarray
    p: np.ndarray
    num_obs: np.ndarray


def epistasis_test(X: np.ndarray, y: np.ndarray, pairs: Optional[np.ndarray] = None,
                   cov: Optional[np.ndarray] = None, chunk_size: int = 2000, **opts) -> EpistasisResult:


    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    s, n = X.shape
    if y.size != n:
        raise ValueError("y must have one entry per column of X.")
    if pairs is None:
        jj, kk = np.triu_indices(s, k=1)
        pairs = np.column_stack([jj, kk]).astype(np.int64)
    else:
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if np.any((pairs < 0) | (pairs >= s)) or np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("pairs must be a (P, 2) list of distinct SNP indices.")
    cov = np.zeros((n, 0)) if cov is None else np.asarray(cov, dtype=float)
    C = np.column_stack([np.ones(n), cov])
    Xt = X.T
    num_pair = pairs.shape[0]
    b3 = np.full(num_pair, np.nan)
    se3 = np.full(num_pair, np.nan)
    nobs = np.zeros(num_pair)
    for first in range(0, num_pair, chunk_size):
        idx = np.arange(first, min(first + chunk_size, num_pair))
        A = Xt[:, pairs[idx, 0]]
        Bm = Xt[:, pairs[idx, 1]]
        V = np.stack([A, Bm, A * Bm], axis=2)
        out = logistic_batch(y, C, V, np.isfinite(A) & np.isfinite(Bm), **opts)
        b3[idx] = out.coef[-1]
        se3[idx] = out.se[-1]
        nobs[idx] = out.num_obs
    stat = (b3 / se3) ** 2
    pval = special.erfc(np.sqrt(stat / 2.0))
    orint = np.exp(b3)
    return EpistasisResult(
        pairs=pairs, pair_beta3=b3, pair_se3=se3, pair_or_int=orint, pair_stat=stat, pair_p=pval,
        beta3=pairs_to_matrix(pairs, b3, s), se3=pairs_to_matrix(pairs, se3, s),
        or_int=pairs_to_matrix(pairs, orint, s), stat=pairs_to_matrix(pairs, stat, s),
        p=pairs_to_matrix(pairs, pval, s), num_obs=pairs_to_matrix(pairs, nobs, s))


def pairs_to_matrix(pairs: np.ndarray, values: np.ndarray, s: int) -> np.ndarray:

    A = np.full((s, s), np.nan)
    if len(pairs):
        A[pairs[:, 0], pairs[:, 1]] = values
        A[pairs[:, 1], pairs[:, 0]] = values
    return A


def hwe_exact_test(hom1: np.ndarray, het: np.ndarray, hom2: np.ndarray, mid_p: bool = False) -> np.ndarray:


    a = np.asarray(hom1, dtype=float).ravel()
    h = np.asarray(het, dtype=float).ravel()
    b = np.asarray(hom2, dtype=float).ravel()
    if not (a.size == h.size == b.size):
        raise ValueError("genotype count vectors must have equal length.")
    counts = np.concatenate([a, h, b])
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("genotype counts must be non-negative integers.")
    N = a + h + b
    nA = 2 * a + h
    n_minor = np.minimum(nA, 2 * N - nA)
    P = np.full(a.size, np.nan)
    valid = np.flatnonzero(N > 0)
    if valid.size == 0:
        return P
    keys = np.column_stack([N[valid], n_minor[valid]])
    ukeys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    for u, (nn, na) in enumerate(ukeys):
        nn, na = int(nn), int(na)
        nb = 2 * nn - na
        hets = np.arange(na % 2, na + 1, 2, dtype=float)
        h1 = (na - hets) / 2
        h2 = (nb - hets) / 2
        logp = (special.gammaln(nn + 1) - special.gammaln(h1 + 1) - special.gammaln(hets + 1)
                - special.gammaln(h2 + 1) + hets * np.log(2.0) + special.gammaln(na + 1)
                + special.gammaln(nb + 1) - special.gammaln(2 * nn + 1))
        prob = np.exp(logp - logp.max())
        prob /= prob.sum()
        members = valid[inverse == u]
        for obs_h in np.unique(h[members]):
            pos = np.flatnonzero(hets == obs_h)
            if pos.size == 0:
                raise ValueError("infeasible genotype counts.")
            p_obs = prob[pos[0]]
            if mid_p:
                pv = prob[prob < p_obs * (1 - 1e-7)].sum() + 0.5 * prob[np.abs(prob - p_obs) <= p_obs * 1e-7].sum()
            else:
                pv = prob[prob <= p_obs * (1 + 1e-7)].sum()
            P[members[h[members] == obs_h]] = min(1.0, pv)
    return P


def two_sample_ttest(A: np.ndarray, B: np.ndarray, kind: str = "pooled") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:


    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n1, n2 = A.shape[1], B.shape[1]
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of rows.")
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least two observations.")
    m1, m2 = A.mean(axis=1), B.mean(axis=1)
    v1, v2 = A.var(axis=1, ddof=1), B.var(axis=1, ddof=1)
    if kind == "pooled":
        df = np.full(m1.shape, float(n1 + n2 - 2))
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    elif kind == "welch":
        a1, a2 = v1 / n1, v2 / n2
        se = np.sqrt(a1 + a2)
        with np.errstate(invalid="ignore", divide="ignore"):
            df = (a1 + a2) ** 2 / (a1 ** 2 / (n1 - 1) + a2 ** 2 / (n2 - 1))
    else:
        raise ValueError("kind must be 'pooled' or 'welch'.")
    with np.errstate(invalid="ignore", divide="ignore"):
        T = (m1 - m2) / se
    P = np.full(T.shape, np.nan)
    ok = ~np.isnan(T) & ~np.isnan(df)
    P[ok] = special.betainc(df[ok] / 2.0, 0.5, df[ok] / (df[ok] + T[ok] ** 2))
    P[np.isinf(T)] = 0.0
    return P, T, df


def ks_test2_asymptotic(x1: np.ndarray, x2: np.ndarray) -> Tuple[float, float]:


    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    n1, n2 = x1.size, x2.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    v = np.concatenate([x1, x2])
    lab = np.concatenate([np.ones(n1), np.zeros(n2)])
    order = np.argsort(v, kind="stable")
    vs, ls = v[order], lab[order]
    cdf_diff = np.cumsum(ls / n1 - (1 - ls) / n2)
    last_of_tie = np.concatenate([np.flatnonzero(np.diff(vs) != 0), [vs.size - 1]])
    D = float(np.max(np.abs(cdf_diff[last_of_tie])))
    ne = n1 * n2 / (n1 + n2)
    lam = max((np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * D, 0.0)
    j = np.arange(1, 102)
    P = 2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * lam ** 2 * j ** 2))
    return float(min(max(P, 0.0), 1.0)), D


def compute_auc(score: np.ndarray, label: np.ndarray) -> float:


    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label, dtype=float).ravel()
    if score.size != label.size:
        raise ValueError("score and label must have the same number of elements.")
    if np.any((label != 0) & (label != 1)):
        raise ValueError("label must be binary (0/1).")
    if np.any(np.isnan(score)):
        raise ValueError("score contains NaN.")
    n_pos = int(np.sum(label == 1))
    n_neg = int(np.sum(label == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    xs = score[order]
    group = np.concatenate([[0], np.cumsum(np.diff(xs) != 0)])
    ranks = np.arange(1, score.size + 1, dtype=float)
    avg = np.bincount(group, weights=ranks) / np.bincount(group)
    mid_rank = np.empty(score.size)
    mid_rank[order] = avg[group]
    return float((mid_rank[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
