"""Stage 1 of NetPRS: SNP interaction networks (Sec. II of the paper).

* :func:`phenotypic_network`  W_P from the epistasis test (Sec. II-A)
* :func:`genomic_network`     W_G from SNP-gene relations and the gene-gene
  interaction network by graph-based semi-supervised learning (Sec. II-B)
* :func:`graph_laplacian`, :func:`elbow_threshold`, :func:`snp_gene_matrix`,
  :func:`network_summary` (Table III)

Each function mirrors the MATLAB function of the same name.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from .stats import EpistasisResult, ks_test2_asymptotic

__all__ = [
    "graph_laplacian", "PhenotypicInfo", "phenotypic_network", "ElbowInfo", "elbow_threshold",
    "GenomicInfo", "genomic_network", "snp_gene_matrix", "NetworkStats", "NetworkSummary",
    "network_summary",
]

Matrix = Union[np.ndarray, sp.spmatrix]


def graph_laplacian(W: Matrix, kind: str = "unnormalized") -> Matrix:
    """Graph Laplacian ``L = D - W`` (paper) or ``I - D^(-1/2) W D^(-1/2)``.

    ``W`` must be square, symmetric, finite and non-negative (dense or sparse);
    the result has the same storage. Isolated nodes use ``D^(-1/2) = 0`` in the
    normalized form (as PPIxGPN/BIGPN).
    """
    _check_network(W)
    m = W.shape[0]
    deg = np.asarray(W.sum(axis=1)).ravel()
    if kind == "unnormalized":
        L = sp.diags(deg) - W if sp.issparse(W) else np.diag(deg) - W
    elif kind == "normalized":
        inv = np.zeros(m)
        pos = deg > 0
        inv[pos] = 1.0 / np.sqrt(deg[pos])
        if sp.issparse(W):
            Dh = sp.diags(inv)
            L = sp.identity(m) - Dh @ W @ Dh
        else:
            L = np.eye(m) - np.outer(inv, inv) * W
    else:
        raise ValueError("kind must be 'unnormalized' or 'normalized'.")
    L = (L + L.T) / 2.0
    return L.tocsr() if sp.issparse(L) else np.asarray(L)


def _check_network(W: Matrix) -> None:
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix.")
    v = W.data if sp.issparse(W) else W[W != 0]
    if np.any(~np.isfinite(v)):
        raise ValueError("W must contain finite values only.")
    if np.any(v < 0):
        raise ValueError("W must be non-negative.")
    asym = W - W.T
    a = np.abs(asym.data) if sp.issparse(asym) else np.abs(asym[asym != 0])
    if a.size and a.max() > 1e-12 * max(np.abs(v).max() if v.size else 0.0, 1.0):
        raise ValueError("W must be symmetric (undirected network).")


# ---------------------------------------------------------------------------
# Phenotypic network W_P
# ---------------------------------------------------------------------------
@dataclass
class PhenotypicInfo:
    tp: np.ndarray          # T_P of every tested pair (upper triangle, NaN elsewhere)
    num_edge: int
    density: float
    p_threshold: float


def phenotypic_network(or_int: Union[np.ndarray, EpistasisResult], pval: Optional[np.ndarray] = None,
                       p_threshold: float = 0.05) -> Tuple[np.ndarray, PhenotypicInfo]:
    """Phenotypic SNP interaction network W_P (Sec. II-A).

    ``W_P^{jk} = T_P(OR_INT^{jk}) = 1 - exp(-|ln OR_INT|)`` for the pairs with
    epistasis ``P < p_threshold`` (0.05 in the paper). Accepts an
    :class:`EpistasisResult` or the two (s, s) matrices (upper triangle read).
    """
    if isinstance(or_int, EpistasisResult):
        pval = or_int.p
        or_int = or_int.or_int
    if pval is None:
        raise ValueError("pval is required with an OR_INT matrix.")
    or_int = np.asarray(or_int, dtype=float)
    pval = np.asarray(pval, dtype=float)
    s = or_int.shape[0]
    if or_int.shape != (s, s) or pval.shape != (s, s):
        raise ValueError("or_int and pval must be s x s matrices.")
    upper = np.triu(np.ones((s, s), dtype=bool), k=1)
    tp = np.full((s, s), np.nan)
    h = or_int[upper]
    t = np.full(h.shape, np.nan)
    valid = np.isfinite(h) & (h > 0)
    t[valid] = 1.0 - np.exp(-np.abs(np.log(h[valid])))
    tp[upper] = t
    with np.errstate(invalid="ignore"):
        keep = upper & (pval < p_threshold) & np.isfinite(tp)
    Wu = np.zeros((s, s))
    Wu[keep] = tp[keep]
    num_edge = int(keep.sum())
    density = num_edge / (s * (s - 1) / 2) if s > 1 else float("nan")
    return Wu + Wu.T, PhenotypicInfo(tp=tp, num_edge=num_edge, density=density, p_threshold=p_threshold)


# ---------------------------------------------------------------------------
# Elbow threshold
# ---------------------------------------------------------------------------
@dataclass
class ElbowInfo:
    index: int              # 0-based rank of the elbow in the sorted values
    sorted: np.ndarray      # values sorted in decreasing order
    distance: np.ndarray    # normalized distance of every point to the chord


def elbow_threshold(values: np.ndarray, scale: str = "linear") -> Tuple[float, ElbowInfo]:
    """Elbow point: the sorted value with the maximum distance to the chord.

    The values are sorted in decreasing order against their normalized rank;
    the value with the largest distance to the line through the first and the
    last point is returned (values ``>= tau`` are kept). ``scale='log'`` uses
    log10 values.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if scale == "log":
        v = v[v > 0]
    elif scale != "linear":
        raise ValueError("scale must be 'linear' or 'log'.")
    if v.size == 0:
        raise ValueError("no valid values to threshold.")
    v = -np.sort(-v, kind="stable")
    N = v.size
    yv = np.log10(v) if scale == "log" else v
    if N < 3 or yv[0] == yv[-1]:
        return float(v[-1]), ElbowInfo(index=N - 1, sorted=v, distance=np.zeros(N))
    x = np.arange(N) / (N - 1)
    y = (yv - yv[-1]) / (yv[0] - yv[-1])
    distance = np.abs(1.0 - x - y) / np.sqrt(2.0)
    idx = int(np.argmax(distance))
    return float(v[idx]), ElbowInfo(index=idx, sorted=v, distance=distance)


# ---------------------------------------------------------------------------
# Genomic network W_G
# ---------------------------------------------------------------------------
@dataclass
class GenomicInfo:
    H: np.ndarray           # h_G (s, s), zero diagonal
    Tg: np.ndarray          # T_G(h_G)
    threshold: float
    threshold_on: str
    num_edge: int
    density: float
    num_gene_used: int
    solver: str
    max_rel_residual: float


def genomic_network(G: Matrix, Wggi: Matrix, mu: float = 1.0, sigma: float = 1.0,
                    threshold: Union[str, float] = "elbow", threshold_on: str = "transformed",
                    elbow_scale: str = "linear", laplacian: str = "unnormalized", solver: str = "auto",
                    dense_max: int = 12000, pcg_tol: float = 1e-10, pcg_max_iter: int = 2000,
                    tie_tol: float = 1e-9) -> Tuple[np.ndarray, GenomicInfo]:
    """Genomic SNP interaction network W_G (Sec. II-B).

    For SNP j, ``f^j = (I + mu L)^{-1} g^j`` (Eq. (3), the minimizer of Eq. (2))
    propagates its SNP-gene relation ``g^j`` (column of ``G``, genes x SNPs) on the
    gene-gene interaction network; ``h_G^{jk} = f^j' g^k`` is transformed by
    ``T_G(x) = exp(ln(x)/sigma^2)`` and thresholded at the elbow point (or at a
    given value). Values within ``tie_tol * |tau|`` below the threshold are kept,
    so mathematically tied interactions are kept or dropped together.
    GGI components without any SNP-related gene are dropped (exact: ``(I + mu L)^{-1}``
    is block diagonal over the components).
    """
    G = sp.csc_matrix(G, dtype=float)
    W = sp.csr_matrix(Wggi, dtype=float)
    m, s = G.shape
    if W.shape != (m, m):
        raise ValueError("Wggi must be m x m with m = G.shape[0].")
    if np.any(G.data < 0) or np.any(~np.isfinite(G.data)):
        raise ValueError("G must be finite and non-negative.")
    if not (mu >= 0 and sigma > 0):
        raise ValueError("mu must be >= 0 and sigma > 0.")

    # keep the GGI components that contain at least one SNP-related gene
    _, comp = connected_components(W != 0, directed=False)
    related = np.asarray((G != 0).sum(axis=1)).ravel() > 0
    keep_gene = np.isin(comp, np.unique(comp[related]))
    Gk = G[keep_gene, :]
    Wk = W[keep_gene][:, keep_gene]
    mk = int(keep_gene.sum())

    L = graph_laplacian(Wk, laplacian)
    Q = (sp.identity(mk, format="csr") + mu * L).tocsr()
    if solver == "auto":
        solver = "dense" if mk <= dense_max else "pcg"
    cols = np.flatnonzero(np.asarray((Gk != 0).sum(axis=0)).ravel() > 0)
    F = np.zeros((mk, s))
    Gc = Gk[:, cols].toarray()
    if cols.size:
        if solver == "dense":
            try:
                factor = la.cho_factor(Q.toarray(), lower=False)
            except la.LinAlgError:
                raise ValueError("I + mu*L is not positive definite.") from None
            F[:, cols] = la.cho_solve(factor, Gc)
        elif solver == "sparse":
            F[:, cols] = np.asarray(spla.spsolve(Q.tocsc(), Gc)).reshape(mk, -1)
        elif solver == "pcg":
            dq = Q.diagonal()
            precond = spla.LinearOperator(Q.shape, matvec=lambda v: v / dq)
            for t, j in enumerate(cols):
                f, flag = _cg(Q, Gc[:, t], pcg_tol, pcg_max_iter, precond)
                if flag != 0:
                    raise ValueError(f"pcg did not converge for SNP {j} (flag {flag}).")
                F[:, j] = f
        else:
            raise ValueError(f"unknown solver '{solver}'.")
    max_rel_res = 0.0
    if cols.size:
        resid = Q @ F[:, cols] - Gc
        max_rel_res = float(np.max(np.sqrt(np.sum(resid ** 2, axis=0)) / np.sqrt(np.sum(Gc ** 2, axis=0))))

    H = np.asarray((Gk.T @ F).T)                          # h_G^{jk} = f^j' g^k  (F' g)
    H = (H + H.T) / 2.0
    np.fill_diagonal(H, 0.0)
    scale_h = np.max(np.abs(H)) if H.size else 0.0
    if np.any(H < -1e-8 * max(scale_h, np.finfo(float).eps)):
        raise ValueError("negative h_G beyond rounding error (check the inputs).")
    H[H < 0] = 0.0
    with np.errstate(divide="ignore"):
        Tg = np.exp(np.log(H) / sigma ** 2)                 # T_G(0) = 0

    if threshold_on == "transformed":
        Vt = Tg
    elif threshold_on == "raw":
        Vt = H
    else:
        raise ValueError("threshold_on must be 'transformed' or 'raw'.")
    upper = np.triu(np.ones((s, s), dtype=bool), k=1)
    vals = Vt[upper]
    positive = vals[vals > 0]
    if isinstance(threshold, str):
        if threshold == "elbow":
            tau = elbow_threshold(positive, elbow_scale)[0] if positive.size else float("nan")
        elif threshold == "none":
            tau = 0.0
        else:
            raise ValueError(f"unknown threshold '{threshold}'.")
    else:
        tau = float(threshold)
    if positive.size == 0:
        warnings.warn("No SNP pair has a positive genomic interaction (no shared GGI component); "
                      "W_G is empty.", RuntimeWarning, stacklevel=2)
    with np.errstate(invalid="ignore"):
        keep = upper & (Vt >= tau - tie_tol * abs(tau)) & (Vt > 0)
    Wu = np.zeros((s, s))
    Wu[keep] = Tg[keep]
    num_edge = int(keep.sum())
    info = GenomicInfo(H=H, Tg=Tg, threshold=tau, threshold_on=threshold_on, num_edge=num_edge,
                       density=num_edge / (s * (s - 1) / 2) if s > 1 else float("nan"),
                       num_gene_used=mk, solver=solver, max_rel_residual=max_rel_res)
    return Wu + Wu.T, info


def _cg(A, b, tol, maxiter, M):
    try:
        return spla.cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, M=M)
    except TypeError:                                      # SciPy < 1.12
        return spla.cg(A, b, tol=tol, atol=0.0, maxiter=maxiter, M=M)


def snp_gene_matrix(snp_ids: Sequence[str], gene_ids: Sequence[str], rel_snp: Sequence[str],
                    rel_gene: Sequence[str], rel_weight: Optional[np.ndarray] = None) -> sp.csc_matrix:
    """SNP-gene relation matrix ``g`` (genes x SNPs, sparse).

    ``g[i, j]`` is the weight of the relation between gene i and SNP j (1 for
    dbSNP links); relations with unknown IDs are ignored and duplicates keep
    their maximum weight.
    """
    if rel_weight is None:
        rel_weight = np.ones(len(rel_snp))
    rel_weight = np.asarray(rel_weight, dtype=float).ravel()
    if not (len(rel_snp) == len(rel_gene) == rel_weight.size):
        raise ValueError("rel_snp, rel_gene and rel_weight must have equal length.")
    if len(set(snp_ids)) != len(snp_ids) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("snp_ids and gene_ids must be unique.")
    if np.any(rel_weight < 0) or np.any(~np.isfinite(rel_weight)):
        raise ValueError("rel_weight must be finite and non-negative.")
    s, m = len(snp_ids), len(gene_ids)
    si = {x: j for j, x in enumerate(snp_ids)}
    gi = {x: i for i, x in enumerate(gene_ids)}
    js = np.array([si.get(x, -1) for x in rel_snp], dtype=np.int64)
    ig = np.array([gi.get(x, -1) for x in rel_gene], dtype=np.int64)
    ok = (js >= 0) & (ig >= 0)
    if not ok.any():
        return sp.csc_matrix((m, s))
    lin = ig[ok] * s + js[ok]
    w = rel_weight[ok]
    order = np.lexsort((-w, lin))
    lin_sorted = lin[order]
    first = np.ones(lin_sorted.size, dtype=bool)
    first[1:] = lin_sorted[1:] != lin_sorted[:-1]
    keep = order[first]
    return sp.csc_matrix((w[keep], (lin[keep] // s, lin[keep] % s)), shape=(m, s))


# ---------------------------------------------------------------------------
# Table III statistics
# ---------------------------------------------------------------------------
@dataclass
class NetworkStats:
    num_edge: int
    avg_edge: float
    density: float


@dataclass
class NetworkSummary:
    num_node: int
    phe: NetworkStats
    gen: NetworkStats
    num_common: int
    corr: float
    ks_p: float


def network_summary(Wp: Matrix, Wg: Matrix) -> NetworkSummary:
    """Statistics of W_P and W_G (Table III).

    Edges are the non-zero upper-triangle entries; density = edges / (s(s-1)/2).
    For the edges present in both networks: their number, the Pearson
    correlation of the two weights and the two-sample KS P-value (NaN with
    fewer than 3 common edges).
    """
    Wp = Wp.toarray() if sp.issparse(Wp) else np.asarray(Wp, dtype=float)
    Wg = Wg.toarray() if sp.issparse(Wg) else np.asarray(Wg, dtype=float)
    s = Wp.shape[0]
    if Wp.shape != (s, s) or Wg.shape != (s, s):
        raise ValueError("Wp and Wg must be s x s matrices.")
    upper = np.triu(np.ones((s, s), dtype=bool), k=1)
    wp, wg = Wp[upper], Wg[upper]
    num_pair = s * (s - 1) / 2

    def describe(w: np.ndarray) -> NetworkStats:
        e = w != 0
        return NetworkStats(num_edge=int(e.sum()), avg_edge=float(w[e].mean()) if e.any() else float("nan"),
                            density=float(e.sum() / num_pair) if num_pair else float("nan"))

    common = (wp != 0) & (wg != 0)
    nc = int(common.sum())
    if nc >= 3:
        corr = float(np.corrcoef(wp[common], wg[common])[0, 1])
        ksp = ks_test2_asymptotic(wp[common], wg[common])[0]
    else:
        corr, ksp = float("nan"), float("nan")
    return NetworkSummary(num_node=s, phe=describe(wp), gen=describe(wg), num_common=nc, corr=corr, ks_p=ksp)
