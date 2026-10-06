from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

from .data import NetPRSData
from .rng import RandomStream

__all__ = ["SyntheticTruth", "generate_synthetic_data"]

STREAM_GGI, STREAM_RELATION, STREAM_GENOTYPE, STREAM_PHENOTYPE, STREAM_DEFECT = 11, 12, 13, 14, 15
INTERCEPT = -1.0
MAJOR_EFFECT = 1.0
NETWORK_EFFECT = 0.5
NETWORK_MU = 1.0
NUM_SEED_GENE = 2
CAUSAL_NETWORK_EFFECT = 0.05


@dataclass
class SyntheticTruth:


    gene_module: np.ndarray
    risk_module: List[int]
    main_snp: List[np.ndarray]
    main_effect: List[np.ndarray]
    pair_snp: List[np.ndarray]
    pair_effect: List[np.ndarray]
    seed_gene: np.ndarray
    gene_signal: np.ndarray
    network_effect: np.ndarray
    maf: np.ndarray
    defect: Dict[str, np.ndarray]


def _round(x: np.ndarray, digits: int) -> np.ndarray:
    scale = 10.0 ** digits
    return np.floor(x * scale + 0.5) / scale


def generate_synthetic_data(num_subject: int = 1000, num_validation: int = 300,
                            num_snp: int = 300, num_gene: int = 120, num_module: int = 6,
                            seed: int = 20261006) -> Tuple[NetPRSData, SyntheticTruth]:

    n, s, m, k = int(num_subject), int(num_snp), int(num_gene), int(num_module)
    if not (0 <= num_validation < n and k >= 2 and m >= 2 * k and s >= 30):
        raise ValueError("Invalid synthetic data dimensions (two risk modules need num_module >= 2 "
                         "and num_gene >= 2 * num_module).")


    module = (np.arange(m) * k) // m
    ii, jj = np.triu_indices(m, k=1)
    rs = RandomStream(seed, STREAM_GGI)
    u_edge = rs.uniform(ii.size)
    u_weight = rs.uniform(ii.size)
    same = module[ii] == module[jj]
    edge = u_edge < np.where(same, 0.30, 0.01)
    weight = _round(np.where(same, 0.4 + 0.6 * u_weight, 0.15 + 0.35 * u_weight), 4)
    upper = sp.csr_matrix((weight[edge], (ii[edge], jj[edge])), shape=(m, m))
    ggi = (upper + upper.T).tocsr()


    u = RandomStream(seed, STREAM_RELATION).uniform(4 * s).reshape(s, 4)
    has_rel = u[:, 0] >= 0.10
    gene1 = np.floor(u[:, 1] * m).astype(np.int64)
    has_second = has_rel & (u[:, 2] < 0.20)
    rel_snp: List[int] = []
    rel_gene: List[int] = []
    for j in range(s):
        if not has_rel[j]:
            continue
        rel_snp.append(j)
        rel_gene.append(int(gene1[j]))
        if has_second[j]:
            cand = np.flatnonzero((module == module[gene1[j]]) & (np.arange(m) != gene1[j]))
            rel_snp.append(j)
            rel_gene.append(int(cand[int(np.floor(u[j, 3] * cand.size))]))


    rs = RandomStream(seed, STREAM_GENOTYPE)
    maf = 0.05 + 0.45 * rs.uniform(s)
    ua = rs.uniform(n * s).reshape(n, s)
    ub = rs.uniform(n * s).reshape(n, s)
    geno = ((ua < maf).astype(float) + (ub < maf).astype(float)).T


    rs = RandomStream(seed, STREAM_PHENOTYPE)
    snp_module = np.where(has_rel, module[gene1], -1)
    risk_module = [0, 1]
    main_snp, main_effect, pair_snp, pair_effect, seed_gene = [], [], [], [], []
    for r in risk_module:
        cand = np.flatnonzero((snp_module == r) & (maf > 0.15))
        if cand.size < 2:
            raise ValueError("too few common SNPs in a risk module; increase num_snp.")
        main = cand[rs.permutation(cand.size)[:8]]
        effect = 0.30 + 0.25 * rs.uniform(main.size)
        if r == risk_module[0]:
            effect[0] = MAJOR_EFFECT
        main_snp.append(main)
        main_effect.append(effect)
        pairs, effects = [], []
        for _ in range(6):
            perm = rs.permutation(cand.size)
            uc = rs.uniform(2)
            pairs.append((cand[perm[0]], cand[perm[1]]))
            effects.append((0.35 + 0.25 * uc[0]) * (1.0 if uc[1] < 0.65 else -1.0))
        pair_snp.append(np.array(pairs, dtype=np.int64))
        pair_effect.append(np.array(effects))
        genes_r = np.flatnonzero(module == r)
        seed_gene.extend(genes_r[rs.permutation(genes_r.size)[:NUM_SEED_GENE]])
    seed_gene = np.array(seed_gene, dtype=np.int64)
    s_ind = np.zeros(m)
    s_ind[seed_gene] = 1.0
    lap = np.diag(np.asarray(ggi.sum(axis=1)).ravel()) - ggi.toarray()
    gene_signal = np.linalg.solve(np.eye(m) + NETWORK_MU * lap, s_ind)
    relation = np.zeros((m, s))
    relation[rel_gene, rel_snp] = 1.0
    v = relation.T @ gene_signal
    network_effect = _round(NETWORK_EFFECT * v / v.max(), 8)
    eta = np.full(n, INTERCEPT)
    for t in range(len(risk_module)):
        for j, a in zip(main_snp[t], main_effect[t]):
            eta = eta + a * (geno[j] - 2 * maf[j])
        for (j, jk), c in zip(pair_snp[t], pair_effect[t]):
            eta = eta + c * (geno[j] * geno[jk] - 4 * maf[j] * maf[jk])
    for j in range(s):
        eta = eta + network_effect[j] * (geno[j] - 2 * maf[j])
    prob = 1.0 / (1.0 + np.exp(-eta))
    diagnosis = (rs.uniform(n) < prob).astype(float)


    rs = RandomStream(seed, STREAM_DEFECT)
    causal = np.unique(np.concatenate([*main_snp, *(p.ravel() for p in pair_snp),
                                       np.flatnonzero(network_effect >= CAUSAL_NETWORK_EFFECT)]))
    non_causal = np.setdiff1d(np.arange(s), causal)
    perm = non_causal[rs.permutation(non_causal.size)]
    defect = {"missing": perm[0:3], "rare": perm[3:6], "hwe": perm[6:9]}
    for j in defect["missing"]:
        geno[j, rs.uniform(n) < 0.05] = np.nan
    for j in defect["rare"]:
        geno[j] = (rs.uniform(n) < 0.005).astype(float)
    for j in defect["hwe"]:
        uh = rs.uniform(n)
        geno[j] = 1.0
        geno[j, uh < 0.05] = 0.0
    miss = rs.uniform(n * s).reshape(n, s) < 0.002
    geno[miss.T] = np.nan

    width_s = max(3, len(str(s)))
    width_n = max(4, len(str(n)))
    width_m = max(3, len(str(m)))
    snp_id = [f"snp{j + 1:0{width_s}d}" for j in range(s)]
    gene_id = [f"GENE{i + 1:0{width_m}d}" for i in range(m)]
    data = NetPRSData(
        genotype=geno, diagnosis=diagnosis,
        is_validation=np.arange(n) >= n - num_validation,
        subject_id=[f"S{i + 1:0{width_n}d}" for i in range(n)],
        snp_id=snp_id, gene_id=gene_id, ggi=ggi,
        rel_snp=[snp_id[j] for j in rel_snp], rel_gene=[gene_id[g] for g in rel_gene],
        rel_weight=np.ones(len(rel_snp)), source="synthetic")
    truth = SyntheticTruth(gene_module=module, risk_module=risk_module, main_snp=main_snp,
                           main_effect=main_effect, pair_snp=pair_snp, pair_effect=pair_effect,
                           seed_gene=seed_gene, gene_signal=gene_signal, network_effect=network_effect,
                           maf=maf, defect=defect)
    return data, truth
