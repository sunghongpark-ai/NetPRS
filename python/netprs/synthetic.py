"""Synthetic NetPRS data (the generator of ``dataset/sample.csv``).

SYNTHETIC DATA ONLY: the cohorts of the paper (ADNI, BICWALZS), dbSNP and
STRING extracts are not redistributable. The generator reproduces their
structure (genotypes of a discovery and a validation cohort, binary AD
diagnosis, SNP-gene relations and a modular gene-gene interaction network) so
that the whole pipeline can be executed. It says nothing about AD biology.

The algorithm draws every random number from :class:`netprs.rng.RandomStream`
in a fixed order. Apart from element-wise arithmetic it uses one linear solve
and one matrix product, whose results are rounded to 8 decimals, so the MATLAB
function ``GenerateSyntheticData.m`` produces the same data and, through
``WriteNetPRSCSV.m``, a byte-identical CSV file (verified in Octave and Python).

Generative model (defaults in brackets)
---------------------------------------
* Genes [120] form equal-sized modules [6]. A gene pair is linked with
  probability 0.30 within a module (weight U(0.4, 1)) and 0.01 between modules
  (weight U(0.15, 0.5)); weights are rounded to 4 decimals.
* Every SNP [300] is related to one random gene with probability 0.9 and,
  with probability 0.2, to a second gene of the same module.
* Genotypes: independent Binomial(2, MAF), MAF ~ U(0.05, 0.5).
* Diagnosis: ``logit P(AD) = -1 + sum a_j (x_j - 2 p_j)
  + sum c_jk (x_j x_k - 4 p_j p_k) + sum d_j (x_j - 2 p_j)`` (expected, not
  sample, centring). The three terms mirror the three effects of NetPRS:

  - independent effects: 8 main-effect SNPs in each of two risk modules
    (a ~ U(0.30, 0.55); the first one is a major locus with a = 1.0, like
    APOE e4);
  - phenotypic interactions: 6 epistatic SNP pairs per risk module
    (|c| ~ U(0.35, 0.60), 65% positive);
  - genomic interactions: 2 seed genes per risk module whose signal is
    propagated on the gene-gene network by the GSSL closed form of Eq. (3),
    ``e = (I + L)^{-1} s`` (``s`` = seed indicator, ``L = D - W``), and passed to
    the SNPs through the SNP-gene relations: ``d = 0.5 * g' e / max(g' e)``
    (rounded to 8 decimals), so SNPs of interacting genes share effects.
* Quality-control defects among non-causal SNPs (no main or epistatic effect,
  network effect below 0.05): 3 SNPs with 5% missing calls, 3 rare SNPs
  (MAF ~ 0.25%), 3 SNPs violating HWE; then 0.2% missing calls everywhere.
* The last ``num_validation`` subjects form the validation cohort.
"""

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
NETWORK_EFFECT = 0.5           # largest SNP effect of the gene-network term
NETWORK_MU = 1.0               # smoothness of the propagation (as mu of Eq. (3))
NUM_SEED_GENE = 2              # seed genes per risk module
CAUSAL_NETWORK_EFFECT = 0.05   # SNPs with a larger network effect never get QC defects


@dataclass
class SyntheticTruth:
    """Generating parameters (0-based indices)."""

    gene_module: np.ndarray
    risk_module: List[int]
    main_snp: List[np.ndarray]
    main_effect: List[np.ndarray]
    pair_snp: List[np.ndarray]
    pair_effect: List[np.ndarray]
    seed_gene: np.ndarray           # seed genes of the gene-network term
    gene_signal: np.ndarray         # e = (I + mu L)^{-1} s
    network_effect: np.ndarray      # d, SNP effects of the gene-network term
    maf: np.ndarray
    defect: Dict[str, np.ndarray]


def _round(x: np.ndarray, digits: int) -> np.ndarray:
    scale = 10.0 ** digits
    return np.floor(x * scale + 0.5) / scale


def generate_synthetic_data(num_subject: int = 1000, num_validation: int = 300,
                            num_snp: int = 300, num_gene: int = 120, num_module: int = 6,
                            seed: int = 2024) -> Tuple[NetPRSData, SyntheticTruth]:
    """Generate a synthetic NetPRS dataset (see the module docstring)."""
    n, s, m, k = int(num_subject), int(num_snp), int(num_gene), int(num_module)
    if not (0 <= num_validation < n and k >= 2 and m >= 2 * k and s >= 30):
        raise ValueError("Invalid synthetic data dimensions (two risk modules need num_module >= 2 "
                         "and num_gene >= 2 * num_module).")

    # Gene-gene interaction network --------------------------------------
    module = (np.arange(m) * k) // m
    ii, jj = np.triu_indices(m, k=1)                    # row-major upper triangle
    rs = RandomStream(seed, STREAM_GGI)
    u_edge = rs.uniform(ii.size)
    u_weight = rs.uniform(ii.size)
    same = module[ii] == module[jj]
    edge = u_edge < np.where(same, 0.30, 0.01)
    weight = _round(np.where(same, 0.4 + 0.6 * u_weight, 0.15 + 0.35 * u_weight), 4)
    upper = sp.csr_matrix((weight[edge], (ii[edge], jj[edge])), shape=(m, m))
    ggi = (upper + upper.T).tocsr()

    # SNP-gene relations ---------------------------------------------------
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

    # Genotypes --------------------------------------------------------------
    rs = RandomStream(seed, STREAM_GENOTYPE)
    maf = 0.05 + 0.45 * rs.uniform(s)
    ua = rs.uniform(n * s).reshape(n, s)
    ub = rs.uniform(n * s).reshape(n, s)
    geno = ((ua < maf).astype(float) + (ub < maf).astype(float)).T     # (s, n)

    # Diagnosis -----------------------------------------------------------------
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
            effect[0] = MAJOR_EFFECT                    # one APOE-like major locus
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
    gene_signal = np.linalg.solve(np.eye(m) + NETWORK_MU * lap, s_ind)        # Eq. (3)
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

    # Quality-control defects and missing calls ---------------------------
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
