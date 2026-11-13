"""Tensor-cell2cell style decomposition of LR communication scores.

Two tensors are supported:

A. samples x LR pairs x methods      (build_tensor_a, from the per-method tables)
B. samples x LR pairs x sender x receiver lineage
                                     (build_tensor_b, from COMMOT obsp matrices)

Both are decomposed with non-negative CP fitted to the observed entries only
(masked_nn_cp), the same model Tensor-cell2cell fits. The decomposition never sees the condition labels, so the
labels can be permuted afterwards to test the sample loadings (loading_test).
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import tensorly as tl
from scipy.optimize import linear_sum_assignment

GROUP_COLORS = {'adult': '#2a78d6', 'fetal': '#eb6834'}
NEUTRAL = '#8a8984'


# =========================================================================
# 1. Container
# =========================================================================

@dataclass
class CommTensor:
    data: np.ndarray                 # missing entries hold 0; see mask
    mask: np.ndarray                 # 1 = observed, 0 = missing
    mode_names: list
    mode_labels: list                # one list of labels per mode
    info: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape

    def missing_fraction(self):
        return 1 - self.mask.mean()

    def save(self, path):
        np.savez_compressed(path, data=self.data, mask=self.mask,
                            mode_names=np.array(self.mode_names, dtype=object),
                            mode_labels=np.array([np.array(l, dtype=object) for l in self.mode_labels],
                                                 dtype=object))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(z['data'], z['mask'], list(z['mode_names']),
                   [list(l) for l in z['mode_labels']])


# =========================================================================
# 2. Transforms
# =========================================================================

def transform_table(tab, clip_negative=True, sample_norm=False):
    """Put one method's (LR x sample) table on a non-negative log scale.

    - clip_negative: scores below 0 become 0. For SpatialDM's global z this reads
      "no more co-localisation than chance" as no communication.
    - sample_norm: divide each sample by its median score over the pairs observed
      in every sample (a DESeq-style size factor). Off by default because it also
      removes any real sample-wide shift between conditions; use it as a
      sensitivity check.
    - Every method is then divided by ONE global scale (median positive score)
      and log1p'd, so differences between samples are kept.
    Missing values stay missing.
    """
    X = tab.astype(float)
    if clip_negative:
        X = X.clip(lower=0)
    if sample_norm:
        ref = X.dropna()
        ref = ref if len(ref) >= 10 else X
        size = ref.where(ref > 0).median()
        X = X / (size / size.median())
    vals = X.to_numpy()
    scale = np.nanmedian(vals[vals > 0])
    return np.log1p(X / scale)


# =========================================================================
# 3. Tensor A: samples x LR x methods
# =========================================================================

def build_tensor_a(method_tabs, samples, min_obs=None, sample_norm=False):
    """Stack the per-method tables into a masked samples x LR x methods tensor.

    method_tabs : dict method -> DataFrame (LR x sample), raw scores.
    min_obs     : a (pair, method) row is kept only if that method scored the pair
                  in at least this many samples (default: all of them). The
                  filter ignores the labels, so the later permutation test stays
                  valid. Lowering it lets in pairs missing from a whole group;
                  their loadings are then unconstrained for that group and can
                  top a factor for no reason (see presence_absence instead).
    Pairs are the union over methods; a pair a method never scored is masked,
    not zero-filled. Each method's slice is divided by its RMS so that the
    methods pull equally on the fit.
    """
    methods = list(method_tabs)
    min_obs = len(samples) if min_obs is None else min_obs
    logged = {m: transform_table(t.reindex(columns=samples), sample_norm=sample_norm)
              for m, t in method_tabs.items()}
    kept = {m: t[t.notna().sum(1) >= min_obs] for m, t in logged.items()}
    pairs = sorted(set().union(*[t.index for t in kept.values()]))

    data = np.zeros((len(samples), len(pairs), len(methods)))
    mask = np.zeros_like(data)
    for k, m in enumerate(methods):
        t = kept[m].reindex(pairs)
        v = t.to_numpy().T                      # samples x pairs
        obs = ~np.isnan(v)
        rms = np.sqrt(np.mean(v[obs] ** 2))
        data[:, :, k] = np.where(obs, v / rms, 0.0)
        mask[:, :, k] = obs

    info = {'pairs_per_method': {m: len(t) for m, t in kept.items()},
            'sample_norm': sample_norm, 'min_obs': min_obs}
    return CommTensor(data, mask, ['Samples', 'LR pairs', 'Methods'],
                      [list(samples), pairs, methods], info)


def presence_absence(method_tabs, g1, g2):
    """Pairs a method scored in every sample of one group and none of the other.

    These on/off pairs can't be tested by SAM or placed on a factor, since one
    group has no values at all; missing usually means the method's expression
    filter dropped the pair there. Listed separately, not as calibrated calls.
    """
    rows = []
    for m, t in method_tabs.items():
        a, b = t.reindex(columns=g1).notna(), t.reindex(columns=g2).notna()
        for name, on in (('adult only', a.all(axis=1) & ~b.any(axis=1)),
                         ('fetal only', b.all(axis=1) & ~a.any(axis=1))):
            for pair in t.index[on]:
                rows.append({'method': m, 'pair': pair, 'pattern': name})
    return pd.DataFrame(rows, columns=['method', 'pair', 'pattern'])


# =========================================================================
# 4. Tensor B: samples x LR x sender x receiver (COMMOT)
# =========================================================================

# Broad lineages shared by the adult (25-type) and fetal (94-type) references.
# Types mapped to None (erythroid, mesothelium) exist in only one reference and
# are left out rather than given a lineage the other reference can't have.
LINEAGES = ['Epithelial', 'Immune', 'Endothelial', 'Fibroblast', 'Pericyte', 'Muscle', 'Neural']

LINEAGE_MAP = {
    # ---- adult reference ----
    'Colonocytes': 'Epithelial', 'Crypt Top Colonocytes': 'Epithelial', 'Goblets': 'Epithelial',
    'Undifferentiated': 'Epithelial', 'BEST4+/OTOP2+ Cell': 'Epithelial',
    'Enteroendocrines': 'Epithelial',
    'Cycling Cells': 'Epithelial',      # in adult colon mostly crypt TA cells; an assumption
    'Glial': 'Neural',
    'Stromal 1': 'Fibroblast', 'Stromal 2': 'Fibroblast', 'Stromal 3': 'Fibroblast',
    'Stromal 4': 'Fibroblast',
    'Endothelial 1': 'Endothelial', 'Endothelial 2': 'Endothelial',
    'Myofibroblasts': 'Muscle', 'Pericytes': 'Pericyte',
    'ILCs': 'Immune', 'T-Cells': 'Immune', 'Macrophages&Monocytes': 'Immune', 'DC2': 'Immune',
    'DC1': 'Immune', 'NK': 'Immune', 'Mast Cells': 'Immune', 'B-Cells': 'Immune',
    'Plasma Cells': 'Immune',
    # ---- fetal reference ----
    'pDCs': 'Immune', 'Cell Cycle': 'Immune', 'Type 3 ILCs': 'Immune', 'DCs': 'Immune',
    'Mast': 'Immune', 'pre B-Cells': 'Immune', 'NKs & γδ': 'Immune', 'Monocytes': 'Immune',
    'Naive T-Cells': 'Immune', 'Macrophages SPP1+': 'Immune', 'Macrophages': 'Immune',
    'Lymphatic': 'Endothelial', 'Venous (L)': 'Endothelial', 'Arterial (CP) 2': 'Endothelial',
    'Venous (CP) 1': 'Endothelial', 'Venous (M) 2': 'Endothelial', 'Arterial (L)': 'Endothelial',
    'Endothelium G2M-phase': 'Endothelial', 'Venous (CP) 2': 'Endothelial',
    'Arterial (CP)': 'Endothelial', 'Arterial (M)': 'Endothelial', 'Venous (M) 1': 'Endothelial',
    'Endothelium S-phase': 'Endothelial',
    'Proximal Progenitor': 'Epithelial', 'Proximal Mature Enterocytes': 'Epithelial',
    'BEST4/OTOP2 Cells': 'Epithelial', 'Proximal TA': 'Epithelial',
    'Proximal Early Enterocytes': 'Epithelial', 'Proximal Enterocytes': 'Epithelial',
    'Proximal Stem Cells': 'Epithelial', 'EECs': 'Epithelial', 'Distal Enterocytes': 'Epithelial',
    'Distal TA': 'Epithelial', 'Distal Absorptive': 'Epithelial',
    'Distal Stem Cells': 'Epithelial', 'Secretory Progenitors': 'Epithelial',
    'Distal Mature Enterocytes': 'Epithelial',
    'S1': 'Fibroblast', 'S1 COL6A5+': 'Fibroblast', 'S4 CCL21+': 'Fibroblast',
    'Proximal S2 (2)': 'Fibroblast', 'S1 IFIT3+': 'Fibroblast', 'Distal S2': 'Fibroblast',
    'Fibroblasts S-phase': 'Fibroblast', 'Proximal S2 (1)': 'Fibroblast',
    'S3 Progenitor': 'Fibroblast', 'Fibroblasts G2M-phase': 'Fibroblast',
    'S4 CXCL14+': 'Fibroblast', 'S3': 'Fibroblast', 'Fibroblast Progenitor': 'Fibroblast',
    'S3 Transitional': 'Fibroblast', 'S3 EBF+': 'Fibroblast', 'S3 HAND1+': 'Fibroblast',
    'Erythroid': None,
    'BMPER+ Pericytes': 'Pericyte', 'Contractile Pericytes': 'Pericyte',
    'Pericytes S-phase': 'Pericyte', 'Angiogenic Pericytes': 'Pericyte',
    'Pericytes G2M-phase': 'Pericyte', 'WNT6+ Pericytes': 'Pericyte',
    'Pericyte Progenitors': 'Pericyte', 'Undifferentiated Pericytes': 'Pericyte',
    'ICC PDGFRA+': 'Muscle', 'MYOCD+ Muscularis': 'Muscle', 'Muscularis S-phase': 'Muscle',
    'Muscularis G2M-phase': 'Muscle', 'HOXP+ Proximal Muscularis': 'Muscle', 'ICC KIT+': 'Muscle',
    'FOXF2+ Distal Muscularis': 'Muscle', 'FOXF2- Muscularis': 'Muscle',
    'MORN5+ Distal Muscularis': 'Muscle', 'IGF1+ Distal Muscularis': 'Muscle',
    'FOXF2+ Proximal Muscularis': 'Muscle', 'Myofibroblast Progenitors': 'Muscle',
    'Myofibroblasts S-phase': 'Muscle', 'Myofibroblasts G2M-phase': 'Muscle',
    'Mesothelium SOX6+': None, 'Mesothelium IL18+': None,
    'Neural G2M- & S-phase': 'Neural', 'Glial Progenitors': 'Neural',
    'Excitory Motor Neuron': 'Neural', 'Interneuron': 'Neural',
    'Differentiating Submucosal Glial': 'Neural', 'ENS Progenitors': 'Neural',
    'Inhibitory Motor Neuron Precursor': 'Neural', 'Submucosal Glial': 'Neural',
    'Inhibitory Motor Neuron': 'Neural', 'Intraganglionic Glial': 'Neural',
    'Neuroendocrine (2)': 'Neural', 'Lymphoid Associated Glial': 'Neural',
    'Neuroendocrine (1)': 'Neural',
}


def lineage_proportions(obs, lineage_map=LINEAGE_MAP, lineages=LINEAGES):
    """Sum per-spot deconvolution proportions into broad lineages (spots x lineages)."""
    unknown = [c for c in obs.columns if c not in lineage_map]
    if unknown:
        raise KeyError(f'cell types missing from lineage_map: {unknown}')
    P = np.zeros((len(obs), len(lineages)))
    for c in obs.columns:
        lin = lineage_map[c]
        if lin is not None:
            P[:, lineages.index(lin)] += obs[c].to_numpy(float)
    return P


def build_tensor_b(adatas, lineage_map=LINEAGE_MAP, lineages=LINEAGES,
                   normalize='spots', min_abundance=5.0, prefix='commot-cellchat'):
    """samples x LR x sender lineage x receiver lineage tensor from COMMOT output.

    adatas : dict sample -> AnnData or path to a *_commot.h5ad file. obsp[key][i, j]
             is COMMOT's signal sent from spot i to spot j; .obs holds per-spot
             cell-type proportions.
    For each LR pair the lineage-level signal is P.T @ W @ P, which splits each
    spot's sent/received signal by its proportions instead of forcing one label.
    normalize : 'spots' divides by the number of spots (signal per spot, so larger
                sections don't score higher); 'abundance' further divides by
                sender and receiver lineage abundance (signal per unit of each
                lineage, closer to Tensor-cell2cell's per-cell-type scores).
    min_abundance : a lineage with fewer spot-equivalents than this in a sample
                is treated as unobserved there (masked), not as silent.
    """
    import anndata as ad

    samples = list(adatas)
    slices, masks, lr_names, abund = [], [], None, {}
    for s in samples:
        a = adatas[s]
        a = ad.read_h5ad(a) if isinstance(a, str) else a
        lr = a.uns[f'{prefix}-info']['df_ligrec']
        names = [f'{l}_{r}' for l, r in zip(lr['ligand'], lr['receptor'])]
        keys = [f'{prefix}-{l}-{r}' for l, r in zip(lr['ligand'], lr['receptor'])]
        if lr_names is None:
            lr_names = names
        elif names != lr_names:
            raise ValueError(f'{s}: LR pairs differ from the first sample')

        P = lineage_proportions(a.obs, lineage_map, lineages)
        size = P.sum(0)
        abund[s] = dict(zip(lineages, size.round(1)))
        S = np.empty((len(keys), len(lineages), len(lineages)))
        for i, k in enumerate(keys):
            W = a.obsp[k]
            S[i] = P.T @ (W @ P)
        S /= a.n_obs
        if normalize == 'abundance':
            S /= np.outer(np.maximum(size, 1e-12), np.maximum(size, 1e-12)) / a.n_obs ** 2
        present = size >= min_abundance
        slices.append(S)
        masks.append(np.broadcast_to(np.outer(present, present), S.shape).astype(float))
        del a

    info = {'lineage_abundance': pd.DataFrame(abund).T, 'normalize': normalize}
    return _finish_tensor_b(np.stack(slices), np.stack(masks), samples, lr_names, lineages, info)


def _finish_tensor_b(data, mask, samples, lr_names, lineages, info):
    scale = np.median(data[(mask > 0) & (data > 0)])
    data = np.where(mask > 0, np.log1p(data / scale), 0.0)
    return CommTensor(data, mask, ['Samples', 'LR pairs', 'Senders', 'Receivers'],
                      [list(samples), list(lr_names), list(lineages), list(lineages)], info)


def tensor_b_from_slices(slices, abundance, lineages, min_abundance=5.0):
    """Tensor B from precomputed sender x receiver signal.

    slices    : dict sample -> dict LR pair -> (lineages x lineages) array,
                already normalised the way build_tensor_b would (e.g. per spot).
    abundance : DataFrame samples x lineages, spot-equivalents per lineage.
    Pairs missing from a sample are masked.
    """
    samples = list(slices)
    lr_names = sorted(set().union(*[s.keys() for s in slices.values()]))
    K = len(lineages)
    data = np.zeros((len(samples), len(lr_names), K, K))
    mask = np.zeros_like(data)
    for i, s in enumerate(samples):
        present = abundance.loc[s, list(lineages)].to_numpy() >= min_abundance
        for j, p in enumerate(lr_names):
            if p in slices[s]:
                data[i, j] = slices[s][p]
                mask[i, j] = np.outer(present, present)
    return _finish_tensor_b(data, mask, samples, lr_names, lineages,
                            {'lineage_abundance': abundance})


# =========================================================================
# 5. Decomposition
# =========================================================================

@dataclass
class Decomposition:
    weights: np.ndarray
    factors: dict                    # mode name -> DataFrame (labels x factors)
    error: float                     # relative error on observed entries
    run_errors: np.ndarray
    run_factors: list = field(repr=False, default_factory=list)

    @property
    def rank(self):
        return len(self.weights)

    def reconstruct(self):
        mats = [f.to_numpy() for f in self.factors.values()]
        return tl.cp_to_tensor((self.weights, mats))


def _normalize_cp(weights, factors):
    """Unit-norm columns, scale moved into weights, factors ordered by weight."""
    factors = [np.asarray(f, float).copy() for f in factors]
    w = np.asarray(weights, float).copy()
    for f in factors:
        n = np.linalg.norm(f, axis=0)
        n[n == 0] = 1
        f /= n
        w *= n
    order = np.argsort(-w)
    return w[order], [f[:, order] for f in factors]


def _rel_error(T, M, weights, factors):
    R = tl.cp_to_tensor((weights, factors))
    return np.linalg.norm(M * (T - R)) / np.linalg.norm(M * T)


def masked_nn_cp(T, M, rank, n_iter_max=2000, tol=1e-6, random_state=0):
    """Non-negative CP fitted to the observed entries only.

    Weighted multiplicative updates (Lee & Seung with a 0/1 weight tensor):
        A_n <- A_n * [(M*T)_(n) KR] / [(M*R)_(n) KR]
    which never increases ||M * (T - R)||. tensorly 0.9's masked
    non_negative_parafac imputes missing entries each sweep instead, and its
    error rose over iterations on these tensors, so it is not used here.
    """
    rng = np.random.default_rng(random_state)
    scale = (np.linalg.norm(M * T) / np.sqrt(M.sum())) ** (1 / T.ndim)
    factors = [rng.uniform(0.1, 1, (s, rank)) * scale for s in T.shape]
    MT = M * T
    MT_unf = [tl.unfold(MT, n) for n in range(T.ndim)]
    norm_T = np.linalg.norm(MT)
    prev = np.inf
    for it in range(n_iter_max):
        for n in range(T.ndim):
            kr = tl.tenalg.khatri_rao(factors, skip_matrix=n)
            R = tl.cp_to_tensor((np.ones(rank), factors))
            num = MT_unf[n] @ kr
            den = tl.unfold(M * R, n) @ kr + 1e-12
            factors[n] *= num / den
        if it % 10 == 0:
            err = np.linalg.norm(M * (T - tl.cp_to_tensor((np.ones(rank), factors)))) / norm_T
            if prev - err < tol:
                break
            prev = err
    return np.ones(rank), factors


def decompose(ct, rank, n_runs=10, n_iter_max=2000, tol=1e-6, seed=0):
    """Masked non-negative CP; keeps the run with the lowest error on observed entries."""
    T = ct.data.astype(float)
    M = ct.mask.astype(float)
    runs = []
    for r in range(n_runs):
        w, f = masked_nn_cp(T, M, rank, n_iter_max=n_iter_max, tol=tol, random_state=seed + r)
        w, f = _normalize_cp(w, f)
        runs.append((_rel_error(T, M, w, f), w, f))
    errs = np.array([e for e, _, _ in runs])
    err, w, f = runs[int(np.argmin(errs))]
    cols = [f'Factor {i + 1}' for i in range(rank)]
    factors = {name: pd.DataFrame(fm, index=labels, columns=cols)
               for name, labels, fm in zip(ct.mode_names, ct.mode_labels, f)}
    return Decomposition(w, factors, err, errs, [x[2] for x in runs])


def factor_match_score(fa, fb):
    """Similarity of two CP solutions in [0, 1] (1 = same factors up to order).

    Congruence of factor r with s is the product over modes of |cos|; factors are
    paired by the Hungarian algorithm and the matched congruences averaged.
    """
    C = np.ones((fa[0].shape[1], fb[0].shape[1]))
    for a, b in zip(fa, fb):
        an = a / np.maximum(np.linalg.norm(a, axis=0), 1e-12)
        bn = b / np.maximum(np.linalg.norm(b, axis=0), 1e-12)
        C *= np.abs(an.T @ bn)
    r, c = linear_sum_assignment(-C)
    return C[r, c].mean()


def stability(dec):
    """Mean factor match score of every run against the best run."""
    best = dec.run_factors[int(np.argmin(dec.run_errors))]
    others = [f for i, f in enumerate(dec.run_factors) if i != int(np.argmin(dec.run_errors))]
    return float(np.mean([factor_match_score(best, f) for f in others])) if others else np.nan


def rank_scan(ct, ranks=range(1, 6), n_runs=10, seed=0, **kw):
    """Error and stability per rank, for choosing the rank (label-blind)."""
    rows = []
    for r in ranks:
        d = decompose(ct, r, n_runs=n_runs, seed=seed, **kw)
        rows.append({'rank': r, 'error': d.error, 'error_sd': d.run_errors.std(),
                     'stability': stability(d) if r > 1 else 1.0})
    return pd.DataFrame(rows).set_index('rank')


# =========================================================================
# 6. Testing sample loadings
# =========================================================================

def _tstat(L, mask):
    A, B = L[mask], L[~mask]
    n1, n2 = len(A), len(B)
    sse = ((A - A.mean(0)) ** 2).sum(0) + ((B - B.mean(0)) ** 2).sum(0)
    se = np.sqrt((1 / n1 + 1 / n2) * sse / (n1 + n2 - 2))
    return (A.mean(0) - B.mean(0)) / np.maximum(se, 1e-12)


def loading_test(dec, g1, g2, mode='Samples', names=('adult', 'fetal')):
    """Compare each factor's sample loadings between two groups.

    Every relabelling of the samples into groups of the same sizes is enumerated.
    p_perm is per factor; p_maxT uses the maximum |t| over factors in each
    relabelling, which is valid even when the condition-linked factor is picked
    after looking. Both are bounded below by p_floor = 1 / (number of labellings),
    e.g. 1/15 = 0.067 for 2 vs 4, so they can never fall under 0.05 here.
    Read t, log2fc and 'separated' as effect sizes.
    """
    L = dec.factors[mode].loc[list(g1) + list(g2)].to_numpy()
    n, n1 = L.shape[0], len(g1)
    obs = np.arange(n) < n1
    t_obs = _tstat(L, obs)

    t_all = []
    for idx in combinations(range(n), n1):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        t_all.append(_tstat(L, m))
    t_all = np.abs(np.array(t_all))             # labellings x factors (observed included)
    p_perm = (t_all >= np.abs(t_obs) - 1e-12).mean(0)
    p_maxT = (t_all.max(1)[:, None] >= np.abs(t_obs) - 1e-12).mean(0)

    A, B = L[obs], L[~obs]
    hi = np.where(A.mean(0) >= B.mean(0), names[0], names[1])
    separated = np.where(A.mean(0) >= B.mean(0), A.min(0) > B.max(0), B.min(0) > A.max(0))
    return pd.DataFrame({
        f'mean_{names[0]}': A.mean(0), f'mean_{names[1]}': B.mean(0),
        'log2fc': np.log2((A.mean(0) + 1e-12) / (B.mean(0) + 1e-12)),
        't': t_obs, 'higher_in': hi, 'separated': separated,
        'p_perm': p_perm, 'p_maxT': p_maxT, 'p_floor': 1 / len(t_all),
    }, index=dec.factors[mode].columns)


def top_loadings(dec, mode, factor, n=20):
    s = dec.factors[mode][factor].sort_values(ascending=False)
    return s.head(n)


# =========================================================================
# 7. Plots
# =========================================================================

def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#c3c2b7')
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e', labelsize=7)
    ax.grid(axis='x' if ax.get_xlabel() == 'loading' else 'y', color='#e8e7e2', lw=0.6)
    ax.set_axisbelow(True)


def plot_factors(dec, groups=None, top_n=12, max_full=15, figsize_per=(2.8, 2.1)):
    """One row per factor, one column per mode (the Tensor-cell2cell layout).

    groups : dict sample -> group name, used to colour the sample bars.
    Modes with more than max_full entries (LR pairs) show their top_n loadings.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    modes = list(dec.factors)
    R = dec.rank
    fig, axes = plt.subplots(R, len(modes), squeeze=False,
                             figsize=(figsize_per[0] * len(modes), figsize_per[1] * R))
    for j, mode in enumerate(modes):
        F = dec.factors[mode]
        for i, col in enumerate(F.columns):
            ax = axes[i, j]
            v = F[col]
            if len(v) > max_full:
                v = v.sort_values().tail(top_n)
                ax.barh(range(len(v)), v.values, color=NEUTRAL, height=0.7)
                ax.set_yticks(range(len(v)))
                ax.set_yticklabels(v.index, fontsize=6)
                ax.set_xlabel('loading', fontsize=7)
            else:
                colors = [GROUP_COLORS.get(groups.get(k), NEUTRAL) if groups else NEUTRAL
                          for k in v.index]
                ax.bar(range(len(v)), v.values, color=colors, width=0.7)
                ax.set_xticks(range(len(v)))
                ax.set_xticklabels(v.index, rotation=90, fontsize=7)
            _style(ax)
            if i == 0:
                ax.set_title(mode, fontsize=9, color='#0b0b0b')
            if j == 0:
                ax.set_ylabel(col, fontsize=9, color='#0b0b0b')
    if groups:
        handles = [Patch(color=GROUP_COLORS.get(g, NEUTRAL), label=g)
                   for g in dict.fromkeys(groups.values())]
        fig.tight_layout()
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 0.98),
                   fontsize=8, frameon=False)
    else:
        fig.tight_layout()
    return fig


def plot_rank_scan(scan):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    axes[0].errorbar(scan.index, scan['error'], yerr=scan['error_sd'], marker='o',
                     ms=5, lw=2, color='#2a78d6')
    axes[0].set_ylabel('relative error (observed entries)', fontsize=8)
    axes[1].plot(scan.index, scan['stability'], marker='o', ms=5, lw=2, color='#2a78d6')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('stability across runs', fontsize=8)
    for ax in axes:
        ax.set_xlabel('rank', fontsize=8)
        ax.set_xticks(scan.index)
        _style(ax)
    fig.tight_layout()
    return fig


def plot_sender_receiver(dec, factor):
    """Sender x receiver outer product of one factor's cell loadings (tensor B)."""
    import matplotlib.pyplot as plt
    s, r = dec.factors['Senders'][factor], dec.factors['Receivers'][factor]
    M = np.outer(s, r)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(M, cmap='Blues', vmin=0)
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels(r.index, rotation=90, fontsize=7)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=7)
    ax.set_xlabel('receiver', fontsize=8)
    ax.set_ylabel('sender', fontsize=8)
    ax.set_title(factor, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='sender x receiver loading')
    fig.tight_layout()
    return fig


# =========================================================================
# 8. Simulation (sanity check of the whole pipeline)
# =========================================================================

def simulate_tables(g1=('A1', 'A2'), g2=('A3', 'A4', 'A8', 'A9'), n_pairs=300,
                    n_planted=30, effect=4.0, scales=(1.0, 100.0, 0.01), noise=0.3, seed=0):
    """Fake per-method tables: n_planted pairs up in g1, n_planted up in g2.

    Pairs have log-normal baselines that differ a lot between pairs; each method
    sees the same biology on its own scale with independent multiplicative noise.
    Returns (method_tabs, truth) where truth maps pair -> 'g1' / 'g2' / None.
    """
    rng = np.random.default_rng(seed)
    samples = list(g1) + list(g2)
    pairs = [f'LR{i:03d}' for i in range(n_pairs)]
    base = rng.lognormal(0, 1.2, n_pairs)
    is_g1 = np.array([s in g1 for s in samples])
    fold = np.ones((n_pairs, len(samples)))
    fold[:n_planted, is_g1] = effect
    fold[n_planted:2 * n_planted, ~is_g1] = effect
    truth = pd.Series([None] * n_pairs, index=pairs, dtype=object)
    truth.iloc[:n_planted] = 'g1'
    truth.iloc[n_planted:2 * n_planted] = 'g2'
    tabs = {}
    for k, sc in enumerate(scales):
        X = sc * base[:, None] * fold * rng.lognormal(0, noise, fold.shape)
        tabs[f'method{k + 1}'] = pd.DataFrame(X, index=pairs, columns=samples)
    return tabs, truth
