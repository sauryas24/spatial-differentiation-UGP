"""Structured differential ranking: the Tensor-cell2cell idea used as a supervised
ranking instead of an unsupervised decomposition.

Per sample we have COMMOT signal for every LR pair between every sender and
receiver type, S[n, l, s, r], plus per-pair scores from the other tools.

1. Offset removal. Composition and depth move every LR pair of a cell-type pair
   together, so log S[n, l, s, r] - median_l log S[n, l, s, r] keeps only what is
   specific to an LR pair.
2. Effect matrix. SAM d-statistics per (pair, sender, receiver) and per (pair, tool)
   form a pairs x columns matrix D. A real change is a few programs (sets of pairs
   sharing a sender/receiver pattern and agreeing tools), so D is close to low rank;
   a rank-k SVD of D shrinks isolated noise and borrows strength across pairs.
3. Calibration. Steps 1-2 are rerun for every relabelling of the samples, and the
   pooled null of pair scores gives SAM-style q-values.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from samdiff import sam, tensor as tc


def relabelings(n, n1):
    """Boolean group-1 masks; the observed labelling (first n1 samples) comes first."""
    obs = np.arange(n) < n1
    out = [obs]
    for idx in combinations(range(n), n1):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        if not np.array_equal(m, obs):
            out.append(m)
    return out


def commot_features(slices, samples, pairs, offset=True, pseudo=0.05, column_threshold=False,
                    abundance=None):
    """samples x pairs x (K*K) log signal.

    offset           : False, True (per (sample, sender, receiver) median-of-ratios
                       offset) or 'trend' (intensity trend per sample first, then the
                       offset).
    column_threshold : pairs used for a column's offset are those above the median
                       positive signal *of that column* (True) or of the whole tensor
                       (False, the original rule, which skips columns of rare types).
    abundance        : samples x types fractions; if given, signal is divided by
                       sender x receiver abundance before the log (communication per
                       cell pair) instead of estimating the offset from the pairs.
    """
    S = np.stack([[slices[s][p] for p in pairs] for s in samples])
    n, P, K = len(samples), len(pairs), S.shape[-1]
    if abundance is not None:
        a = np.asarray(abundance, float)
        S = S / np.maximum(a[:, None, :, None] * a[:, None, None, :], 1e-6)
    S = S.reshape(n, P, -1)
    med = np.median(S[S > 0])
    L = np.log(S + pseudo * med)
    if offset == 'trend':
        L = L - _intensity_trend(L)
    if offset and abundance is None:
        m = S.mean(0)
        if column_threshold:
            col_med = np.array([np.median(c[c > 0]) if (c > 0).any() else np.inf for c in m.T])
            expressed = m > col_med[None]
        else:
            expressed = m > med
        dev = np.where(expressed[None], L - L.mean(0, keepdims=True), np.nan)
        with np.errstate(all='ignore'):
            alpha = np.nanmedian(dev, axis=1, keepdims=True)
        L = L - np.nan_to_num(alpha)
    return L


def _intensity_trend(L, n_bins=20):
    """Per-sample deviation from the cross-sample mean as a function of the mean
    (binned median, interpolated): the depth effect, which is stronger for weak
    signals that sit near the detection floor."""
    m = L.mean(0).ravel()
    edges = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    centres = (edges[:-1] + edges[1:]) / 2
    bins = np.clip(np.searchsorted(edges, m, side='right') - 1, 0, n_bins - 1)
    out = np.empty_like(L)
    for i in range(len(L)):
        dev = (L[i] - L.mean(0)).ravel()
        meds = np.array([np.median(dev[bins == b]) if (bins == b).any() else 0.0
                         for b in range(n_bins)])
        out[i] = np.interp(m, centres, meds).reshape(L.shape[1:])
    return out


def tool_features(tab, samples, pairs):
    """samples x pairs x 1 from an (LR pair x sample) table; missing rows are NaN."""
    return tab.reindex(pairs)[samples].to_numpy(float).T[:, :, None]


def _block_d(F, masks):
    """d-statistics (relabellings x pairs x columns) for one feature block. s0 is
    chosen on the observed labels, as in SAM. Columns with missing values give 0."""
    n, P, C = F.shape
    X = F.reshape(n, P * C).T
    ok = ~np.isnan(X).any(1)
    out = np.zeros((len(masks), P * C))
    diff, se = sam.diff_and_se(X[ok], masks[0])
    s0 = sam.choose_s0(diff, se)
    for k, m in enumerate(masks):
        diff, se = sam.diff_and_se(X[ok], m)
        out[k, ok] = diff / (se + s0)
    return out.reshape(len(masks), P, C)


def _center_columns(D, F):
    """Subtract each column's median effect over the pairs that carry signal there
    (above the column's median mean feature); applied to every relabelling."""
    m = F.mean(0)
    use = m > np.median(m, axis=0, keepdims=True)
    out = D.copy()
    for c in range(D.shape[2]):
        if use[:, c].sum() >= 5:
            out[:, :, c] -= np.median(D[:, use[:, c], c], axis=1, keepdims=True)
    return out


def _scores(D, rank, score='max'):
    """Pair score from the (optionally rank-truncated) effect row: its largest
    |entry| ('max') or its norm ('l2'); the sign is that of the largest entry."""
    if rank:
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        D = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    j = np.abs(D).argmax(1)
    v = D[np.arange(len(D)), j]
    val = np.linalg.norm(D, axis=1) if score == 'l2' else np.abs(v)
    return val, np.sign(v)


def _adaptive_rank(D, null_idx, max_rank=5):
    """Number of leading components whose singular value beats every relabelling
    (sequentially): 0 means no program is distinguishable from relabelled data."""
    sv = np.array([np.linalg.svd(D[k], compute_uv=False)[:max_rank] for k in range(len(D))])
    k = 0
    while k < max_rank and sv[0, k] > sv[null_idx, k].max() * (1 + 1e-9):
        k += 1
    return k


def _qvalues(obs, null, mode='mean'):
    """q-values for non-negative scores against relabelled null scores
    (relabellings x pairs).

    'mean' : SAM: expected false calls at a cut-off = pi0 x the average number of
             null scores above it per relabelling.
    'q90'  : as 'mean' but with the 90th percentile over relabellings of that count,
             so a cut-off is only accepted if it also beats the more extreme
             relabellings. Guards against the labelling lining up with sample-level
             structure shared by all pairs.
    'maxT' : Westfall-Young single-step adjusted p-values: the fraction of
             relabellings (counting the observed one) whose largest score over all
             pairs reaches the pair's score. Controls the family-wise error rate.
    """
    n_perm = len(null)
    if mode == 'maxT':
        mx = null.max(1)
        return (1 + (mx[None, :] >= obs[:, None]).sum(1)) / (n_perm + 1)
    pi0 = min(np.mean(obs < np.median(null)) / 0.5, 1.0)
    if mode == 'q90':
        srt = np.sort(null, axis=1)
        counts = null.shape[1] - np.stack([np.searchsorted(r, obs, side='left') for r in srt])
        fp = np.quantile(counts, 0.9, axis=0) + 1 / n_perm
    else:
        a_null = np.sort(null.ravel())
        fp = (len(a_null) - np.searchsorted(a_null, obs, side='left') + 1) / n_perm
    called = len(obs) - np.searchsorted(np.sort(obs), obs, side='left')
    fdr = np.minimum(pi0 * fp / called, 1.0)
    order = np.argsort(obs)
    q = np.empty_like(fdr)
    q[order] = np.minimum.accumulate(fdr[order])
    return q


def structured_rank(blocks, pairs, n, n1, rank=None, weights=None, score='max',
                    center=False, adaptive=False, gate=False, drop_complement=True,
                    qmode='mean'):
    """blocks : list of samples x pairs x columns feature arrays (one per source).

    rank     : fixed SVD rank of the effect matrix (None = no truncation).
    center   : subtract each column's median effect (per relabelling).
    adaptive : choose the rank as the number of components beating every
               relabelling (falls back to rank 1 for the ranking if none does).
    gate     : with adaptive, make no calls (q = 1) when no component survives.
    drop_complement : leave out the relabelling that only swaps the two groups,
               which for equal group sizes reproduces the observed statistic exactly
               (False reproduces the original version).
    qmode    : 'mean' (SAM), 'q90' or 'maxT'; see _qvalues.
    Returns DataFrame(index=pairs, columns=[score, sign, q]) with attrs['rank'].
    """
    masks = relabelings(n, n1)
    null_idx = [k for k in range(1, len(masks))
                if not (drop_complement and np.array_equal(masks[k], ~masks[0]))]
    Ds = [_block_d(F, masks) for F in blocks]
    if center:
        Ds = [_center_columns(D, F) for D, F in zip(Ds, blocks)]
    if weights is not None:
        Ds = [D * w for D, w in zip(Ds, weights)]
    D = np.concatenate(Ds, axis=2)
    k_sig = None
    if adaptive:
        k_sig = _adaptive_rank(D, null_idx)
        rank = max(k_sig, 1)
    res = [_scores(D[k], rank, score) for k in [0] + null_idx]
    score_obs, sign = res[0]
    null = np.stack([r[0] for r in res[1:]])
    q = _qvalues(score_obs, null, qmode)
    if gate and k_sig == 0:
        q = np.ones_like(q)
    out = pd.DataFrame({'score': score_obs, 'sign': sign, 'q': q}, index=pairs)
    out.attrs['rank'] = rank if k_sig is None else k_sig
    return out


def variants(tabs, ct_sources, g1, g2):
    """Every configuration compared in the benchmark. ct_sources: name -> slices
    (dict sample -> dict pair -> types x types)."""
    samples = g1 + g2
    n, n1 = len(samples), len(g1)
    out = {}
    for src, slices in ct_sources.items():
        pairs = sorted(set.intersection(*[set(slices[s]) for s in samples]))
        tools = [tool_features(tabs['spatialdm'], samples, pairs),
                 tool_features(tc.transform_table(tabs['stlearn'], sample_norm=True), samples, pairs)]
        for norm, offset in (('', False), (' +off', True), (' +trend', 'trend')):
            F = commot_features(slices, samples, pairs, offset=offset)
            out[f'{src}{norm}'] = structured_rank([F], pairs, n, n1)
            for k in (1, 2):
                out[f'{src}{norm} +LR{k}'] = structured_rank([F], pairs, n, n1, rank=k)
            if offset == 'trend':
                out[f'{src}{norm} +tools'] = structured_rank([F] + tools, pairs, n, n1)
                out[f'{src}{norm} +tools +LR1'] = structured_rank([F] + tools, pairs, n, n1, rank=1)
    return out


# =========================================================================
# More than two conditions
# =========================================================================
#
# Each (pair, column) entry gets one standardised effect per condition,
#     e_g = sqrt(n_g) (mean_g - grand mean) / (s_within + s0),
# so sum_g e_g^2 is the between-group sum of squares over the within-group
# variance (multiclass SAM's statistic, squared). The effects form a
# pairs x columns x conditions tensor; its low-rank part (rank-k SVD of the
# pairs x (columns * conditions) unfolding, or a CP decomposition) is the shared
# structure, and a pair's score is the norm of its denoised slice. Samples are
# relabelled at random across all groups for the null.

def multi_relabelings(sizes, n_perm=500, seed=0):
    """Group-label vectors: the observed one first, then up to n_perm random ones
    giving distinct partitions of the samples (a labelling that only renames groups
    of equal size is the same partition and is skipped)."""
    obs = np.repeat(np.arange(len(sizes)), sizes)
    key = lambda lab: frozenset(frozenset(np.flatnonzero(lab == g)) for g in range(len(sizes)))
    seen, out = {key(obs)}, [obs]
    rng = np.random.default_rng(seed)
    from collections import Counter
    from math import factorial
    total = factorial(len(obs))
    for k in sizes:
        total //= factorial(k)
    for mult in Counter(sizes).values():
        total //= factorial(mult)          # renaming equal-sized groups gives the same partition
    if total - 1 <= n_perm:                 # small designs: enumerate every partition
        for perm in __import__('itertools').permutations(range(len(obs))):
            lab = obs[list(perm)]
            k = key(lab)
            if k not in seen:
                seen.add(k)
                out.append(lab)
        return out
    while len(out) - 1 < n_perm:
        lab = rng.permutation(obs)
        k = key(lab)
        if k not in seen:
            seen.add(k)
            out.append(lab)
    return out


def _multi_effects(X, lab, G):
    """X: entries x samples -> (effects entries x G, between norm, within sd)."""
    m = X.mean(1, keepdims=True)
    within, eff = np.zeros(len(X)), []
    for g in range(G):
        Xg = X[:, lab == g]
        mg = Xg.mean(1, keepdims=True)
        within += ((Xg - mg) ** 2).sum(1)
        eff.append(np.sqrt(Xg.shape[1]) * (mg - m)[:, 0])
    s = np.sqrt(within / (X.shape[1] - G))
    return np.stack(eff, 1), np.maximum(s, 1e-12)


def _block_multi(F, labs, G):
    """Effects (relabellings x pairs x columns x G) for one feature block."""
    n, P, C = F.shape
    X = F.reshape(n, P * C).T
    ok = ~np.isnan(X).any(1)
    e, s = _multi_effects(X[ok], labs[0], G)
    s0 = sam.choose_s0(np.linalg.norm(e, axis=1), s)
    out = np.zeros((len(labs), P * C, G))
    for k, lab in enumerate(labs):
        e, s = _multi_effects(X[ok], lab, G)
        out[k, ok] = e / (s + s0)[:, None]
    return out.reshape(len(labs), P, C, G)


def _denoise_multi(E, rank, method):
    P, C, G = E.shape
    if not rank:
        return E
    if method == 'cp':
        import tensorly as tl
        from tensorly.decomposition import parafac
        cp = parafac(tl.tensor(E), rank, init='svd', n_iter_max=200, tol=1e-7)
        return tl.cp_to_tensor(cp)
    U, s, Vt = np.linalg.svd(E.reshape(P, C * G), full_matrices=False)
    return ((U[:, :rank] * s[:rank]) @ Vt[:rank]).reshape(P, C, G)


def multi_rank(blocks, pairs, sizes, groups=None, rank=1, method='svd', qmode='q90',
               n_perm=500, seed=0):
    """Structured ranking for G >= 2 conditions.

    blocks : samples x pairs x columns feature arrays, samples ordered by group.
    sizes  : samples per group, in that order; groups names them.
    Returns DataFrame(index=pairs) with score, q and the denoised condition profile
    of each pair (its effect per condition in the column where it is largest).
    """
    G = len(sizes)
    groups = groups or [f'g{g}' for g in range(G)]
    labs = multi_relabelings(sizes, n_perm, seed)
    E = np.concatenate([_block_multi(F, labs, G) for F in blocks], axis=2)
    scores, prof = [], None
    for k in range(len(labs)):
        Eh = _denoise_multi(E[k], rank, method)
        nrm = np.linalg.norm(Eh, axis=2)
        scores.append(np.linalg.norm(nrm, axis=1))
        if k == 0:
            c = nrm.argmax(1)
            prof = Eh[np.arange(len(pairs)), c]
    scores = np.stack(scores)
    out = pd.DataFrame(prof, index=pairs, columns=[f'effect_{g}' for g in groups])
    out.insert(0, 'score', scores[0])
    out.insert(1, 'q', _qvalues(scores[0], scores[1:], qmode))
    out['sign'] = 1.0
    out.attrs['n_relabelings'] = len(labs) - 1
    return out


def multi_sam(tab, sizes, n_perm=500, seed=0, qmode='mean'):
    """Multiclass SAM on an (LR pair x sample) table, samples ordered by group."""
    X = tab.dropna().to_numpy(float)
    labs = multi_relabelings(sizes, n_perm, seed)
    G = len(sizes)
    e, s = _multi_effects(X, labs[0], G)
    s0 = sam.choose_s0(np.linalg.norm(e, axis=1), s)
    sc = []
    for lab in labs:
        e, s = _multi_effects(X, lab, G)
        sc.append(np.linalg.norm(e, axis=1) / (s + s0))
    sc = np.stack(sc)
    return pd.DataFrame({'score': sc[0], 'q': _qvalues(sc[0], sc[1:], qmode), 'sign': 1.0},
                        index=tab.dropna().index)
