"""Univariate SAM on an (LR pair x sample) table.

Same procedure as the SAM cell in final.ipynb, wrapped as a function so it can be
re-run on transformed tables (e.g. the log scale used by the tensor module).
"""
from itertools import combinations

import numpy as np
import pandas as pd


def diff_and_se(X, mask):
    A, B = X[:, mask], X[:, ~mask]
    n1, n2 = A.shape[1], B.shape[1]
    m1, m2 = A.mean(1), B.mean(1)
    sse = ((A - m1[:, None]) ** 2).sum(1) + ((B - m2[:, None]) ** 2).sum(1)
    se = np.sqrt((1 / n1 + 1 / n2) * sse / (n1 + n2 - 2))
    return m1 - m2, np.maximum(se, 1e-12)


def choose_s0(diff, se, n_bins=None):
    n = len(se)
    n_bins = n_bins or int(np.clip(n // 20, 10, 100))
    cands = np.unique(np.concatenate([[0.0], np.percentile(se, np.arange(5, 101, 5))]))
    edges = np.quantile(se, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, se, side='right') - 1, 0, n_bins - 1)
    cvs = []
    for s0 in cands:
        d = diff / (se + s0)
        mads = []
        for b in range(n_bins):
            db = d[bins == b]
            if len(db) > 2:
                mads.append(1.4826 * np.median(np.abs(db - np.median(db))))
        mads = np.array(mads)
        cvs.append(mads.std() / mads.mean() if len(mads) > 1 and mads.mean() > 0 else np.inf)
    return cands[np.argmin(cvs)]


def sam(tab, g1, g2, s0=None):
    """Two-group SAM with exact enumeration of relabellings.

    tab : DataFrame, rows = LR pairs, columns include g1 + g2. Rows with any
          missing value in those columns are dropped (SAM needs complete cases).
    Returns a DataFrame sorted by |d_sam| with mean_g1, mean_g2, diff, se, d_sam, q,
    and the run's s0, pi0 and n_perm in .attrs.
    """
    cols = list(g1) + list(g2)
    tab_c = tab[cols].dropna()
    X = tab_c.to_numpy(float)
    n_pairs, n_samp = X.shape
    obs_mask = np.array([True] * len(g1) + [False] * len(g2))

    diff_obs, se_obs = diff_and_se(X, obs_mask)
    if s0 is None:
        s0 = choose_s0(diff_obs, se_obs)
    d_obs = diff_obs / (se_obs + s0)

    perm_masks = []
    for idx in combinations(range(n_samp), obs_mask.sum()):
        m = np.zeros(n_samp, bool)
        m[list(idx)] = True
        if not np.array_equal(m, obs_mask):
            perm_masks.append(m)
    n_perm = len(perm_masks)

    d_perm = np.empty((n_perm, n_pairs))
    for k, m in enumerate(perm_masks):
        dp, sp = diff_and_se(X, m)
        d_perm[k] = dp / (sp + s0)

    q25, q75 = np.percentile(d_perm, [25, 75])
    pi0 = min(np.mean((d_obs > q25) & (d_obs < q75)) / 0.5, 1.0)

    a_obs = np.abs(d_obs)
    a_null = np.sort(np.abs(d_perm).ravel())
    fp = (len(a_null) - np.searchsorted(a_null, a_obs, side='left') + 1) / n_perm
    called = len(a_obs) - np.searchsorted(np.sort(a_obs), a_obs, side='left')
    fdr = np.minimum(pi0 * fp / called, 1.0)
    order = np.argsort(a_obs)
    q = np.empty_like(fdr)
    q[order] = np.minimum.accumulate(fdr[order])

    res = pd.DataFrame({
        'mean_g1': X[:, obs_mask].mean(1),
        'mean_g2': X[:, ~obs_mask].mean(1),
        'diff': diff_obs,
        'se': se_obs,
        'd_sam': d_obs,
        'q': q,
    }, index=tab_c.index).iloc[np.argsort(-a_obs)]
    res.attrs.update(s0=float(s0), pi0=float(pi0), n_perm=n_perm,
                     n_dropped=len(tab) - n_pairs)
    return res
