"""Fast sender x receiver LR scores computed straight from spot counts.

For each LR pair, the ligand in spot i is credited to cell type t in proportion to
    P[i, t]                    ('abundance': what P.T @ W @ P implicitly does), or
    P[i, t] * mu[t, ligand]    ('expression': types that actually make the ligand),
and likewise for the receptor. The score is then
    S[s, r] = sum_ij K(i, j) L_s[i] R_r[j] / n_spots
with a Gaussian kernel K, as in SpatialDM's weight matrix.
mu comes from the reference the cohort was simulated from, so the 'expression'
weighting has an advantage here that a real analysis (independent reference) lacks.
"""
import pickle

import numpy as np
from scipy.spatial.distance import cdist

import simulate as sim

REF = None


def _ref():
    global REF
    if REF is None:
        REF = pickle.load(open(sim.__file__.replace('simulate.py', 'ref.pkl'), 'rb'))
    return REF


def kernel(xy, l=0.9 * sim.SPACING, cutoff=0.2):
    K = np.exp(-cdist(xy, xy) ** 2 / (2 * l ** 2))
    K[K < cutoff] = 0
    return K


def type_shares(P, mu_g, weighting):
    w = P * mu_g[None, :] if weighting == 'expression' else P.copy()
    tot = w.sum(1, keepdims=True)
    return np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)


def ct_slices(adata, pairs, weighting='expression', K=None, mu=None):
    """dict pair -> (types x types) score for one sample. pairs: [(name, l, r)].
    mu (types x reference genes) overrides the reference profile used for weighting."""
    ref = _ref()
    mu = ref.mu if mu is None else mu
    gi = {g: i for i, g in enumerate(ref.genes)}
    X = np.asarray(adata.X, float)
    X = X / X.sum(1, keepdims=True) * 1e4
    vi = {g: i for i, g in enumerate(adata.var_names)}
    P = adata.obs[sim.TYPES].to_numpy(float)
    K = kernel(adata.obsm['spatial']) if K is None else K
    out = {}
    for name, l, r in pairs:
        Ls = X[:, vi[l], None] * type_shares(P, mu[:, gi[l]], weighting)
        Rr = X[:, vi[r], None] * type_shares(P, mu[:, gi[r]], weighting)
        out[name] = Ls.T @ (K @ Rr) / len(X)
    return out
