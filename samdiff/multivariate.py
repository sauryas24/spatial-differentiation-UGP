"""Multivariate SAM: Hotelling T^2 per LR pair across methods, with a Wishart
shrinkage prior on the covariance and an exact relabelling null.

Same procedure as the MULTIVARIATE SAM cell in final.ipynb (prints removed).
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import special, optimize


def build_multivariate_table(method_tabs: dict, cols):
    methods = list(method_tabs.keys())
    common_pairs = set(method_tabs[methods[0]].index)
    for m in methods[1:]:
        common_pairs &= set(method_tabs[m].index)
    common_pairs = sorted(common_pairs)

    multi = {}
    for pair in common_pairs:
        row = pd.DataFrame({m: method_tabs[m].loc[pair, cols] for m in methods})
        multi[pair] = row.dropna()        # complete cases only, per pair
    # drop pairs that lost too many samples to missingness
    multi = {k: v for k, v in multi.items() if len(v) == len(cols)}
    return multi, sorted(multi.keys())


# =========================================================================
# 2. PER-PAIR MEAN-DIFFERENCE VECTOR AND POOLED COVARIANCE
# =========================================================================

def pair_stats(df, g1, g2):
    A = df.loc[g1].to_numpy(float)     # (n1, p)
    B = df.loc[g2].to_numpy(float)     # (n2, p)
    n1, n2 = A.shape[0], B.shape[0]
    m1, m2 = A.mean(0), B.mean(0)
    diff = m1 - m2
    Ac, Bc = A - m1, B - m2
    scatter = Ac.T @ Ac + Bc.T @ Bc            # (p, p)
    d_resid = n1 + n2 - 2
    S = scatter / d_resid if d_resid > 0 else np.full(scatter.shape, np.nan)
    return n1, n2, diff, S, d_resid


def compute_all_pair_stats(multi, pairs, g1, g2):
    diffs, covs, dfs = {}, {}, {}
    for pair in pairs:
        _, _, diff, S, d_resid = pair_stats(multi[pair], g1, g2)
        diffs[pair], covs[pair], dfs[pair] = diff, S, d_resid
    return diffs, covs, dfs


# =========================================================================
# 3. ESTIMATE THE SHARED COVARIANCE PRIOR (S0, d0)
# =========================================================================
# Generalizes your moderated-t-test's fit_prior() to matrices, using the
# Bartlett decomposition of the Wishart log-determinant -- the matrix
# analogue of the log-chi-squared trick used for the scalar s0^2/d0.

def fit_matrix_prior(covs, dfs, p):
    pairs = [k for k in covs if dfs[k] > p]      # need d_i > p for det(S) to behave
    d_arr = np.array([dfs[k] for k in pairs], float)

    S_stack = np.stack([covs[k] for k in pairs])
    w = d_arr / d_arr.sum()
    S0 = np.tensordot(w, S_stack, axes=(0, 0))    # df-weighted average target matrix

    logdetS = np.array([np.log(max(np.linalg.det(covs[k]), 1e-300)) for k in pairs])

    def bartlett_mean(d):
        return sum(special.digamma((d - k + 1) / 2) - np.log(d / 2) for k in range(1, p + 1))

    def bartlett_var(d):
        return sum(special.polygamma(1, (d - k + 1) / 2) for k in range(1, p + 1))

    e = logdetS - np.array([bartlett_mean(d) for d in d_arr])
    evar = e.var(ddof=1) - np.mean([bartlett_var(d) for d in d_arr])

    if evar > 0:
        def f(d0):
            return sum(special.polygamma(1, (d0 - k + 1) / 2) for k in range(1, p + 1)) - evar
        try:
            d0 = optimize.brentq(f, p + 1e-3, 1e6)
        except ValueError:
            d0 = np.inf
    else:
        d0 = np.inf

    return S0, d0


# =========================================================================
# 4. SHRINKAGE COVARIANCE AND HOTELLING T^2
# =========================================================================

def shrink_cov(S_i, d_i, S0, d0, p, ridge=1e-6):
    S_tilde = (d0 * S0 + d_i * S_i) / (d0 + d_i) if np.isfinite(d0) else S0.copy()
    S_tilde += ridge * np.trace(S_tilde) / p * np.eye(p)   # numerical safety
    return S_tilde


def hotelling_t2(diff, S_tilde, n1, n2):
    S_inv = np.linalg.pinv(S_tilde)     # pseudo-inverse: safe near-singular
    return (n1 * n2 / (n1 + n2)) * diff @ S_inv @ diff


def directional_t(diff, S_tilde, n1, n2):
    """Signed test of one common shift shared by all methods: the generalised least
    squares estimate mu = 1'S^-1 diff / 1'S^-1 1 divided by its standard error. Noisy
    methods weigh less and opposite-signed differences cancel; t^2 is the part of T^2
    along the all-methods-agree direction."""
    a = np.linalg.pinv(S_tilde).sum(0)          # 1' S^-1
    return np.sqrt(n1 * n2 / (n1 + n2)) * (a @ diff) / np.sqrt(a.sum())


# =========================================================================
# 5. PERMUTATION NULL (same structure as your SAM code)
# =========================================================================

def all_relabelings(samples, n1):
    obs = set(samples[:n1])
    return [set(c) for c in combinations(samples, n1) if set(c) != obs]


def run_multivariate_sam(multi, pairs, g1, g2, p):
    samples = g1 + g2
    n1 = len(g1)

    # fit the shrinkage prior ONCE, from the real labels
    diffs0, covs0, dfs0 = compute_all_pair_stats(multi, pairs, g1, g2)
    S0, d0 = fit_matrix_prior(covs0, dfs0, p)

    # observed T^2 for every pair, using the real labels
    T2_obs, t_obs, diff_obs = {}, {}, {}
    for pair in pairs:
        n1o, n2o, diff, S_i, d_i = pair_stats(multi[pair], g1, g2)
        S_tilde = shrink_cov(S_i, d_i, S0, d0, p)
        T2_obs[pair] = hotelling_t2(diff, S_tilde, n1o, n2o)
        t_obs[pair] = directional_t(diff, S_tilde, n1o, n2o)
        diff_obs[pair] = diff

    # every other relabeling, same fixed (S0, d0)
    masks = all_relabelings(samples, n1)
    T2_perm = np.empty((len(masks), len(pairs)))
    t_perm = np.empty((len(masks), len(pairs)))
    for k, g1set in enumerate(masks):
        g1p = [s for s in samples if s in g1set]
        g2p = [s for s in samples if s not in g1set]
        for j, pair in enumerate(pairs):
            n1p, n2p, diffp, Sp, dp = pair_stats(multi[pair], g1p, g2p)
            Sp_tilde = shrink_cov(Sp, dp, S0, d0, p)
            T2_perm[k, j] = hotelling_t2(diffp, Sp_tilde, n1p, n2p)
            t_perm[k, j] = directional_t(diffp, Sp_tilde, n1p, n2p)

    # pi0, then pooled FDR (T^2 already >= 0, no abs() needed)
    null_flat = T2_perm.ravel()
    obs_arr = np.array([T2_obs[pair] for pair in pairs])
    # +1 smoothing: with few pairs that all sit above the null median the plain
    # estimate is 0, which forces every q-value to 0
    below = np.sum(obs_arr < np.percentile(null_flat, 50))
    pi0 = min((below + 1) / (0.5 * (len(obs_arr) + 1)), 1.0)

    null_sorted = np.sort(null_flat)
    n_perm = T2_perm.shape[0]
    fp = (len(null_sorted) - np.searchsorted(null_sorted, obs_arr, side='left') + 1) / n_perm
    called = len(obs_arr) - np.searchsorted(np.sort(obs_arr), obs_arr, side='left')
    fdr = np.minimum(pi0 * fp / called, 1.0)

    order = np.argsort(obs_arr)
    q = np.empty_like(fdr)
    q[order] = np.minimum.accumulate(fdr[order])

    res = pd.DataFrame({'T2': obs_arr, 'q': q, 't_dir': [t_obs[pair] for pair in pairs]}, index=pairs)
    method_names = multi[pairs[0]].columns.tolist()
    for i, m in enumerate(method_names):
        res[f'diff_{m}'] = [diff_obs[pair][i] for pair in pairs]

    res = res.sort_values('T2', ascending=False)
    res.attrs['null'] = pd.DataFrame(T2_perm, columns=pairs)[res.index]   # relabellings x pairs
    res.attrs['null_t_dir'] = pd.DataFrame(t_perm, columns=pairs)[res.index]
    return res, S0, d0
