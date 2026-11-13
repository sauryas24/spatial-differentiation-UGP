"""Summary of T2 SAM for the report (Table 1 and the top calls of Table 2): the fitted
covariance prior (S0, d0 and the log-determinant moments behind d0), the share of
Hotelling's T^2 explained by disagreement between tools, the number of pairs with a
negative tool weight, and the calls of T^2 and of the directional t, on
  gut   : adult vs fetal (inputs prepared as in t2_gut.py)
  heart : control vs ischemic zone (per-section tool scores from mi_tool_scores.py)
    python3 pipeline/t2_summary.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS = HERE / 'results'
sys.path[:0] = [str(HERE), str(HERE / 'benchmark')]
from samdiff import multivariate as mv  # noqa: E402
from samdiff import structured as st  # noqa: E402

CONTROL = ['control_P1', 'control_P7', 'control_P8', 'control_P17']
ISCHEMIC = ['IZ_P3', 'IZ_P10', 'IZ_P15', 'IZ_P16']


def summarise(tabs, g1, g2, label):
    z = {m: (t - np.nanmean(t.values)) / np.nanstd(t.values) for m, t in tabs.items()}
    multi, pairs = mv.build_multivariate_table({m: t[g1 + g2] for m, t in z.items()}, g1 + g2)
    d, c, f = mv.compute_all_pair_stats(multi, pairs, g1, g2)
    S0, d0 = mv.fit_matrix_prior(c, f, 3)
    dd = np.array([f[k] for k in pairs], float)
    logdet = np.array([np.log(np.linalg.det(c[k])) for k in pairs])
    bmean = lambda x: sum(special.digamma((x - k + 1) / 2) - np.log(x / 2) for k in range(1, 4))
    bvar = lambda x: sum(special.polygamma(1, (x - k + 1) / 2) for k in range(1, 4))
    e = logdet - np.array([bmean(x) for x in dd])
    print(f'\n===== {label}: {len(pairs)} pairs, d_i = {sorted(set(f.values()))}')
    print('S0:\n', np.round(S0, 3))
    print(f'Var(e) = {e.var(ddof=1):.2f}, expected from sampling = {np.mean([bvar(x) for x in dd]):.2f}, '
          f'd0 = {d0:.2f}, weight on S0 = {d0 / (d0 + dd[0]):.2f}')
    n1, n2 = len(g1), len(g2)
    share, neg = [], 0
    for k in pairs:
        S = mv.shrink_cov(c[k], f[k], S0, d0, 3)
        a = np.linalg.pinv(S).sum(0)
        neg += bool(((a / a.sum()) < 0).any())
        share.append(1 - mv.directional_t(d[k], S, n1, n2) ** 2 / mv.hotelling_t2(d[k], S, n1, n2))
    print(f'median disagreement share of T^2 = {np.median(share):.2f}; pairs with a negative tool weight = {neg}')
    res, _, _ = mv.run_multivariate_sam(multi, pairs, g1, g2, 3)
    q = st._qvalues(res.t_dir.abs().to_numpy(), res.attrs['null_t_dir'].abs().to_numpy(), 'q90')
    qT = st._qvalues(res.T2.to_numpy(), res.attrs['null'].to_numpy(), 'q90')
    c_ = q < 0.1
    print(f'relabellings {res.attrs["null"].shape[0]}; T^2 calls {(qT < 0.1).sum()}; '
          f't calls {c_.sum()} ({(c_ & (res.t_dir > 0)).sum()} higher in first group, '
          f'{(c_ & (res.t_dir < 0)).sum()} in second)')
    res['q90_dir'] = q
    top = res.reindex(res.t_dir.abs().sort_values(ascending=False).index).head(10)
    print(top[['t_dir', 'q90_dir'] + [x for x in res if x.startswith('diff_')]].round(3).to_string())


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    import t2_gut
    from real_structured import G1, G2
    summarise(t2_gut.fixed_tables(), G1, G2, 'gut: adult vs fetal')

    import mi_structured as mis
    names = json.load(open(RESULTS / 'tc2c' / 'mi_meta.json'))['pairs']
    tabs = {m: pd.read_csv(RESULTS / f'mi_tool_results_{m}.csv', index_col=0).reindex(names)
            for m in ('commot', 'spatialdm', 'stlearn')}
    samples = CONTROL + ISCHEMIC
    n_spots = pd.Series({s: len(mis.load(s)[2]) for s in samples})
    commot = np.log1p(tabs['commot'][samples] / n_spots)
    stl = tabs['stlearn'][samples]
    stl = np.log1p(stl / stl.values[stl.values > 0].mean())
    prep = {'spatialdm': tabs['spatialdm'][samples], 'commot': commot, 'stlearn': stl}
    summarise({m: t - t.median() for m, t in prep.items()}, CONTROL, ISCHEMIC, 'heart: control vs ischemic')
