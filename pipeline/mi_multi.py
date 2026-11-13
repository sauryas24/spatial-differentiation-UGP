"""Multi-condition structured ranking on the myocardial infarction data:
control vs remote zone (RZ) vs ischemic zone (IZ) vs fibrotic zone (FZ), all
different patients.
    python3 mi_multi.py [--rank 2] [--method cp]
Writes mi_multi_pairs.csv (score, q90, p_maxT, denoised effect per zone) and
mi_multi_components.csv (zone loadings of each CP component and its top pairs).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'benchmark'))
from samdiff import structured as st  # noqa: E402
from mi_structured import MIN_ABUND, L_SCALE, features, load  # noqa: E402
from real_structured import detected, kernel, lineage_slices, lr_table  # noqa: E402

GROUPS = {'control': ['control_P1', 'control_P7', 'control_P8', 'control_P17'],
          'RZ': ['RZ_P6', 'RZ_P9', 'RZ_P11'],
          'IZ': ['IZ_P3', 'IZ_P10', 'IZ_P15', 'IZ_P16'],
          'FZ': ['FZ_P14', 'FZ_P18', 'FZ_P20']}


def cp_components(F, sizes, rank, names, zones):
    """CP of the observed effect tensor: zone loadings and top pairs per component."""
    import tensorly as tl
    from tensorly.decomposition import parafac
    labs = st.multi_relabelings(sizes, 1)
    E = st._block_multi(F, labs[:1], len(sizes))[0]
    w, (A, B, C) = parafac(tl.tensor(E), rank, init='svd', n_iter_max=500, tol=1e-8)
    rows = []
    for r in range(rank):
        sgn = np.sign(C[:, r][np.abs(C[:, r]).argmax()])          # make the largest zone loading positive
        a = sgn * A[:, r] * w[r] * np.linalg.norm(B[:, r]) * np.linalg.norm(C[:, r])
        c = sgn * C[:, r] / np.linalg.norm(C[:, r])
        top_up = [names[i] for i in np.argsort(-a)[:8]]
        top_dn = [names[i] for i in np.argsort(a)[:8]]
        rows.append({'component': r + 1, **{f'zone_{z}': c[i] for i, z in enumerate(zones)},
                     'pairs_following_pattern': ', '.join(top_up),
                     'pairs_opposite_pattern': ', '.join(top_dn)})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=2)
    ap.add_argument('--method', default='svd')
    ap.add_argument('--n_perm', type=int, default=1000)
    a = ap.parse_args()
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 120)
    zones = list(GROUPS)
    samples = sum(GROUPS.values(), [])
    sizes = [len(g) for g in GROUPS.values()]
    pairs = lr_table()
    ann = {n: x for n, _, _, x in pairs}
    data = {s: load(s) for s in samples}
    names = sorted(set.intersection(*[detected(data[s][0], pairs) for s in samples]))
    sel = [p for p in pairs if p[0] in set(names)]
    types = data[samples[0]][4]
    slices = {s: lineage_slices(X, P, kernel(xy, L_SCALE * nn), sel) for s, (X, P, xy, nn, _) in data.items()}
    abund = pd.DataFrame({s: data[s][1].sum(0) for s in samples}, index=types).T
    keep = [i for i, t in enumerate(types) if abund[t].min() >= MIN_ABUND]
    F = features(slices, names, samples, keep, len(types))
    print(f'{len(names)} pairs, {len(keep)} cell types, groups {dict(zip(zones, sizes))}')
    q90 = st.multi_rank([F], names, sizes, zones, rank=a.rank, method=a.method, n_perm=a.n_perm)
    mxt = st.multi_rank([F], names, sizes, zones, rank=a.rank, method=a.method, n_perm=a.n_perm, qmode='maxT')
    res = q90.drop(columns='sign').rename(columns={'q': 'q90'})
    res.insert(2, 'p_maxT', mxt['q'])
    eff = res[[f'effect_{z}' for z in zones]]
    res['highest_in'] = eff.idxmax(axis=1).str.replace('effect_', '')
    res['lowest_in'] = eff.idxmin(axis=1).str.replace('effect_', '')
    res['annotation'] = [ann[n] for n in res.index]
    res = res.sort_values('score', ascending=False)
    res.index.name = 'pair'
    res.to_csv(RESULTS / 'mi_multi_pairs.csv')
    comp = cp_components(F, sizes, max(a.rank, 2), names, zones)
    comp.to_csv(RESULTS / 'mi_multi_components.csv', index=False)
    print(f'null relabellings: {q90.attrs["n_relabelings"]}')
    for thr in (0.05, 0.1):
        print(f'q90 < {thr}: {(res.q90 < thr).sum()}   ', end='')
    print(f'maxT p < 0.05: {(res.p_maxT < 0.05).sum()}')
    called = res[res.q90 < 0.1]
    print('\ncalled pairs by (highest zone, lowest zone):')
    print(called.groupby(['highest_in', 'lowest_in']).size().sort_values(ascending=False).to_string())
    print('\nCP components (zone loadings, unit norm):')
    print(comp.round(2).to_string(index=False))
    print('\ntop 30 pairs:')
    print(res.head(30).round(2).to_string())
