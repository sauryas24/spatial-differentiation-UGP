"""Structured differential ranking (v2) on the Kuppe et al. 2022 myocardial
infarction Visium data (compact copies from mi_fetch.py).

Contrasts: control vs ischemic zone (IZ), control vs fibrotic zone (FZ, with and
without the two GT samples). Same pipeline as real_structured.py; the kernel length
is 0.9 x the median nearest-spot distance (75 / 83 in the gut data).
    python3 mi_structured.py
Writes mi_structured_<contrast>.csv, mi_structured_program_<contrast>.csv and
mi_structured_robustness.csv.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'benchmark'))
from samdiff import structured as st  # noqa: E402
from real_structured import detected, gene_expr, kernel, lineage_slices, lr_table  # noqa: E402
from samdiff import sam, tensor as tc  # noqa: E402

DATA = HERE / 'data' / 'kuppe_mi'
CONTROL = ['control_P1', 'control_P7', 'control_P8', 'control_P17']
CONTRASTS = {
    'IZ': (CONTROL, ['IZ_P3', 'IZ_P10', 'IZ_P15', 'IZ_P16']),
    'FZ': (CONTROL, ['FZ_P14', 'FZ_P18', 'FZ_P20', 'FZ_GT_P4', 'FZ_GT_P19']),
    'FZ_noGT': (CONTROL, ['FZ_P14', 'FZ_P18', 'FZ_P20']),
}
MIN_FRAC, MIN_ABUND, L_SCALE = 0.05, 5.0, 0.9


def load(s):
    z = np.load(DATA / f'Visium_{s}.npz', allow_pickle=False)
    X = pd.DataFrame(z['X'] / np.maximum(z['lib'], 1)[:, None] * 1e4, columns=z['genes'])
    xy = z['xy']
    nn = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
    return X, z['P'], xy, nn, list(z['types'])


def features(slices, names, samples, keep, K):
    cols = [i * K + j for i in keep for j in keep]
    return st.commot_features(slices, samples, names, offset='trend')[:, :, cols]


def analyse(data, g1, g2, pairs, l_scale=L_SCALE, rank=1, names=None):
    samples = g1 + g2
    types = data[samples[0]][4]
    K = len(types)
    if names is None:
        names = sorted(set.intersection(*[detected(data[s][0], pairs) for s in samples]))
    sel = [p for p in pairs if p[0] in set(names)]
    slices = {s: lineage_slices(X, P, kernel(xy, l_scale * nn), sel)
              for s, (X, P, xy, nn, _) in ((s, data[s]) for s in samples)}
    abund = pd.DataFrame({s: data[s][1].sum(0) for s in samples}, index=types).T
    keep = [i for i, t in enumerate(types) if abund[t].min() >= MIN_ABUND]
    F = features(slices, names, samples, keep, K)
    n, n1 = len(samples), len(g1)
    q90 = st.structured_rank([F], names, n, n1, rank=rank, score='l2', qmode='q90')
    mxt = st.structured_rank([F], names, n, n1, rank=rank, score='l2', qmode='maxT')
    D = st._block_d(F, st.relabelings(n, n1))[0]
    U, sv, Vt = np.linalg.svd(D, full_matrices=False)
    sgn = np.sign(Vt[0][np.abs(Vt[0]).argmax()])
    kt = [types[i] for i in keep]
    prog = pd.DataFrame((sgn * Vt[0]).reshape(len(keep), len(keep)), index=kt, columns=kt)
    tot = pd.DataFrame({s: {p: slices[s][p].sum() for p in names} for s in samples})
    base = sam.sam(tc.transform_table(tot), g1, g2)
    res = pd.DataFrame({'score': q90['score'],
                        'higher_in': pd.Series(np.where(q90['sign'] > 0, 'control', 'disease'),
                                               index=q90.index),
                        'q90': q90['q'], 'p_maxT': mxt['q']})
    res['sam_total_d'] = base['d_sam'].reindex(res.index)
    res['sam_total_q'] = base['q'].reindex(res.index)
    return res.sort_values('score', ascending=False), prog, sv, abund, names


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 220)
    pairs = lr_table()
    ann = {n: a for n, _, _, a in pairs}
    allsamp = sorted({s for g1, g2 in CONTRASTS.values() for s in g1 + g2})
    data = {s: load(s) for s in allsamp}
    print('median UMI/spot:', {s: int(np.median(np.load(DATA / f'Visium_{s}.npz')['lib'])) for s in allsamp})
    rob = []
    for c, (g1, g2) in CONTRASTS.items():
        res, prog, sv, abund, names = analyse(data, g1, g2, pairs)
        res['annotation'] = [ann[n] for n in res.index]
        res.index.name = 'pair'
        res.to_csv(RESULTS / f'mi_structured_{c}.csv')
        prog.to_csv(RESULTS / f'mi_structured_program_{c}.csv')
        n_lab = len(st.relabelings(len(g1 + g2), len(g1)))
        print(f'\n===== control ({len(g1)}) vs {c} ({len(g2)}): {len(names)} pairs, {n_lab} labellings')
        print('cell-type abundance (spot-equivalents):\n', abund.round(0).astype(int).to_string())
        print('singular values:', np.round(sv[:5], 1))
        print('leading program (sender rows x receiver cols), positive = higher in control:\n', prog.round(2).to_string())
        for thr in (0.05, 0.1):
            k = res[res.q90 < thr]
            print(f'q90 < {thr}: {len(k)} pairs ({(k.higher_in == "control").sum()} control, {(k.higher_in == "disease").sum()} {c})')
        print(f'maxT p < 0.05: {(res.p_maxT < 0.05).sum()}   |  SAM on totals q < 0.1: {(res.sam_total_q < 0.1).sum()}')
        print('top 25:\n', res.head(25).round(3).to_string())
        if c == 'FZ_noGT':
            continue
        top = set(res.index[:50])
        ov = lambda r: len(top & set(r.index[:50]))
        rob.append({'contrast': c, 'setting': 'rank 2', 'top50_overlap': ov(analyse(data, g1, g2, pairs, rank=2, names=names)[0])})
        for ls in (0.5, 1.8):
            rob.append({'contrast': c, 'setting': f'kernel x{ls / L_SCALE:.1f}',
                        'top50_overlap': ov(analyse(data, g1, g2, pairs, l_scale=ls, names=names)[0])})
        for drop in g1 + g2:
            a, b = [s for s in g1 if s != drop], [s for s in g2 if s != drop]
            rob.append({'contrast': c, 'setting': f'drop {drop}',
                        'top50_overlap': ov(analyse(data, a, b, pairs, names=names)[0])})
    rob = pd.DataFrame(rob)
    rob.to_csv(RESULTS / 'mi_structured_robustness.csv', index=False)
    print('\nrobustness (top-50 overlap with main run):\n', rob.to_string(index=False))
