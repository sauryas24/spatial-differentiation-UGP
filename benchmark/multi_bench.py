"""Three-condition benchmark of the structured ranking.
A = control, B = program A on, C = programs A and B on. A pair counts as
differential if it is planted in A or B (any difference among conditions).
    python3 multi_bench.py --reps 50 --workers 8      -> metrics_multi.csv
Methods
  multiclass SAM        on the kernel score summed over cell pairs
  pairwise v2           two-group structured v2 (l2, q90) for A-B, A-C, B-C;
                        q = 3 x the smallest q, score = largest score
  multi svd / cp        multi-condition structured ranking, rank 1 or 2
Also reports how well each planted pair's denoised condition profile matches its
true pattern (program A: up in B and C; program B: up in C only).
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import argparse
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import ct_scores as cs
import evaluate as ev
import run_benchmark as rb
import run_tools as rt
import simulate as sim
from samdiff import structured as st
from samdiff import tensor as tc

SCEN = ['multi_null', 'multi_fold4', 'multi_fold2', 'multi_comp', 'multi_4each']
TEMPLATE = {'A': np.array([-2., 1, 1]), 'B': np.array([-1., -1, 2])}


def cohort(scenario, rep):
    ref = pickle.load(open(ev.HERE / 'ref.pkl', 'rb'))
    lr = sim.single_chain_secreted(*rb.LR_ARGS)
    layout = sim.load_layout(rb.DATA / 'coord.csv', rb.DATA / 'interest_region_Spot_IDs.csv')
    adatas, truth, groups = sim.simulate_cohort(ref, layout, lr, sim.SCENARIOS[scenario],
                                                70000 + 1000 * SCEN.index(scenario) + rep)
    db = rt.commot_db(list(adatas.values()), lr)
    pairs = [(f'{a}_{b}', a, b) for a, b in zip(db.iloc[:, 0], db.iloc[:, 1])]
    return adatas, truth, groups, pairs, cs.kernel(layout[0].to_numpy())


def pattern_corr(res, truth):
    cols = [c for c in res if c.startswith('effect_')]
    out = []
    for prog, tmpl in TEMPLATE.items():
        for p in truth.loc[truth.program == prog, 'pair']:
            if p in res.index:
                out.append(np.corrcoef(res.loc[p, cols].to_numpy(float), tmpl)[0, 1])
    return np.nanmean(out) if out else np.nan


def one(job):
    warnings.filterwarnings('ignore')
    scenario, rep = job
    adatas, truth, groups, pairs, K = cohort(scenario, rep)
    samples = sum(groups, [])
    sizes = [len(g) for g in groups]
    sl = {s: cs.ct_slices(adatas[s], pairs, 'abundance', K) for s in samples}
    names = sorted(sl[samples[0]])
    F = st.commot_features(sl, samples, names, offset='trend')
    tot = pd.DataFrame({s: {p: sl[s][p].sum() for p in names} for s in samples})
    res = {'multiclass SAM': st.multi_sam(tc.transform_table(tot)[samples], sizes)}
    pw = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        idx = [samples.index(s) for s in groups[a] + groups[b]]
        pw.append(st.structured_rank([F[idx]], names, len(idx), sizes[a], rank=1, score='l2', qmode='q90'))
    res['pairwise v2'] = pd.DataFrame({'score': np.max([r.score for r in pw], 0),
                                       'q': np.minimum(3 * np.min([r.q for r in pw], 0), 1),
                                       'sign': 1.0}, index=names)
    for method, rank in (('svd', 1), ('svd', 2), ('svd', 3), ('cp', 2)):
        res[f'multi {method} r{rank}'] = st.multi_rank([F], names, sizes, list('ABC'),
                                                       rank=rank, method=method)
    res['multi svd r2 maxT'] = st.multi_rank([F], names, sizes, list('ABC'), rank=2, qmode='maxT')
    rows = []
    for m, r in res.items():
        extra = {'pattern_corr': pattern_corr(r, truth)} if 'multi ' in m else {}
        for mode, excl in (('strict', False), ('excl', True)):
            rows.append({'scenario': scenario, 'rep': rep, 'method': m, 'eval': mode, **extra,
                         **ev.score_method(r, truth, names, exclude_collateral=excl)})
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--scen', default=','.join(SCEN))
    ap.add_argument('--out', default='metrics_multi.csv')
    a = ap.parse_args()
    jobs = [(s, r) for s in a.scen.split(',') for r in range(a.reps)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, out in enumerate(ex.map(one, jobs), 1):
            rows += out
            if i % 25 == 0:
                print(f'[{i}/{len(jobs)}]', flush=True)
    pd.DataFrame(rows).to_csv(ev.HERE / a.out, index=False)
