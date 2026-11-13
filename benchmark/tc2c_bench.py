"""Benchmark: the Tensor-cell2cell-style analysis (tc2c_analysis.py) on the simulated
cohorts of fair_sdm.py (same seeds, so the numbers line up with metrics_fair_sdm.csv),
run next to the structured method and SpatialDM on the same pairs.

Per cohort: magnitude-rank tensor (contexts x pairs x senders x receivers), elbow
rank, best-of-10 non-negative CP. Scored two ways:
  programs : a planted program (A or B) is recovered if some factor's context loadings
             are higher in the program's condition (t-test with BH q < 0.05, the
             protocol's test) and at least half of that factor's top-|program| LR
             pairs are planted pairs of the program. False programs: on cohorts with
             no planted change, factors whose context loadings differ between groups
             (t-test q < 0.05, or the exact relabelling p at its minimum).
  pairs    : rank pairs by the group difference the factors reconstruct,
             sum_f w_f (mean_A c_f - mean_B c_f) b_lf (sum s_f)(sum r_f), then AUPRC.
    python3 tc2c_bench.py --reps 10 --workers 6
Writes metrics_tc2c.csv (pair metrics) and programs_tc2c.csv (program metrics).
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'

import argparse
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import ct_scores as cs
import evaluate as ev
import fair_sdm as fs
import run_benchmark as rb
import run_tools as rt
import simulate as sim
from samdiff import structured as st

sys.path[:0] = [str(ev.HERE.parent), str(ev.HERE.parent / 'pipeline')]
import tc2c_analysis as ta  # noqa: E402
from tc2c_build import magnitude  # noqa: E402

SCEN = ['fold4', 'sdm_flat', 'sdm_hiexpr', 'sdm_reloc_flat', 'null', 'sdm_null_flat']


def tensor_scores(F, names, cond, rank=None):
    """Decompose (rank by elbow unless given); return the pair table and per-factor
    context tests."""
    if rank is None:
        _, rank = ta.elbow(F, upper=6, runs=3)
    w, factors, err = ta.nn_cp(F, rank, runs=10)
    w, (C, B, S, R) = ta.normalise(w, factors)
    cols = [f'Factor {i + 1}' for i in range(rank)]
    Cdf = pd.DataFrame(C, index=list(cond), columns=cols)
    tests, groups = ta.condition_tests(Cdf, cond)
    a = np.array([cond[s] == 'A' for s in cond])
    dc = C[a].mean(0) - C[~a].mean(0)
    delta = B @ (w * dc * S.sum(0) * R.sum(0))
    pairs = pd.DataFrame({'score': np.abs(delta), 'sign': np.sign(delta), 'q': np.nan}, index=names)
    return pairs, tests, pd.DataFrame(B, index=names, columns=cols), rank, err


def program_metrics(tests, Bdf, truth, names):
    out = {}
    q = tests['q_A_vs_B'] if 'q_A_vs_B' in tests else tests.filter(like='q_').iloc[:, 0]
    higher = tests['highest_in']
    min_p = tests['p_relabel'].min()
    for prog, cond in (('A', 'A'), ('B', 'B')):
        planted = [p for p in truth.pair[truth.program == prog] if p in names]
        if not planted:
            continue
        hit = False
        for f in Bdf.columns:
            top = Bdf[f].nlargest(len(planted)).index
            if q[f] < 0.05 and higher[f] == cond and np.mean([p in planted for p in top]) >= 0.5:
                hit = True
        out[f'recovered_{prog}'] = hit
    out['n_factors'] = len(Bdf.columns)
    out['factors_q05'] = int((q < 0.05).sum())
    out['factors_relabel_min'] = int((tests['p_relabel'] <= min_p + 1e-12).sum()) if min_p < 0.2 else 0
    return out


def one(job):
    warnings.filterwarnings('ignore')
    scenario, rep = job
    ref = pickle.load(open(ev.HERE / 'ref.pkl', 'rb'))
    lr = sim.single_chain_secreted(*rb.LR_ARGS)
    layout = sim.load_layout(rb.DATA / 'coord.csv', rb.DATA / 'interest_region_Spot_IDs.csv')
    adatas, truth, (g1, g2) = sim.simulate_cohort(ref, layout, lr, sim.SCENARIOS[scenario],
                                                  90000 + 1000 * fs.SCEN.index(scenario) + rep)
    db = rt.commot_db(list(adatas.values()), lr)
    pairs = [(f'{a}_{b}', a, b) for a, b in zip(db.iloc[:, 0], db.iloc[:, 1])]
    K = cs.kernel(layout[0].to_numpy())
    samples = g1 + g2
    slices = {s: cs.ct_slices(adatas[s], pairs, 'abundance', K) for s in samples}
    z = pd.DataFrame({s: fs.spatialdm_z(adatas[s], 0.9) for s in samples})
    names = sorted(set(slices[samples[0]]) & set(z.index))
    n, n1 = len(samples), len(g1)
    raw = np.stack([[slices[s][p] for p in names] for s in samples])
    cond = {s: s[0] for s in samples}                       # sample names start with A / B
    variants = {'Tensor-c2c (protocol: elbow rank, rank scores)': ('rank', None),
                'Tensor-c2c (rank 6)': ('rank', 6),
                'Tensor-c2c (log scores)': ('log', None)}
    res, progs = {}, []
    for label, (score, rk) in variants.items():
        tpairs, tests, Bdf, rank, err = tensor_scores(magnitude(raw, score), names, cond, rk)
        res[label] = tpairs
        progs.append({'scenario': scenario, 'rep': rep, 'variant': label, 'rank': rank, 'error': err,
                      **program_metrics(tests, Bdf, truth, names)})
    F = st.commot_features(slices, samples, names, offset='trend')
    res['Structured v2'] = st.structured_rank([F], names, n, n1, rank=1, score='l2', qmode='q90')
    res['SpatialDM LRT'] = ev.spatialdm_lrt(z.reindex(names), g1, g2)
    rows = [{'scenario': scenario, 'rep': rep, 'method': m, 'eval': e,
             **ev.score_method(r, truth, names, exclude_collateral=(e == 'excl'))}
            for m, r in res.items() for e in ('strict', 'excl')]
    return rows, progs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--scen', default=','.join(SCEN))
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    jobs = [(s, r) for s in a.scen.split(',') for r in range(a.reps)]
    rows, progs = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (out, prog) in enumerate(ex.map(one, jobs), 1):
            rows += out
            progs += prog
            print(f'[{i}/{len(jobs)}] {prog[0]["scenario"]} {prog[0]["rep"]}', flush=True)
    pd.DataFrame(rows).to_csv(ev.HERE / f'metrics_tc2c{a.tag}.csv', index=False)
    pd.DataFrame(progs).to_csv(ev.HERE / f'programs_tc2c{a.tag}.csv', index=False)
