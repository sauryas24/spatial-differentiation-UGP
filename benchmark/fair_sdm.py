"""Head-to-head with SpatialDM, and how much the outcome depends on the simulator.

Per cohort, SpatialDM is run with the package (log-normalised data, weight_matrix,
extract_lr, spatialdm_global z-scores) at two kernel widths, then tested three ways:
its differential_test LRT, its group_differential_pairs call set, and SAM on its z.
Against: structured v2 (q90, maxT) on the kernel scores, SAM on the kernel total,
and SAM on log(mean L x mean R), a non-spatial expression baseline. All methods are
scored on the pairs both SpatialDM and the kernel score test.

Scenarios (simulate.py, sdm_*): baseline fold4, no / weak composition patches,
well-expressed planted genes, 6 vs 6, and 'reloc' where only the co-localisation of
ligand and receptor changes (same amounts in both conditions).
    python3 fair_sdm.py --reps 10 --workers 6
Writes metrics_fair_sdm.csv and zstats_fair_sdm.csv (null-pair z by scenario).
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'

import argparse
import contextlib
import io
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

SCEN = ['fold4', 'sdm_null_flat', 'sdm_flat', 'sdm_weak', 'sdm_hiexpr', 'sdm_n6',
        'sdm_reloc', 'sdm_reloc_flat', 'sdm_reloc_n6', 'null']
SDM_L = {'': 0.9, ' l x2': 1.8}       # SpatialDM kernel length in spot spacings


def spatialdm_z(adata, l):
    with contextlib.redirect_stdout(io.StringIO()):
        return rt.run_spatialdm(adata, l=l * sim.SPACING)


def expr_table(adatas, pairs):
    """log(mean ligand x mean receptor) of library-normalised expression per sample."""
    out = {}
    for s, a in adatas.items():
        X = np.asarray(a.X, float)
        X = X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4
        m = pd.Series(X.mean(0), index=a.var_names)
        out[s] = {p: np.log(m[l] * m[r] + 1e-6) for p, l, r in pairs}
    return pd.DataFrame(out)


def one(job):
    warnings.filterwarnings('ignore')
    scenario, rep = job
    ref = pickle.load(open(ev.HERE / 'ref.pkl', 'rb'))
    lr = sim.single_chain_secreted(*rb.LR_ARGS)
    layout = sim.load_layout(rb.DATA / 'coord.csv', rb.DATA / 'interest_region_Spot_IDs.csv')
    adatas, truth, (g1, g2) = sim.simulate_cohort(ref, layout, lr, sim.SCENARIOS[scenario],
                                                  90000 + 1000 * SCEN.index(scenario) + rep)
    db = rt.commot_db(list(adatas.values()), lr)
    pairs = [(f'{a}_{b}', a, b) for a, b in zip(db.iloc[:, 0], db.iloc[:, 1])]
    K = cs.kernel(layout[0].to_numpy())
    slices = {s: cs.ct_slices(a, pairs, 'abundance', K) for s, a in adatas.items()}
    samples = g1 + g2
    z = {tag: pd.DataFrame({s: spatialdm_z(adatas[s], l) for s in samples}) for tag, l in SDM_L.items()}
    names = sorted(set(slices[samples[0]]) & set(z[''].index))
    pairs = [p for p in pairs if p[0] in set(names)]
    n, n1 = len(samples), len(g1)

    from samdiff import tensor as tc
    res = {}
    for tag, zt in z.items():
        zt = zt.reindex(names)
        res[f'SpatialDM LRT{tag}'] = ev.spatialdm_lrt(zt, g1, g2)
        res[f'SpatialDM group_diff_pairs{tag}'] = ev.spatialdm_group(zt, g1, g2)
        res[f'SAM on SpatialDM z{tag}'] = ev.sam_method(zt.fillna(0), g1, g2)
    tot = pd.DataFrame({s: {p: slices[s][p].sum() for p in names} for s in samples})
    res['SAM kernel total'] = ev.sam_method(tc.transform_table(tot), g1, g2)
    res['SAM expression (non-spatial)'] = ev.sam_method(expr_table(adatas, pairs), g1, g2)
    F = st.commot_features({s: {p: slices[s][p] for p in names} for s in samples}, samples, names, offset='trend')
    for q in ('q90', 'maxT'):
        res[f'Structured v2 {q}'] = st.structured_rank([F], names, n, n1, rank=1, score='l2', qmode=q)

    rows = [{'scenario': scenario, 'rep': rep, 'method': m, 'eval': e,
             **ev.score_method(r.reindex(names), truth, names, exclude_collateral=(e == 'excl'))}
            for m, r in res.items() for e in ('strict', 'excl')]
    planted = set(truth.pair[truth.direction != 0])
    col = set(ev.collateral(truth, names))
    zrow = []
    for tag, zt in z.items():
        zt = zt.reindex(names).fillna(0)
        kind = pd.Series(['planted' if p in planted else 'collateral' if p in col else 'null'
                          for p in names], index=names)
        for k in ('null', 'planted'):
            zz = zt[kind == k]
            if len(zz):
                zrow.append({'scenario': scenario, 'rep': rep, 'kernel': tag or ' l x1', 'pairs': k,
                             'median_z': np.median(zz.values), 'frac_sig': (zz.values > 1.64).mean(),
                             'between_sample_sd': np.median(np.r_[zz[g1].std(1), zz[g2].std(1)]),
                             'median_abs_group_diff': (zz[g1].mean(1) - zz[g2].mean(1)).abs().median()})
    return rows, zrow


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--scen', default=','.join(SCEN))
    ap.add_argument('--out', default='metrics_fair_sdm.csv')
    a = ap.parse_args()
    jobs = [(s, r) for s in a.scen.split(',') for r in range(a.reps)]
    rows, zrows = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (out, zr) in enumerate(ex.map(one, jobs), 1):
            rows += out
            zrows += zr
            print(f'[{i}/{len(jobs)}] {out[0]["scenario"]} {out[0]["rep"]}', flush=True)
    pd.DataFrame(rows).to_csv(ev.HERE / a.out, index=False)
    pd.DataFrame(zrows).to_csv(ev.HERE / a.out.replace('metrics', 'zstats'), index=False)
