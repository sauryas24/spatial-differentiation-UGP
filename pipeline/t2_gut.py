"""T2 SAM (multivariate SAM over SpatialDM, COMMOT and stLearn) on the gut, fixed.

Fixes against the original run in final.ipynb:
1. Pair coverage: COMMOT and stLearn are rerun on the same 139 candidate pairs the
   structured method tests (commot_full.py, stlearn_full.py) instead of their own
   pre-filtered databases, which left only 13 pairs common to all three tools.
2. Section size: COMMOT's score is a total over spots, so it grew with section size
   (adult sections are 2-4x larger) and made nearly every pair look higher in adult.
   It is divided by the number of spots (mean signal per spot) and log-transformed.
   stLearn's fraction of significant spots also tracked section size/depth, so every
   tool's scores are then expressed relative to each sample's median pair, the same
   principle as the structured method's composition offset.
3. q-values: the pi0 estimate could reach 0 and force every q-value to 0
   (fixed in samdiff/multivariate.py). The q90 column uses the stricter q-value of
   the structured method (90th percentile of null counts instead of the mean),
   which guards against labels that line up with sample-wide differences.
4. Direction: t_dir (samdiff.multivariate.directional_t) tests one common shift
   shared by the three tools, so tools pointing opposite ways cancel instead of adding
   up as in T2; q90_dir uses the same relabellings and q90 rule.
    python3 t2_gut.py [--old]     (--old reproduces the original inputs)
Writes t2_gut_results.csv.
"""
import argparse
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path[:0] = [str(HERE), str(HERE / 'benchmark')]
import evaluate as ev  # noqa: E402,F401
from real_structured import G1, G2  # noqa: E402


def fixed_tables():
    names = list(pd.read_csv(RESULTS / 'real_structured_pairs.csv', index_col=0).index)
    n_spots = pd.Series({s: ad.read_h5ad(GUT_DATA / f'{s}.h5ad', backed='r').n_obs for s in G1 + G2})
    commot = pd.read_csv(RESULTS / 'commot_results_full.csv', index_col=0)[G1 + G2]
    commot = np.log1p(commot / n_spots)                     # mean signal per spot, log scale
    stlearn = pd.read_csv(RESULTS / 'stlearn_results_full.csv', index_col=0)[G1 + G2]
    stlearn = np.log1p(stlearn / stlearn.values[stlearn.values > 0].mean())
    spatialdm = pd.read_csv(RESULTS / 'spatialdm_results.csv', index_col=0)[G1 + G2]
    tabs = {'spatialdm': spatialdm, 'commot': commot, 'stlearn': stlearn}
    # each sample relative to its own median pair: removes sample-wide shifts (depth,
    # section size, number of spots), as the structured method's offset does
    return {m: (t - t.reindex(names).median()).reindex(names) for m, t in tabs.items()}


def run(tabs):
    from samdiff import multivariate as mv
    z = {m: (t - np.nanmean(t.values)) / np.nanstd(t.values) for m, t in tabs.items()}
    multi, pairs = mv.build_multivariate_table(z, G1 + G2)
    res, _, d0 = mv.run_multivariate_sam(multi, pairs, G1, G2, len(z))
    from samdiff import structured as st
    res.insert(2, 'q90', st._qvalues(res['T2'].to_numpy(), res.attrs['null'].to_numpy(), 'q90'))
    # directional version: |t_dir| against the relabelled |t_dir|, same q90 rule
    res['q90_dir'] = st._qvalues(res['t_dir'].abs().to_numpy(), res.attrs['null_t_dir'].abs().to_numpy(), 'q90')
    res['higher_in_dir'] = np.where(res['t_dir'] > 0, 'adult', 'fetal')
    res['higher_in'] = np.where(res[[c for c in res if c.startswith('diff_')]].mean(1) > 0, 'adult', 'fetal')
    res.attrs['d0'] = d0
    return res


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', action='store_true')
    a = ap.parse_args()
    tabs = ({m: pd.read_csv(RESULTS / f'{m}_results.csv', index_col=0) for m in ['spatialdm', 'commot', 'stlearn']}
            if a.old else fixed_tables())
    res = run(tabs)
    for col in ('q', 'q90', 'q90_dir'):
        c = res[res[col] < 0.1]
        print(f'pairs tested: {len(res)} | called at {col} < 0.1: {len(c)} '
              f'({(c.higher_in == "adult").sum()} adult, {(c.higher_in == "fetal").sum()} fetal by mean diff; '
              f'{(c.higher_in_dir == "adult").sum()} adult, {(c.higher_in_dir == "fetal").sum()} fetal by t_dir)')
    if not a.old:
        res.to_csv(RESULTS / 't2_gut_results.csv')
    print(res.head(20).round(3).to_string())
