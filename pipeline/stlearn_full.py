"""Rerun stLearn on the gut sections for the same candidate pairs the structured
method tests (real_structured_pairs.csv), instead of stLearn's own database behind
stlearn_results.csv. stLearn takes one gene per side, so a multi-subunit ligand or
receptor becomes a pseudo-gene whose count is the minimum over its subunits (the
heteromeric rule COMMOT and CellChat use). Settings as in benchmark/run_tools.py.
    python3 stlearn_full.py [--samples A1,A2,...]
Writes stlearn_results_full.csv (pairs x samples, fraction of spots with significant
activity; 0 = no significant spots or below stLearn's min_spots).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path[:0] = [str(HERE), str(HERE / 'benchmark')]
from real_structured import G1, G2, lr_table  # noqa: E402
import run_tools as rt  # noqa: E402


def counts_with_complexes(s, sides):
    a = ad.read_h5ad(GUT_DATA / f'{s}.h5ad')
    X = sparse.csr_matrix(a.raw.X)
    vi = {g: i for i, g in enumerate(a.raw.var_names)}
    names, extra = [], []
    for genes in sides:
        if len(genes) > 1:
            names.append('.'.join(genes))      # no '_' : stLearn splits pairs on it
            extra.append(X[:, [vi[g] for g in genes]].min(axis=1).toarray())
    X = sparse.hstack([X] + [sparse.csr_matrix(e) for e in extra]).tocsr() if extra else X
    var = pd.DataFrame(index=list(a.raw.var_names) + names)
    return ad.AnnData(X, obs=pd.DataFrame(index=a.obs_names), var=var, obsm={'spatial': np.asarray(a.obsm['spatial'])})


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default=','.join(G1 + G2))
    ap.add_argument('--out', default='stlearn_results_full.csv')
    a = ap.parse_args()
    names = list(pd.read_csv(RESULTS / 'real_structured_pairs.csv', index_col=0).index)
    lr = {p[0]: p for p in lr_table()}
    sides = {tuple(lr[n][k]) for n in names for k in (1, 2)}
    pairs = [(n, '.'.join(lr[n][1]), '.'.join(lr[n][2])) for n in names]
    back = {f'{l}_{r}': n for n, l, r in pairs}
    out = {}
    for s in a.samples.split(','):
        t = time.time()
        res = rt.run_stlearn(counts_with_complexes(s, sides), pairs)
        out[s] = res.rename(index=back).reindex(names).fillna(0.0)
        print(f'{s}: {(out[s] > 0).sum()} of {len(names)} pairs with significant spots, {time.time() - t:.0f}s', flush=True)
    pd.DataFrame(out).to_csv(RESULTS / a.out)
