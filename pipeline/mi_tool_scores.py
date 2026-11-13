"""COMMOT, SpatialDM and stLearn per-sample scores on the 14 myocardial infarction
sections, for the 126 pairs of tc2c/mi_meta.json (input for T2 SAM and SpatialDM's
differential test on the heart).

The section files hold counts for the ligand/receptor genes only, plus each spot's
total count, so normalisation uses those totals: log(1 + 1e4 * count / total).
Settings follow the gut runs, scaled by the spot spacing (nn):
  COMMOT    : total signal (sum over spots of sent signal), distance limit 6 spacings
  SpatialDM : global z-score, kernel l = 0.9 spacing, cutoff 0.2
  stLearn   : fraction of spots with significant activity (counts as given)
    python3 mi_tool_scores.py [--workers 4]
Writes tc2c/mi_tools/<sample>.csv (pairs x [commot, spatialdm, stlearn]) per section,
then mi_tool_results_<tool>.csv (pairs x samples).
"""
import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path[:0] = [str(HERE), str(HERE / 'benchmark')]
OUT = RESULTS / 'tc2c' / 'mi_tools'
META = json.load(open(RESULTS / 'tc2c' / 'mi_meta.json'))


def load(s):
    import mi_structured as mis
    from scipy.spatial import cKDTree
    z = np.load(mis.DATA / f'Visium_{s}.npz', allow_pickle=False)
    counts = np.asarray(z['X'], float)
    lib = np.maximum(np.asarray(z['lib'], float), 1)
    xy = np.asarray(z['xy'], float)
    nn = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    genes = [str(g) for g in z['genes']]
    obs = pd.DataFrame(index=[f'spot{i}' for i in range(len(xy))])
    norm = ad.AnnData(np.log1p(counts / lib[:, None] * 1e4), obs=obs, var=pd.DataFrame(index=genes),
                      obsm={'spatial': xy})
    raw = ad.AnnData(counts, obs=obs.copy(), var=pd.DataFrame(index=genes), obsm={'spatial': xy})
    return norm, raw, nn


def complexes(raw, sides):
    """Counts plus minimum-of-subunit pseudo-genes for multi-subunit ligands/receptors."""
    X, vi = sparse.csr_matrix(raw.X), {g: i for i, g in enumerate(raw.var_names)}
    names, extra = [], []
    for genes in sides:
        if len(genes) > 1 and all(g in vi for g in genes):
            names.append('.'.join(genes))
            extra.append(X[:, [vi[g] for g in genes]].min(axis=1).toarray())
    X = sparse.hstack([X] + [sparse.csr_matrix(e) for e in extra]).tocsr() if extra else X
    return ad.AnnData(X, obs=raw.obs.copy(), var=pd.DataFrame(index=list(raw.var_names) + names),
                      obsm={'spatial': raw.obsm['spatial']})


def one(s):
    warnings.filterwarnings('ignore')
    import run_tools as rt
    import spot_scores as ss
    from real_structured import lr_table
    path = OUT / f'{s}.csv'
    if path.exists():
        return s, 'already done'
    t = time.time()
    lr = {p[0]: p for p in lr_table()}
    pairs = META['pairs']
    norm, raw, nn = load(s)
    out = pd.DataFrame(index=pairs)

    sdm = ss._spatialdm_module()
    a = norm.copy()
    sdm.weight_matrix(a, l=0.9 * nn, cutoff=0.2, single_cell=False)
    sdm.extract_lr(a, 'human', min_cell=3)
    sdm.spatialdm_global(a, 1000, specified_ind=None, method='z-score', nproc=1)
    out['spatialdm'] = pd.Series(np.asarray(a.uns['global_res']['z'], float), index=a.uns['global_res'].index)

    S = ss.commot_scores(norm, [(p, '_'.join(lr[p][1]), '_'.join(lr[p][2]), lr[p][3]) for p in pairs], 6 * nn)
    out['commot'] = S.sum(0) / 2                       # sent + received per spot -> total flow

    sides = {tuple(lr[p][k]) for p in pairs for k in (1, 2)}
    stl_pairs = [(p, '.'.join(lr[p][1]), '.'.join(lr[p][2])) for p in pairs]
    back = {f'{l}_{r}': p for p, l, r in stl_pairs}
    res = rt.run_stlearn(complexes(raw, sides), stl_pairs)
    out['stlearn'] = res.rename(index=back).reindex(pairs).fillna(0.0)
    out.to_csv(path)
    return s, f'{time.time() - t:.0f}s, SpatialDM scored {out.spatialdm.notna().sum()} pairs'


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--samples', default=','.join(META['contexts']))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    samples = a.samples.split(',')
    with ProcessPoolExecutor(a.workers) as ex:
        for s, msg in ex.map(one, samples):
            print(s, msg, flush=True)
    done = [s for s in META['contexts'] if (OUT / f'{s}.csv').exists()]
    tabs = {s: pd.read_csv(OUT / f'{s}.csv', index_col=0) for s in done}
    for tool in ('commot', 'spatialdm', 'stlearn'):
        pd.DataFrame({s: t[tool] for s, t in tabs.items()}).to_csv(RESULTS / f'mi_tool_results_{tool}.csv')
