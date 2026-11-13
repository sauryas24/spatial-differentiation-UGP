"""Run SpatialDM, COMMOT and stLearn on one simulated sample.

Settings follow the project's real-data runs (final.ipynb, STdiff_fixed.ipynb),
with distances rescaled from that data's spot spacing (~83 units) to the
simulator's 100 um: SpatialDM l = 75 -> 90, COMMOT dis_thr = 500 -> 600.
Each tool returns one score per LR pair; COMMOT also returns the
sender x receiver cell-type signal (P.T @ W @ P) used by tensor B.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STLEARN_PY = Path(__file__).resolve().parent / 'stlearn_env' / 'bin' / 'python'
SPACING = 100.0


def _normalized(adata):
    import scanpy as sc
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def run_spatialdm(adata, l=0.9 * SPACING):
    """Global bivariate Moran's z per LR pair (z-score method, as in final.ipynb)."""
    sys.path.insert(0, str(ROOT / 'SpatialDM'))
    import spatialdm as sdm
    import spatialdm.main as sdm_main
    local = {'https://figshare.com/ndownloader/files/36638943': ROOT / 'resources' / 'interaction_input_CellChatDB.csv',
             'https://figshare.com/ndownloader/files/36638940': ROOT / 'resources' / 'complex_input_CellChatDB.csv'}
    if not getattr(sdm_main.pd, '_local_patch', False):
        orig = sdm_main.pd.read_csv

        def read_csv(path, *a, **k):
            return orig(local.get(path, path), *a, **k)
        sdm_main.pd = type(sys)('pd_patched')
        sdm_main.pd.__dict__.update(pd.__dict__)
        sdm_main.pd.read_csv = read_csv
        sdm_main.pd._local_patch = True
    a = _normalized(adata)
    sdm.weight_matrix(a, l=l, cutoff=0.2, single_cell=False)
    sdm.extract_lr(a, 'human', min_cell=3)
    sdm.spatialdm_global(a, 1000, specified_ind=None, method='z-score', nproc=1)
    res = a.uns['global_res']
    return pd.Series(np.asarray(res['z'], float), index=res.index)


def commot_db(adatas, lr_pairs, min_cell_pct=0.05):
    """CellChat single-chain secreted pairs passing COMMOT's filter on the pooled
    samples (label-blind; the real-data run filtered on A1 alone)."""
    import anndata as ad
    import commot as ct
    df = pd.DataFrame([(l, r, 'x') for _, l, r in lr_pairs], columns=[0, 1, 2])
    pooled = ad.concat([_normalized(a) for a in adatas], join='inner')
    return ct.pp.filter_lr_database(df, pooled, min_cell_pct=min_cell_pct)


def run_commot(adata, df_ligrec, types):
    """Total COMMOT signal per pair, plus lineage-level P.T @ W @ P per pair."""
    import commot as ct
    a = _normalized(adata)
    ct.tl.spatial_communication(a, database_name='cellchat', df_ligrec=df_ligrec,
                                dis_thr=6 * SPACING, heteromeric=False, pathway_sum=False)
    P = adata.obs[types].to_numpy(float)
    totals, slices = {}, {}
    for l, r in zip(df_ligrec.iloc[:, 0], df_ligrec.iloc[:, 1]):
        key = f'commot-cellchat-{l}-{r}'
        if key in a.obsp:
            W = a.obsp[key]
            totals[f'{l}_{r}'] = W.sum()
            slices[f'{l}_{r}'] = P.T @ (W @ P) / a.n_obs
    return pd.Series(totals), slices


_STLEARN_SCRIPT = r'''
import sys, json, numpy as np, pandas as pd, anndata as ad, stlearn as st
from sklearn.neighbors import NearestNeighbors
inp, lrs_path, out, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
np.random.seed(seed)
adata = ad.read_h5ad(inp)
lrs = np.array(json.load(open(lrs_path)))
st.pp.filter_genes(adata, min_cells=3)
st.pp.normalize_total(adata)
coords = adata.obsm['spatial']
adata.obs['imagerow'] = coords[:, 1]
adata.obs['imagecol'] = coords[:, 0]
d, _ = NearestNeighbors(n_neighbors=6).fit(coords).kneighbors(coords)
st.tl.cci.run(adata, lrs, min_spots=20, distance=d[:, 1:].mean() * 1.5, n_pairs=100, n_cpus=1)
s = adata.uns['lr_summary']
(s['n_spots_sig'] / s['n_spots']).to_csv(out)
'''


def run_stlearn(adata, lr_pairs, seed=42):
    """stLearn fraction of spots with significant LR activity (n_spots_sig / n_spots)."""
    if not STLEARN_PY.exists():
        raise FileNotFoundError('stlearn_env missing; run the install step first')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = adata.copy()
        a.obs = a.obs[[]]
        a.write_h5ad(tmp / 'in.h5ad')
        json.dump([f'{l}_{r}' for _, l, r in lr_pairs], open(tmp / 'lrs.json', 'w'))
        (tmp / 'run.py').write_text(_STLEARN_SCRIPT)
        env = dict(os.environ, OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
        proc = subprocess.run([str(STLEARN_PY), '-I', str(tmp / 'run.py'), str(tmp / 'in.h5ad'),
                               str(tmp / 'lrs.json'), str(tmp / 'out.csv'), str(seed)],
                              capture_output=True, text=True, env=env)
        if proc.returncode:
            raise RuntimeError(proc.stderr[-2000:])
        return pd.read_csv(tmp / 'out.csv', index_col=0).iloc[:, 0]
