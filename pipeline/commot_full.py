"""Rerun COMMOT on the gut sections for the same candidate pairs the structured
method tests (real_structured_pairs.csv), instead of the 36 pre-filtered secreted
pairs behind commot_results.csv. Same settings as the original run (cellchat
database, 500 um distance threshold), heteromeric complexes allowed; contact and
ECM pairs are included so every tool sees the same pairs.
    python3 commot_full.py [--samples A1,A2,...] [--chunk 20] [--limit N]
Writes commot_results_full.csv (pairs x samples, total signal; 0 = no signal).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import pandas as pd

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path.insert(0, str(HERE))
from real_structured import G1, G2, lr_table  # noqa: E402

DIS_THR = 500


def run_sample(s, df, chunk):
    import commot as ct
    a = ad.read_h5ad(GUT_DATA / f'{s}_commot.h5ad')
    a = ad.AnnData(a.X, obs=a.obs[[]], var=a.var[[]], obsm={'spatial': a.obsm['spatial']})
    tot = {}
    for i in range(0, len(df), chunk):              # chunks keep the per-pair matrices small
        part = df.iloc[i:i + chunk].reset_index(drop=True)
        ct.tl.spatial_communication(a, database_name='full', df_ligrec=part, dis_thr=DIS_THR,
                                    heteromeric=True, pathway_sum=False)
        for l, r in zip(part.ligand, part.receptor):
            key = f'commot-full-{l}-{r}'
            tot[f'{l}_{r}'] = a.obsp[key].sum() if key in a.obsp else 0.0
        for k in [k for k in a.obsp if k.startswith('commot-full')]:
            del a.obsp[k]
        for k in [k for k in a.obsm if k.startswith('commot-full')]:
            del a.obsm[k]
    return pd.Series(tot)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default=','.join(G1 + G2))
    ap.add_argument('--chunk', type=int, default=20)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='commot_results_full.csv')
    a = ap.parse_args()
    names = list(pd.read_csv(RESULTS / 'real_structured_pairs.csv', index_col=0).index)
    lr = {p[0]: p for p in lr_table()}
    df = pd.DataFrame([('_'.join(lr[n][1]), '_'.join(lr[n][2]), lr[n][3]) for n in names],
                      columns=['ligand', 'receptor', 'pathway'])
    if a.limit:
        df = df.iloc[:a.limit]
    # saved after every sample; a rerun skips samples already in the output file
    path = RESULTS / a.out
    out = pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()
    for s in a.samples.split(','):
        if s in out:
            print(f'{s}: already done, skipped', flush=True)
            continue
        t = time.time()
        out[s] = run_sample(s, df, a.chunk)
        out.to_csv(path)
        print(f'{s}: {len(df)} pairs, {(out[s] > 0).sum()} with signal, {time.time() - t:.0f}s', flush=True)
