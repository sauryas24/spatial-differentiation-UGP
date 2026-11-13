"""Fetch Kuppe et al. 2022 (myocardial infarction) Visium samples from Zenodo 6578047
and keep a compact copy: raw counts for CellChat LR genes, library size, cell-type
proportions (from the authors' deconvolution) and coordinates. Each original file
is deleted after extraction to save disk.
    python3 mi_fetch.py
Writes data/kuppe_mi/<sample>.npz
"""
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path.insert(0, str(HERE))
from real_structured import lr_table  # noqa: E402

OUT = HERE / 'data' / 'kuppe_mi'
URL = 'https://zenodo.org/records/6578047/files/{}.h5ad?download=1'
SAMPLES = ['Visium_control_P1', 'Visium_control_P7', 'Visium_control_P8', 'Visium_control_P17',
           'Visium_IZ_P3', 'Visium_IZ_P10', 'Visium_IZ_P15', 'Visium_IZ_P16',
           'Visium_FZ_P14', 'Visium_FZ_P18', 'Visium_FZ_P20', 'Visium_FZ_GT_P4', 'Visium_FZ_GT_P19',
           'Visium_RZ_P6', 'Visium_RZ_P9', 'Visium_RZ_P11']
TYPES = ['Adipocyte', 'Cardiomyocyte', 'Endothelial', 'Fibroblast', 'Lymphoid', 'Mast',
         'Myeloid', 'Neuronal', 'Pericyte', 'Cycling.cells', 'vSMCs']


def extract(path, genes):
    a = ad.read_h5ad(path)
    raw = a.raw.to_adata()
    keep = [g for g in genes if g in raw.var_names]
    X = raw[:, keep].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return dict(X=X.astype(np.float32), genes=np.array(keep),
                lib=np.asarray(raw.X.sum(1)).ravel(),
                P=a.obs[TYPES].to_numpy(float), types=np.array(TYPES),
                xy=np.asarray(a.obsm['X_spatial'], float))


if __name__ == '__main__':
    genes = sorted({g for _, l, r, _ in lr_table() for g in l + r})
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SAMPLES:
        dest = OUT / f'{s}.npz'
        if dest.exists():
            continue
        tmp = OUT / f'{s}.h5ad'
        if not tmp.exists():
            subprocess.run(['curl', '-sSL', '--fail', '--retry', '5', '--retry-all-errors', '-o', str(tmp), URL.format(s)], check=True)
        np.savez_compressed(dest, **extract(tmp, genes))
        tmp.unlink()
        print(s, f'{dest.stat().st_size / 1e6:.1f} MB', flush=True)
