"""Structured differential ranking (v2) on the real adult vs fetal samples.

Per sample: raw counts -> library-size normalised expression; each LR pair's
ligand / receptor (complex = min over subunits, as in CellChat) is credited to the
7 shared lineages by deconvolution abundance; a Gaussian kernel (l = 75, cutoff 0.2,
the SpatialDM settings used for these samples) gives a lineage x lineage score.
Then benchmark/structured.py: depth-trend + offset normalisation, rank-1 effect
shrinkage, L2 pair score, q90 and maxT q-values over the 15 relabellings (2 vs 4).

    python3 real_structured.py
Writes real_structured_pairs.csv, real_structured_program.csv and
real_structured_robustness.csv.
"""
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'benchmark'))
from samdiff import tensor as tc  # noqa: E402
from samdiff import structured as st  # noqa: E402

G1 = ['A1', 'A2']                 # adult
G2 = ['A3', 'A4', 'A8', 'A9']     # fetal
SAMPLES = G1 + G2
L_KERNEL, CUTOFF = 75.0, 0.2
MIN_FRAC = 0.05                   # ligand and receptor detected in >= 5% of spots of every sample
MIN_ABUND = 5.0                   # lineages below this many spot-equivalents in any sample are dropped


def lr_table(annotations=None):
    """CellChat pairs as (name, ligand subunits, receptor subunits, annotation)."""
    i = pd.read_csv(RESOURCES / 'interaction_input_CellChatDB.csv', index_col=0)
    comp = pd.read_csv(RESOURCES / 'complex_input_CellChatDB.csv', index_col=0)
    sub = lambda x: [g for g in comp.loc[x].dropna() if g] if x in comp.index else [x]
    if annotations:
        i = i[i['annotation'].isin(annotations)]
    return [(name, sub(r['ligand']), sub(r['receptor']), r['annotation']) for name, r in i.iterrows()]


def load_sample(s, genes):
    a = ad.read_h5ad(GUT_DATA / f'{s}_commot.h5ad')
    raw = a.raw.to_adata()
    keep = [g for g in genes if g in raw.var_names]
    X = raw[:, keep].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    lib = np.asarray(raw.X.sum(1)).ravel()
    X = X / np.maximum(lib, 1)[:, None] * 1e4
    P = tc.lineage_proportions(a.obs)
    return pd.DataFrame(X, columns=keep), P, np.asarray(a.obsm['spatial'], float)


def kernel(xy, l=L_KERNEL, cutoff=CUTOFF):
    K = np.exp(-cdist(xy, xy) ** 2 / (2 * l ** 2))
    K[K < cutoff] = 0
    return K


def gene_expr(X, subunits):
    return X[subunits].min(axis=1).to_numpy() if all(g in X for g in subunits) else None


def lineage_slices(X, P, K, pairs):
    """dict pair -> lineages x lineages score (abundance attribution)."""
    w = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    out = {}
    for name, lig, rec, _ in pairs:
        xl, xr = gene_expr(X, lig), gene_expr(X, rec)
        if xl is None or xr is None:
            continue
        Ls, Rr = xl[:, None] * w, xr[:, None] * w
        out[name] = Ls.T @ (K @ Rr) / len(X)
    return out


def detected(X, pairs):
    ok = set()
    for name, lig, rec, _ in pairs:
        xl, xr = gene_expr(X, lig), gene_expr(X, rec)
        if xl is not None and xr is not None and (xl > 0).mean() >= MIN_FRAC and (xr > 0).mean() >= MIN_FRAC:
            ok.add(name)
    return ok


def features(slices, names, samples, keep):
    """Normalised features restricted to sender x receiver columns of kept lineages."""
    K = len(tc.LINEAGES)
    cols = [i * K + j for i in keep for j in keep]
    return st.commot_features(slices, samples, names, offset='trend')[:, :, cols]


def run(slices, names, samples, g1, keep, rank=1, qmode='q90'):
    F = features(slices, names, samples, keep)
    return st.structured_rank([F], names, len(samples), len(g1), rank=rank, score='l2', qmode=qmode)


def program(slices, names, samples, g1, keep):
    """Rank-1 component of the observed effect matrix: lineage pattern and pair loadings."""
    F = features(slices, names, samples, keep)
    D = st._block_d(F, st.relabelings(len(samples), len(g1)))[0]
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    sgn = np.sign(Vt[0][np.abs(Vt[0]).argmax()])
    lin = [tc.LINEAGES[i] for i in keep]
    v = pd.DataFrame((sgn * Vt[0]).reshape(len(keep), len(keep)), index=lin, columns=lin)
    return v, pd.Series(sgn * U[:, 0] * s[0], index=names), s


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pairs = lr_table()
    genes = sorted({g for _, l, r, _ in pairs for g in l + r})
    data = {s: load_sample(s, genes) for s in SAMPLES}
    names = sorted(set.intersection(*[detected(X, pairs) for X, _, _ in data.values()]))
    print(f'{len(names)} pairs detected in >= {MIN_FRAC:.0%} of spots in all six samples')
    sel = [p for p in pairs if p[0] in set(names)]
    Ks = {s: kernel(xy) for s, (_, _, xy) in data.items()}
    slices = {s: lineage_slices(X, P, Ks[s], sel) for s, (X, P, _) in data.items()}
    abund = pd.DataFrame({s: P.sum(0) for s, (_, P, _) in data.items()}, index=tc.LINEAGES).T
    print('lineage abundance (spot-equivalents):\n', abund.round(0).to_string())

    keep = [i for i, l in enumerate(tc.LINEAGES) if abund[l].min() >= MIN_ABUND]
    print('lineages kept:', [tc.LINEAGES[i] for i in keep])
    q90 = run(slices, names, SAMPLES, G1, keep, qmode='q90')
    mxt = run(slices, names, SAMPLES, G1, keep, qmode='maxT')
    v, load, sv = program(slices, names, SAMPLES, G1, keep)
    ann = {n: a for n, _, _, a in pairs}
    res = pd.DataFrame({'score': q90['score'],
                        'higher_in': pd.Series(np.where(q90['sign'] > 0, 'adult', 'fetal'), index=q90.index),
                        'q90': q90['q'], 'p_maxT': mxt['q'], 'program_loading': load})
    res['annotation'] = [ann[n] for n in res.index]
    res = res.sort_values('score', ascending=False)
    res.index.name = 'pair'
    res.to_csv(RESULTS / 'real_structured_pairs.csv')
    v.to_csv(RESULTS / 'real_structured_program.csv')
    print(f'\nsingular values of the effect matrix: {np.round(sv[:5], 1)}')
    print('leading program, sender (rows) x receiver (cols); positive = higher in adult:')
    print(v.round(2).to_string())
    for thr in (0.05, 0.1, 0.2):
        c = res[res.q90 < thr]
        print(f'q90 < {thr}: {len(c)} pairs ({(c.higher_in == "adult").sum()} adult, {(c.higher_in == "fetal").sum()} fetal)')
    print(f'maxT p < 0.1: {(res.p_maxT < 0.1).sum()} pairs (floor 1/15 = 0.067)')

    # robustness: top-50 overlap under alternative settings
    top = set(res.index[:50])
    rows = []
    def overlap(r):
        return len(top & set(r.sort_values('score', ascending=False).index[:50]))
    rows.append({'setting': 'rank 2', 'top50_overlap': overlap(run(slices, names, SAMPLES, G1, keep, rank=2))})
    rows.append({'setting': 'Immune kept', 'top50_overlap': overlap(run(slices, names, SAMPLES, G1, list(range(len(tc.LINEAGES)))))})
    for l in (50.0, 150.0):
        sl = {s: lineage_slices(X, P, kernel(xy, l), sel) for s, (X, P, xy) in data.items()}
        rows.append({'setting': f'kernel l={l:.0f}', 'top50_overlap': overlap(run(sl, names, SAMPLES, G1, keep))})
    sec = [n for n in names if ann[n] == 'Secreted Signaling']
    r = run({s: {p: slices[s][p] for p in sec} for s in SAMPLES}, sec, SAMPLES, G1, keep)
    rows.append({'setting': f'secreted only ({len(sec)} pairs)',
                 'top50_overlap': len(set(res.index[res.index.isin(sec)][:50]) & set(r.sort_values('score', ascending=False).index[:50]))})
    for drop in G2:
        sm = [s for s in SAMPLES if s != drop]
        rows.append({'setting': f'drop {drop} (2 vs 3)', 'top50_overlap': overlap(run(slices, names, sm, G1, keep))})
    rob = pd.DataFrame(rows)
    rob.to_csv(RESULTS / 'real_structured_robustness.csv', index=False)
    print('\nrobustness (top-50 overlap with the main run):\n', rob.to_string(index=False))
    pd.set_option('display.width', 200)
    print('\ntop 30 pairs:\n', res.head(30).round(3).to_string())
