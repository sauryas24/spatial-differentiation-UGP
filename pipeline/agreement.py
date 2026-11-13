"""How much do the methods agree?
  gut (adult vs fetal): SpatialDM's differential test, T2 SAM, the structured method and
      the Tensor-cell2cell-style analysis
  heart (four zones, each vs the rest): the structured method and the tensor analysis

Per comparison, each method gives a signed per-pair score (positive = higher in the
first group) and, except the tensor analysis, a set of calls:
  SpatialDM  : differential_test (likelihood-ratio on per-sample z), sign of the mean z
               difference x -log10 p; calls FDR < 0.1
  T2 SAM     : t_dir; calls q90 < 0.1
  Structured : gut, signed score and q90 < 0.1 (two conditions); heart, the
               multi-condition run: the zone's denoised effect; called for a zone when
               q90 < 0.1 overall and the zone is where the pair is highest or lowest
  Tensor     : group difference reconstructed from the factors,
               sum_f w_f (mean_group c_f - mean_rest c_f) l_f (sum s_f)(sum r_f); no calls
Agreement: Spearman correlation of signed scores, overlap of calls and direction
agreement among shared calls, overlap of the top 30 pairs.
Program check (tensor analysis): for every program, are its top 20 LR pairs the pairs
the other methods call higher in the program's condition? Count against the number
expected by chance (hypergeometric p) and AUC of the program's LR loadings.
    python3 agreement.py [--heart]
Writes agreement_gut.csv, agreement_heart.csv and agreement_programs.csv.
"""
import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path[:0] = [str(HERE), str(HERE / 'benchmark')]
import evaluate as ev  # noqa: E402
from samdiff import structured as st  # noqa: E402
from samdiff import multivariate as mv  # noqa: E402

METHODS = ['SpatialDM', 'T2 SAM', 'Structured', 'Tensor']
TOP = 30
TOP_PROGRAM = 20


def spatialdm(z, g1, g2):
    r = ev.spatialdm_lrt(z, g1, g2)
    return pd.DataFrame({'score': r['sign'] * r['score'], 'called': r['q'] < 0.1})


def t2(tabs, g1, g2, n_spots):
    """T2 SAM (directional) with the preparation of t2_gut.py."""
    commot = np.log1p(tabs['commot'] / n_spots)
    stl = tabs['stlearn']
    stl = np.log1p(stl / stl.values[stl.values > 0].mean())
    prep = {'spatialdm': tabs['spatialdm'], 'commot': commot, 'stlearn': stl}
    prep = {m: t - t.median() for m, t in prep.items()}
    z = {m: (t - np.nanmean(t.values)) / np.nanstd(t.values) for m, t in prep.items()}
    samples = g1 + g2
    multi, pairs = mv.build_multivariate_table({m: t[samples] for m, t in z.items()}, samples)
    r, _, _ = mv.run_multivariate_sam(multi, pairs, g1, g2, 3)
    q = st._qvalues(r['t_dir'].abs().to_numpy(), r.attrs['null_t_dir'].abs().to_numpy(), 'q90')
    return pd.DataFrame({'score': r['t_dir'], 'called': q < 0.1}, index=r.index)


def tensor(dataset, members):
    d = RESULTS / 'tc2c' / dataset
    C = pd.read_csv(d / 'loadings_contexts.csv', index_col=0)
    B = pd.read_csv(d / 'loadings_ligand-receptor.csv', index_col=0)
    S = pd.read_csv(d / 'loadings_sender.csv', index_col=0)
    R = pd.read_csv(d / 'loadings_receiver.csv', index_col=0)
    w = pd.read_csv(d / 'condition_tests.csv', index_col=0)['weight']
    a = C.index.isin(members)
    dc = C[a].mean() - C[~a].mean()
    delta = B @ (w * dc * S.sum() * R.sum())
    return pd.DataFrame({'score': delta, 'called': np.nan})


def compare(res, label):
    """Pairwise agreement between methods on their common pairs."""
    rows = []
    common = sorted(set.intersection(*[set(r.index) for r in res.values()]))
    for a, b in combinations(res, 2):
        x, y = res[a].loc[common], res[b].loc[common]
        row = {'comparison': label, 'method_1': a, 'method_2': b, 'n_pairs': len(common),
               'spearman': spearmanr(x.score, y.score)[0]}
        ta, tb = set(x.score.abs().nlargest(TOP).index), set(y.score.abs().nlargest(TOP).index)
        row[f'top{TOP}_overlap'] = len(ta & tb)
        if x.called.notna().all() and y.called.notna().all():
            ca, cb = set(x.index[x.called.astype(bool)]), set(y.index[y.called.astype(bool)])
            both = sorted(ca & cb)
            row.update({'calls_1': len(ca), 'calls_2': len(cb), 'calls_shared': len(both),
                        'same_direction': np.mean(np.sign(x.score[both]) == np.sign(y.score[both]))
                        if both else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def programs(dataset, calls, direction):
    """Top LR pairs of each tensor program vs the pairs other methods call higher in
    the program's condition. calls: method -> DataFrame(score, called);
    direction: condition -> sign of score meaning 'higher in that condition'."""
    from scipy.stats import hypergeom, mannwhitneyu
    d = RESULTS / 'tc2c' / dataset
    L = pd.read_csv(d / 'loadings_ligand-receptor.csv', index_col=0)
    T = pd.read_csv(d / 'condition_tests.csv', index_col=0)
    rows = []
    for f in L:
        cond = T.loc[f, 'highest_in']
        if cond not in direction:
            continue
        for m, r in calls.items():
            common = L.index.intersection(r.index)
            l = L.loc[common, f]
            up = common[r.loc[common, 'called'].astype(bool).to_numpy()
                        & (np.sign(r.loc[common, 'score']) == direction[cond]).to_numpy()]
            top = l.nlargest(TOP_PROGRAM).index
            k, n, K = len(top.intersection(up)), len(common), len(up)
            rows.append({'dataset': dataset, 'program': f, 'condition': cond,
                         'p_condition': T.loc[f, 'p_relabel'], 'method': m,
                         f'top{TOP_PROGRAM}_called': k, 'expected': TOP_PROGRAM * K / n,
                         'p_enrichment': hypergeom.sf(k - 1, n, K, TOP_PROGRAM),
                         'AUC': mannwhitneyu(l[up], l.drop(up), alternative='greater')[0] / (K * (n - K))})
    return pd.DataFrame(rows)


def gut():
    from real_structured import G1, G2
    import anndata as ad
    t2r = pd.read_csv(RESULTS / 't2_gut_results.csv', index_col=0)
    s = pd.read_csv(RESULTS / 'real_structured_pairs.csv', index_col=0)
    names = list(s.index)
    z = pd.read_csv(RESULTS / 'spatialdm_results.csv', index_col=0).reindex(names)
    res = {'SpatialDM': spatialdm(z, G1, G2),
           'T2 SAM': pd.DataFrame({'score': t2r['t_dir'], 'called': t2r['q90_dir'] < 0.1}),
           'Structured': pd.DataFrame({'score': np.where(s.higher_in == 'adult', 1, -1) * s.score,
                                       'called': s.q90 < 0.1}, index=names),
           'Tensor': tensor('gut', G1)}
    prog = programs('gut', {m: res[m] for m in ('SpatialDM', 'T2 SAM', 'Structured')},
                    {'adult': 1, 'fetal': -1})
    return compare(res, 'gut: adult vs fetal'), res, prog


def heart():
    meta = json.load(open(RESULTS / 'tc2c' / 'mi_meta.json'))
    cond, samples = meta['condition'], meta['contexts']
    multi = pd.read_csv(RESULTS / 'mi_multi_pairs.csv', index_col=0)
    out, allres = [], {}
    for zone in ('control', 'RZ', 'IZ', 'FZ'):
        g1 = [s for s in samples if cond[s] == zone]
        # called for a zone: significant overall and the zone is where the pair is highest or lowest
        res = {'Structured': pd.DataFrame({'score': multi[f'effect_{zone}'],
                                           'called': (multi.q90 < 0.1) & ((multi.highest_in == zone)
                                                                          | (multi.lowest_in == zone))}),
               'Tensor': tensor('mi', g1)}
        out.append(compare(res, f'heart: {zone} vs rest'))
        allres[zone] = res
    # program check: the structured profile of each pair, 'higher in zone' = called and above average there
    zones = ('control', 'RZ', 'IZ', 'FZ')
    prog = pd.concat([programs('mi', {'Structured': pd.DataFrame(
        {'score': multi[f'effect_{z}'], 'called': multi.q90 < 0.1})}, {z: 1}) for z in zones])
    return pd.concat(out), allres, prog


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--heart', action='store_true')
    a = ap.parse_args()
    pd.set_option('display.width', 250)
    g, gres, gprog = gut()
    g.to_csv(RESULTS / 'agreement_gut.csv', index=False)
    progs = [gprog]
    print({m: int(r.called.sum()) if r.called.notna().all() else None for m, r in gres.items()})
    print(g.round(2).to_string(index=False))
    if a.heart:
        h, hres, hprog = heart()
        h.to_csv(RESULTS / 'agreement_heart.csv', index=False)
        progs.append(hprog)
        for zone, res in hres.items():
            print(zone, {m: int(r.called.sum()) if r.called.notna().all() else None for m, r in res.items()})
        print(h.round(2).to_string(index=False))
    pr = pd.concat(progs)
    pr.to_csv(RESULTS / 'agreement_programs.csv', index=False)
    print(pr.round(3).to_string(index=False))
