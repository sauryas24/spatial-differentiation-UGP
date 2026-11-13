"""Tensor-cell2cell-style analysis of spatial communication tensors, written from the
protocol of Baghdassarian et al. (2024, Cell Reports Methods) without the cell2cell
package.

Steps (protocol section in brackets):
1. Rank selection by elbow analysis of the normalised reconstruction error of
   non-negative CP decompositions, rank 1..R, several random starts each (Step 5).
2. Non-negative CP decomposition at the chosen rank, best of several random starts;
   every factor is a communication program with four loading vectors (contexts,
   LR pairs, sender cells, receiver cells), each scaled to unit norm (Step 5).
3. Context loadings compared between condition groups: independent t-tests with
   Benjamini-Hochberg correction as in the protocol's Figure 6, plus an exact
   relabelling test because the groups are small (Downstream visualizations).
4. Pathway enrichment of each factor's LR loadings: GSEA prerank on KEGG LR sets (an
   LR pair belongs to a pathway if all its genes do) and PROGENy footprints with a
   multivariate linear model, LR weights as in LIANA (Step 6).
Input: tc2c/<dataset>_tensor.npz and _meta.json from tc2c_build.py.
    python3 tc2c_analysis.py --dataset gut [--rank N] [--upper 10]
Writes tc2c/<dataset>/: loadings_*.csv, condition_tests.csv, gsea_kegg.csv,
progeny.csv and figures.
"""
import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
OUT = RESULTS / 'tc2c'
MODES = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']


# ---------------------------------------------------------------- decomposition

def nn_cp(X, rank, runs=10, seed=0, n_iter_max=500, tol=1e-7):
    """Best of `runs` random-start non-negative CP fits: (weights, factors, error)."""
    import tensorly as tl
    from tensorly.decomposition import non_negative_parafac
    best = None
    for r in range(runs):
        cp = non_negative_parafac(tl.tensor(X), rank, init='random', n_iter_max=n_iter_max,
                                  tol=tol, random_state=seed + r)
        err = np.linalg.norm(X - tl.cp_to_tensor(cp)) / np.linalg.norm(X)
        if best is None or err < best[2]:
            best = (cp.weights, cp.factors, err)
    return best


def normalise(weights, factors):
    """Unit-norm loading vectors; the norms go into the weights; factors ordered by weight."""
    norms = np.stack([np.linalg.norm(F, axis=0) for F in factors])
    w = weights * norms.prod(0)
    F = [f / np.where(n > 0, n, 1) for f, n in zip(factors, norms)]
    order = np.argsort(-w)
    return w[order], [f[:, order] for f in F]


def stability(X, rank, F_ref, n_rep=5, runs=10):
    """Refit from independent random starts; for each factor, the best match (product
    of absolute cosines over the four modes) to the reference factors. 1 = identical."""
    out = []
    for rep in range(n_rep):
        _, F2 = normalise(*nn_cp(X, rank, runs, seed=1000 * (rep + 1))[:2])
        sim = np.ones((rank, rank))
        for a, b in zip(F_ref, F2):
            sim *= np.abs(a.T @ b)
        out.append(sim.max(1))
    return np.array(out)


def elbow(X, upper, runs=5, seed=0):
    """Mean and sd of the normalised error per rank, and the automatic elbow: the rank
    furthest from the line joining the first and last points (kneedle)."""
    errs = np.array([[nn_cp(X, k, 1, seed + 100 * k + r, n_iter_max=200)[2] for r in range(runs)]
                     for k in range(1, upper + 1)])
    m = errs.mean(1)
    x = np.linspace(0, 1, upper)
    y = (m - m.min()) / max(m.max() - m.min(), 1e-12)
    dist = (1 - x) - y                      # distance below the straight line from (0,1) to (1,0)
    return pd.DataFrame({'rank': range(1, upper + 1), 'error': m, 'sd': errs.std(1)}), int(np.argmax(dist) + 1)


# ---------------------------------------------------------------- condition tests

def condition_tests(C, cond):
    """Per factor: mean context loading per condition, t-tests between every pair of
    conditions (BH over all comparisons, as in the protocol), and an exact relabelling
    test of the overall group difference (between-group sum of squares)."""
    groups = list(dict.fromkeys(cond[c] for c in C.index))
    lab = np.array([groups.index(cond[c]) for c in C.index])
    rows, pvals = [], []

    def between_ss(v, l):
        return sum((l == g).sum() * (v[l == g].mean() - v.mean()) ** 2 for g in range(len(groups)))

    from itertools import permutations
    labs = {tuple(lab[list(p)]) for p in permutations(range(len(lab)))} if len(lab) <= 9 else None
    rng = np.random.default_rng(0)
    for f in C.columns:
        v = C[f].to_numpy()
        row = {'factor': f, **{f'mean_{g}': v[lab == g].mean() for g in range(len(groups))}}
        obs = between_ss(v, lab)
        null = ([between_ss(v, np.array(l)) for l in labs] if labs is not None else
                [between_ss(v, rng.permutation(lab)) for _ in range(9999)])
        row['p_relabel'] = (np.sum(np.array(null) >= obs - 1e-12) + (labs is None)) / (len(null) + (labs is None))
        for a, b in combinations(range(len(groups)), 2):
            t, p = stats.ttest_ind(v[lab == a], v[lab == b])
            row[f't_{groups[a]}_vs_{groups[b]}'] = t
            row[f'p_{groups[a]}_vs_{groups[b]}'] = p
            pvals.append((f, f'{groups[a]}_vs_{groups[b]}', p))
        row['highest_in'] = groups[int(np.argmax([v[lab == g].mean() for g in range(len(groups))]))]
        rows.append(row)
    out = pd.DataFrame(rows).set_index('factor').rename(columns={f'mean_{i}': f'mean_{g}' for i, g in enumerate(groups)})
    q = stats.false_discovery_control([p for *_, p in pvals])
    for (f, cmp, _), qq in zip(pvals, q):
        out.loc[f, f'q_{cmp}'] = qq
    return out, groups


# ---------------------------------------------------------------- enrichment

def kegg_lr_sets(lr_list, genes_of, path=RESOURCES / 'genesets' / 'KEGG_2021_Human.txt', min_size=5):
    """KEGG pathway -> LR pairs whose genes all belong to the pathway."""
    sets = {}
    for line in open(path):
        name, _, *genes = line.rstrip('\n').split('\t')
        g = set(genes)
        members = [p for p in lr_list if set(genes_of[p]) <= g]
        if len(members) >= min_size:
            sets[name] = members
    return sets


def gsea_prerank(scores, sets, n_perm=999, weight=1.0, seed=0):
    """GSEA (Subramanian et al. 2005) on a ranked list: weighted running-sum ES, null
    from random sets of the same size, NES = ES / mean |null ES| of the same sign,
    BH-adjusted p-values over the sets."""
    s = scores.sort_values(ascending=False)
    r = np.abs(s.to_numpy()) ** weight
    idx = {p: i for i, p in enumerate(s.index)}
    N, rng = len(s), np.random.default_rng(seed)

    def es(hit_idx):
        hit = np.zeros(N, bool)
        hit[hit_idx] = True
        ph = np.cumsum(np.where(hit, r, 0)) / r[hit].sum()
        pm = np.cumsum(~hit) / (N - hit.sum())
        d = ph - pm
        return d[np.argmax(np.abs(d))]

    rows = []
    for name, members in sets.items():
        obs = es([idx[m] for m in members])
        null = np.array([es(rng.choice(N, len(members), replace=False)) for _ in range(n_perm)])
        same = null[np.sign(null) == np.sign(obs)]
        nes = obs / np.abs(same).mean() if len(same) else np.nan
        p = (np.sum(np.abs(same) >= abs(obs)) + 1) / (len(same) + 1)
        rows.append({'pathway': name, 'size': len(members), 'ES': obs, 'NES': nes, 'p': p,
                     'members_by_rank': ', '.join(m for m in s.index if m in members)})
    out = pd.DataFrame(rows)
    out['q'] = stats.false_discovery_control(out.p) if len(out) else []
    return out.sort_values('p')


def progeny_lr_weights(lr_list, ann, top=5000, min_lr=5, path=RESOURCES / 'genesets' / 'progeny_raw.csv'):
    """PROGENy gene weights -> LR weights as in LIANA's generate_lr_geneset: a complex
    takes the mean weight of its subunits, the LR weight is the mean of ligand and
    receptor, kept only if every protein is in the pathway and all weights share a sign.
    Pathways with fewer than min_lr weighted pairs are dropped (no fit possible)."""
    raw = pd.read_csv(path)
    wide = raw.pivot_table(index=['record_id', 'genesymbol'], columns='label', values='value',
                           aggfunc='first').reset_index()
    wide['weight'] = wide['weight'].astype(float)
    wide['p_value'] = wide['p_value'].astype(float)
    wide = wide.sort_values('p_value').groupby('pathway').head(top)
    W = {}
    for pw, d in wide.groupby('pathway'):
        g = dict(zip(d.genesymbol, d.weight))
        col = {}
        for p in lr_list:
            L, R = ann['ligand'][p], ann['receptor'][p]
            if all(x in g for x in L + R):
                vals = np.array([g[x] for x in L + R])
                if np.all(vals > 0) or np.all(vals < 0):
                    col[p] = (np.mean([g[x] for x in L]) + np.mean([g[x] for x in R])) / 2
        W[pw] = col
    W = pd.DataFrame(W).reindex(lr_list).fillna(0.0)
    return W.loc[:, (W != 0).sum() >= min_lr]


def mlm(loadings, W):
    """decoupler-style multivariate linear model: for each factor, regress the LR
    loadings on all pathway weights (plus intercept); t-values and p-values per pathway."""
    lrs = W.index[(W != 0).any(axis=1)]
    X = np.column_stack([np.ones(len(lrs)), W.loc[lrs].to_numpy()])
    dof = len(lrs) - X.shape[1]
    t_out, p_out = {}, {}
    for f in loadings.columns:
        y = loadings.loc[lrs, f].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        s2 = res @ res / dof
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
        t = beta / se
        t_out[f] = t[1:]
        p_out[f] = 2 * stats.t.sf(np.abs(t[1:]), dof)
    return (pd.DataFrame(t_out, index=W.columns), pd.DataFrame(p_out, index=W.columns), len(lrs))


# ---------------------------------------------------------------- driver

def run(dataset, rank=None, upper=10, runs=10):
    X = np.load(OUT / f'{dataset}_tensor.npz')['tensor']
    meta = json.load(open(OUT / f'{dataset}_meta.json'))
    out = OUT / dataset
    out.mkdir(exist_ok=True)
    el, auto = elbow(X, upper)
    rank = rank or auto
    el.to_csv(out / 'elbow.csv', index=False)
    weights, factors, err = nn_cp(X, rank, runs)
    w, F = normalise(weights, factors)
    names = [meta['contexts'], meta['pairs'], meta['cells'], meta['cells']]
    cols = [f'Factor {i + 1}' for i in range(rank)]
    L = {m: pd.DataFrame(f, index=n, columns=cols) for m, f, n in zip(MODES, F, names)}
    for m, d in L.items():
        d.to_csv(out / f'loadings_{m.split()[0].lower()}.csv')
    tests, groups = condition_tests(L['Contexts'], meta['condition'])
    tests.insert(0, 'weight', w)
    stab = stability(X, rank, F)
    tests.insert(1, 'stability_min', stab.min(0))
    tests.to_csv(out / 'condition_tests.csv')
    import sys
    sys.path.insert(0, str(HERE))
    from real_structured import lr_table
    lr = {p[0]: p for p in lr_table()}
    ann = {'ligand': {p: lr[p][1] for p in meta['pairs']}, 'receptor': {p: lr[p][2] for p in meta['pairs']}}
    genes_of = {p: ann['ligand'][p] + ann['receptor'][p] for p in meta['pairs']}
    sets = kegg_lr_sets(meta['pairs'], genes_of)
    gsea = pd.concat([gsea_prerank(L['Ligand-Receptor Pairs'][f], sets).assign(factor=f) for f in cols])
    gsea.to_csv(out / 'gsea_kegg.csv', index=False)
    Wp = progeny_lr_weights(meta['pairs'], ann)
    t, p, n_lr = mlm(L['Ligand-Receptor Pairs'], Wp)
    t.to_csv(out / 'progeny_t.csv')
    p.to_csv(out / 'progeny_p.csv')
    return dict(X=X, meta=meta, elbow=el, auto_rank=auto, rank=rank, error=err, weights=w, L=L,
                tests=tests, groups=groups, kegg_sets=sets, gsea=gsea, progeny_t=t, progeny_p=p,
                progeny_n_lr=n_lr)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='gut')
    ap.add_argument('--rank', type=int, default=None)
    ap.add_argument('--upper', type=int, default=10)
    a = ap.parse_args()
    R = run(a.dataset, a.rank, a.upper)
    pd.set_option('display.width', 250)
    print(f'{a.dataset}: tensor {R["X"].shape}, elbow rank {R["auto_rank"]}, used {R["rank"]}, error {R["error"]:.3f}')
    print(R['tests'].round(3).to_string())
    print(f'KEGG LR sets with >= 5 pairs: {len(R["kegg_sets"])}')
    g = R['gsea']
    print(g[g.q < 0.25].sort_values(['factor', 'p'])[['factor', 'pathway', 'size', 'NES', 'p', 'q']].round(3).to_string())
    print(f'PROGENy: {R["progeny_n_lr"]} LR pairs with weights')
    print(R['progeny_t'].round(2).to_string())
