"""Score every differential method on every simulated cohort.

Methods (all given the same per-sample tool outputs):
  SpatialDM LRT        SpatialDM's own differential_test (OLS likelihood ratio on z)
  SAM SpatialDM        univariate SAM on global z
  SAM COMMOT           univariate SAM on log COMMOT totals
  SAM stLearn          univariate SAM on log stLearn scores, sample-normalised
  SAM stLearn (raw)    same without sample normalisation
  T2 SAM               multivariate SAM across the three tools
  Tensor A             label-blind decomposition, samples x LR x methods
  Tensor A (norm)      same with sample_norm=True
  Tensor B             label-blind decomposition, samples x LR x sender x receiver (COMMOT)

Metrics follow Ku et al. (2026): precision, recall, F1, specificity of calls at
q < 0.05 / 0.1 / 0.2 (a call counts only if its direction is right), plus
threshold-free AUROC, AUPRC and precision@K (K = number of planted differential
pairs). The tensors give no q-values, so they are scored on rankings only.
Tensor B is also scored on whether a condition-linked factor puts its largest
sender and receiver loadings on the planted cell types.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.multitest import fdrcorrection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from samdiff import multivariate as mv, sam, tensor as tc  # noqa: E402
import simulate as sim  # noqa: E402

THRESHOLDS = (0.05, 0.1, 0.2)
RANK = 3


# =========================================================================
# Inputs
# =========================================================================

def load_rep(scenario, rep):
    d = HERE / 'sim_data' / scenario / f'rep{rep}'
    o = HERE / 'tool_out' / scenario / f'rep{rep}'
    truth = pd.read_csv(d / 'truth.csv')
    samples = sorted(p.stem for p in o.glob('*.pkl'))
    res = {s: pickle.load(open(o / f'{s}.pkl', 'rb')) for s in samples}
    tabs = {m: pd.DataFrame({s: res[s][m] for s in samples}) for m in ('spatialdm', 'commot', 'stlearn')}
    slices = {s: res[s]['commot_ct'] for s in samples}
    import anndata as ad
    abund = pd.DataFrame({s: ad.read_h5ad(d / f'{s}.h5ad', backed='r').obs[sim.TYPES].sum()
                          for s in samples}).T
    g1 = [s for s in samples if s.startswith('A')]
    g2 = [s for s in samples if s.startswith('B')]
    return truth, tabs, slices, abund, g1, g2


# =========================================================================
# Methods: each returns DataFrame(index=pair, columns=[score, sign, q])
# =========================================================================

def spatialdm_lrt(z, g1, g2):
    """SpatialDM differential_test, reimplemented on the z table (missing -> 0,
    as concat_obj does). Matches the package on the project's real data."""
    y = z[g1 + g2].fillna(0).to_numpy()
    n = y.shape[1]
    A, B = y[:, :len(g1)], y[:, len(g1):]
    rss0 = ((y - y.mean(1, keepdims=True)) ** 2).sum(1)
    rss1 = ((A - A.mean(1, keepdims=True)) ** 2).sum(1) + ((B - B.mean(1, keepdims=True)) ** 2).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = chi2.sf(n * np.log(rss0 / rss1), 1)
    p = np.where(np.isnan(p), 1, p)
    return pd.DataFrame({'score': -np.log10(np.maximum(p, 1e-300)),
                         'sign': np.sign(A.mean(1) - B.mean(1)),
                         'q': fdrcorrection(p)[1]}, index=z.index)


def spatialdm_group(z, g1, g2, fdr_co=0.1, q1=0.7, q2=0.3):
    """SpatialDM group_differential_pairs: pairs selected (BH fdr < 0.1 on the
    one-sided z p-value) in every sample of one condition and none of the other,
    plus pairs with LRT fdr < fdr_co, z difference beyond the q1 / q2 quantiles and
    selected in some but not all samples. A call set, so q is 0 (called) or 1;
    the score is the LRT score."""
    from scipy.stats import norm
    cols = g1 + g2
    sel = pd.DataFrame({s: pd.Series(fdrcorrection(norm.sf(z[s].dropna()))[1] < 0.1,
                                     index=z[s].dropna().index)
                        for s in cols}).reindex(z.index).fillna(False).astype(bool)
    lrt = spatialdm_lrt(z, g1, g2)
    zf = z[cols].fillna(0)
    diff = zf[g1].mean(axis=1) - zf[g2].mean(axis=1)
    n_sel = sel.sum(axis=1)
    partial = n_sel.between(1, len(cols) - 1)
    a_only = sel[g1].all(axis=1) & ~sel[g2].any(axis=1)
    b_only = sel[g2].all(axis=1) & ~sel[g1].any(axis=1)
    a_spec = a_only | ((diff > diff.quantile(q1)) & (lrt['q'] < fdr_co) & partial)
    b_spec = b_only | ((diff < diff.quantile(q2)) & (lrt['q'] < fdr_co) & partial)
    called = a_spec | b_spec
    return pd.DataFrame({'score': lrt['score'],
                         'sign': np.where(a_spec, 1, np.where(b_spec, -1, lrt['sign'])),
                         'q': np.where(called, 0.0, 1.0)}, index=z.index)


def sam_method(tab, g1, g2):
    r = sam.sam(tab, g1, g2)
    return pd.DataFrame({'score': r['d_sam'].abs(), 'sign': np.sign(r['diff']), 'q': r['q']})


def t2_method(tabs, g1, g2):
    z = {m: (t - np.nanmean(t.values)) / np.nanstd(t.values) for m, t in tabs.items()}
    multi, pairs = mv.build_multivariate_table(z, g1 + g2)
    res, _, _ = mv.run_multivariate_sam(multi, pairs, g1, g2, len(z))
    diff = res[[c for c in res if c.startswith('diff_')]].mean(1)
    return pd.DataFrame({'score': res['T2'], 'sign': np.sign(diff), 'q': res['q']})


def tensor_delta(dec, g1, g2):
    """Label-blind low-rank estimate of each pair's group difference:
    sum_f w_f * (mean_A a_f - mean_B a_f) * b_lf * prod_other modes (sum c_f)."""
    A = dec.factors['Samples']
    delta = A.loc[g1].mean() - A.loc[g2].mean()
    other = np.ones(dec.rank)
    for name, F in dec.factors.items():
        if name not in ('Samples', 'LR pairs'):
            other *= F.sum().to_numpy()
    s = dec.factors['LR pairs'].to_numpy() @ (dec.weights * delta.to_numpy() * other)
    return pd.DataFrame({'score': np.abs(s), 'sign': np.sign(s), 'q': np.nan},
                        index=dec.factors['LR pairs'].index)


def tensor_b_recovery(dec, test):
    """For each planted differential program, is there a factor that separates the
    groups in the right direction and has its largest sender / receiver loadings
    on the planted cell types?"""
    out = {}
    for prog in ('A', 'B'):
        s_true, r_true = sim.PROGRAMS[prog]
        hits = [f for f in test.index
                if test.loc[f, 'separated'] and test.loc[f, 'higher_in'] == prog
                and dec.factors['Senders'][f].idxmax() == s_true
                and dec.factors['Receivers'][f].idxmax() == r_true]
        out[f'recovered_{prog}'] = bool(hits)
    out['n_separated'] = int(test['separated'].sum())
    return out


def run_methods(tabs, slices, abund, g1, g2):
    out = {'SpatialDM LRT': spatialdm_lrt(tabs['spatialdm'], g1, g2),
           'SAM SpatialDM': sam_method(tabs['spatialdm'], g1, g2),
           'SAM COMMOT': sam_method(tc.transform_table(tabs['commot']), g1, g2),
           'SAM stLearn': sam_method(tc.transform_table(tabs['stlearn'], sample_norm=True), g1, g2),
           'SAM stLearn (raw)': sam_method(tc.transform_table(tabs['stlearn']), g1, g2),
           'T2 SAM': t2_method(tabs, g1, g2)}
    extra = {}
    for label, norm in (('Tensor A', False), ('Tensor A (norm)', True)):
        dec = tc.decompose(tc.build_tensor_a(tabs, g1 + g2, sample_norm=norm), RANK, n_runs=5)
        out[label] = tensor_delta(dec, g1, g2)
        extra[label] = {'n_separated': int(tc.loading_test(dec, g1, g2, names=('A', 'B'))['separated'].sum())}
    ct_b = tc.tensor_b_from_slices(slices, abund, sim.TYPES)
    dec = tc.decompose(ct_b, RANK, n_runs=5)
    out['Tensor B'] = tensor_delta(dec, g1, g2)
    extra['Tensor B'] = tensor_b_recovery(dec, tc.loading_test(dec, g1, g2, names=('A', 'B')))
    return out, extra


# =========================================================================
# Metrics
# =========================================================================

def collateral(truth, universe):
    """Unplanted pairs that share a ligand or receptor gene with a planted
    differential pair. The simulator scales those genes in the program's cells, so
    these pairs change too; they are neither true positives nor clean negatives.
    Pair names are '<ligand>_<receptor>' (single-chain genes contain no '_')."""
    planted = truth.loc[truth.direction != 0, 'pair']
    lig = {p.split('_')[0] for p in planted}
    rec = {p.split('_')[1] for p in planted}
    return [p for p in universe if p not in set(planted)
            and (p.split('_')[0] in lig or p.split('_')[1] in rec)]


def score_method(res, truth, universe, exclude_collateral=False):
    """Calls and rankings scored over `universe` (pairs every tool tested).
    Planted pairs a method never tested count as missed / ranked last.
    exclude_collateral drops pairs sharing a perturbed gene with a planted pair."""
    if exclude_collateral:
        drop = set(collateral(truth, universe))
        universe = [p for p in universe if p not in drop]
    t = truth.set_index('pair')['direction'].reindex(universe).fillna(0)
    is_diff = (t != 0).to_numpy()
    r = res.reindex(universe)
    score = r['score'].fillna(-np.inf).to_numpy()
    sign = r['sign'].fillna(0).to_numpy()
    rows = {}
    n_pos = int(is_diff.sum())
    if n_pos:
        finite = np.where(np.isfinite(score), score, np.nanmin(score[np.isfinite(score)]) - 1)
        rows['AUROC'] = roc_auc_score(is_diff, finite)
        rows['AUPRC'] = average_precision_score(is_diff, finite)
        top = np.argsort(-finite)[:n_pos]
        rows['precision@K'] = np.mean((t.to_numpy()[top] != 0) & (sign[top] == t.to_numpy()[top]))
    if r['q'].notna().any():
        q = r['q'].fillna(1).to_numpy()
        for thr in THRESHOLDS:
            called = q < thr
            tp = int((called & is_diff & (sign == t.to_numpy())).sum())
            fp = int(called.sum() - tp)
            fn = n_pos - tp
            tn = int((~called & ~is_diff).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / n_pos if n_pos else np.nan
            rows[f'calls@{thr}'] = int(called.sum())
            rows[f'FP@{thr}'] = fp
            rows[f'precision@{thr}'] = prec
            rows[f'recall@{thr}'] = rec
            rows[f'F1@{thr}'] = (2 * prec * rec / (prec + rec)
                                 if prec == prec and rec == rec and prec + rec else 0.0)
            rows[f'specificity@{thr}'] = tn / (tn + fp) if tn + fp else np.nan
    return rows


def evaluate_all(reps=3):
    rows, extras = [], []
    for scenario in sim.SCENARIOS:
        for rep in range(reps):
            o = HERE / 'tool_out' / scenario / f'rep{rep}'
            if not o.exists():
                continue
            truth, tabs, slices, abund, g1, g2 = load_rep(scenario, rep)
            if len(g1) + len(g2) < sim.SCENARIOS[scenario].n1 + sim.SCENARIOS[scenario].n2:
                continue
            universe = sorted(set.intersection(*[set(t.dropna(how='all').index) for t in tabs.values()]))
            res, extra = run_methods(tabs, slices, abund, g1, g2)
            for m, r in res.items():
                rows.append({'scenario': scenario, 'rep': rep, 'method': m,
                             'n_universe': len(universe), **score_method(r, truth, universe)})
            for m, e in extra.items():
                extras.append({'scenario': scenario, 'rep': rep, 'method': m, **e})
            print(scenario, rep, 'done', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(extras)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    metrics, extras = evaluate_all()
    metrics.to_csv(HERE / 'metrics.csv', index=False)
    extras.to_csv(HERE / 'tensor_extras.csv', index=False)
    print(metrics.groupby(['scenario', 'method'])[['AUPRC', 'precision@K', 'F1@0.1', 'FP@0.1']]
          .mean().round(3).to_string())
