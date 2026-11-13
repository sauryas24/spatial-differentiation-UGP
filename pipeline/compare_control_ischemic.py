"""Heart, control (4) vs ischemic zone (4): all four methods on the 126 pairs scored in
every section, for the comparison section of the report (Tables 13 and 14).
  SpatialDM  : its differential test on the per-section z-scores
  T2 SAM     : as in agreement.py
  Structured : two-condition run (mi_structured.analyse) on the same 126 pairs
  Tensor     : programs of the 14-sample decomposition; each program's sample loadings
               are compared between control and ischemic sections (exact relabelling);
               its flagged pairs are the top 20 pairs of every program that differs
               (p <= 0.05), and pairs in programs of both conditions are dropped
    python3 pipeline/compare_control_ischemic.py
Writes results/mi_structured_IZ_126.csv.
"""
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS = HERE / 'results'
sys.path[:0] = [str(HERE), str(HERE / 'benchmark')]
import agreement as ag  # noqa: E402
import mi_structured as mis  # noqa: E402
from real_structured import lr_table  # noqa: E402

CONTROL = ['control_P1', 'control_P7', 'control_P8', 'control_P17']
ISCHEMIC = ['IZ_P3', 'IZ_P10', 'IZ_P15', 'IZ_P16']
TOP = 20


def tensor_scores(g1, g2):
    """Reconstructed control - ischemic difference per pair, and per-program tests."""
    d = RESULTS / 'tc2c' / 'mi'
    C = pd.read_csv(d / 'loadings_contexts.csv', index_col=0)
    B = pd.read_csv(d / 'loadings_ligand-receptor.csv', index_col=0)
    S = pd.read_csv(d / 'loadings_sender.csv', index_col=0)
    R = pd.read_csv(d / 'loadings_receiver.csv', index_col=0)
    w = pd.read_csv(d / 'condition_tests.csv', index_col=0)['weight']
    dc = C.loc[g1].mean() - C.loc[g2].mean()
    score = pd.DataFrame({'score': B @ (w * dc * S.sum() * R.sum()), 'called': np.nan})
    n, n1 = len(g1 + g2), len(g1)
    labs = list(combinations(range(n), n1))
    programs = {}
    for f in C:
        v = C.loc[g1 + g2, f].to_numpy()
        obs = v[:n1].mean() - v[n1:].mean()
        null = np.array([v[list(l)].mean() - np.delete(v, list(l)).mean() for l in labs])
        programs[f] = ('control' if obs > 0 else 'IZ', np.mean(np.abs(null) >= abs(obs) - 1e-12))
    direction = {}
    for f, (cond, p) in programs.items():
        if p <= 0.05:
            for pair in B[f].nlargest(TOP).index:
                direction.setdefault(pair, set()).add(1 if cond == 'control' else -1)
    flagged = {p: next(iter(v)) for p, v in direction.items() if len(v) == 1}
    return score, programs, flagged, sum(len(v) > 1 for v in direction.values())


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    names = json.load(open(RESULTS / 'tc2c' / 'mi_meta.json'))['pairs']
    tabs = {m: pd.read_csv(RESULTS / f'mi_tool_results_{m}.csv', index_col=0).reindex(names)
            for m in ('commot', 'spatialdm', 'stlearn')}
    data = {s: mis.load(s) for s in CONTROL + ISCHEMIC}
    n_spots = pd.Series({s: len(data[s][2]) for s in CONTROL + ISCHEMIC})

    res = {'SpatialDM': ag.spatialdm(tabs['spatialdm'], CONTROL, ISCHEMIC),
           'T2 SAM': ag.t2(tabs, CONTROL, ISCHEMIC, n_spots)}
    s, *_ = mis.analyse(data, CONTROL, ISCHEMIC, lr_table(), names=list(names))
    s.to_csv(RESULTS / 'mi_structured_IZ_126.csv')
    res['Structured'] = pd.DataFrame({'score': np.where(s.higher_in == 'control', 1, -1) * s.score,
                                      'called': s.q90 < 0.1}, index=s.index)
    res['Tensor'], programs, flagged, dropped = tensor_scores(CONTROL, ISCHEMIC)
    print('programs, control vs ischemic (condition, relabelling p):', programs)

    common = sorted(set.intersection(*[set(r.index) for r in res.values()]))
    print(f'{len(common)} common pairs')
    for m in ('SpatialDM', 'T2 SAM', 'Structured'):
        r = res[m].loc[common]
        c = r.called.astype(bool)
        print(f'{m}: flagged {c.sum()}, higher in control {(c & (r.score > 0)).sum()}, '
              f'higher in ischemic {(c & (r.score < 0)).sum()}')
    print(f'Tensor: flagged {len(flagged)}, control {sum(v == 1 for v in flagged.values())}, '
          f'ischemic {sum(v == -1 for v in flagged.values())}, dropped {dropped}')

    rows = []
    for x, y in combinations(res, 2):
        rx, ry = res[x].loc[common], res[y].loc[common]
        cx = set(rx.index[rx.called.astype(bool)])
        if y == 'Tensor':
            cy, sign_y = set(flagged) & set(common), pd.Series(flagged)
        else:
            cy, sign_y = set(ry.index[ry.called.astype(bool)]), np.sign(ry.score)
        both = sorted(cx & cy)
        rows.append({'methods': f'{x} - {y}', 'spearman': round(spearmanr(rx.score, ry.score)[0], 2),
                     'shared_flagged': len(both), 'expected': round(len(cx) * len(cy) / len(common), 1),
                     'same_direction': round(np.mean([np.sign(rx.score[p]) == sign_y[p] for p in both]), 2)
                     if both else np.nan})
    print(pd.DataFrame(rows).to_string(index=False))
