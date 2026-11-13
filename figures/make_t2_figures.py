"""Presentation figures for T2 SAM on the gut (adult A1, A2 vs fetal A3, A4, A8, A9).
  t2_map_<pair>.png    : local ligand x neighbouring receptor in every section, one row
  t2_maps_grid.png     : the four example pairs stacked
  t2_tool_diffs.png    : each tool's adult - fetal difference for example pairs
Reads t2_gut_results.csv (t2_gut.py); labels use the current T2 SAM calls (t_dir, q90_dir).
    python3 figures/make_t2_figures.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'pipeline'), str(ROOT / 'benchmark')]
import real_structured as rs  # noqa: E402

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRAY = '#0b0b0b', '#52514e', '#b4b2a9'
SEQ = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#b7d3f6', '#6da7ec', '#256abf', '#0d366b'])
plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 200, 'font.size': 11, 'axes.titlesize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': GRAY,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})

MAP_PAIRS = ['CDH1_CDH1', 'F11R_F11R', 'MDK_NCL', 'JAG1_NOTCH2']
DIFF_PAIRS = ['CDH1_CDH1', 'F11R_F11R', 'DSC2_DSG2', 'MDK_NCL', 'JAG1_NOTCH2', 'MDK_LRP1']


def local_scores(data, K, lr, pair):
    _, l, r, _ = next(p for p in lr if p[0] == pair)
    return {s: rs.gene_expr(X, l) * (K[s] @ rs.gene_expr(X, r)) for s, (X, _, _) in data.items()}


def map_row(axes, data, loc, pair, t2):
    vmax = np.percentile(np.concatenate(list(loc.values())), 99)
    for ax, s in zip(axes, data):
        xy, v = data[s][2], loc[s]
        o = np.argsort(v)
        ax.scatter(xy[o, 0], xy[o, 1], c=v[o], cmap=SEQ, vmin=0, vmax=vmax, s=3, linewidths=0)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
        cond = 'adult' if s in rs.G1 else 'fetal'
        ax.set_title(f'{s} ({cond})', color=BLUE if cond == 'adult' else ORANGE)
    r = t2.loc[pair]
    q = '< 0.001' if r.q90_dir < 0.001 else f'= {r.q90_dir:.3f}'
    axes[0].text(-0.08, 0.5, f'{pair.replace("_", "–", 1)}\nhigher in {r.higher_in_dir}\nq {q}',
                 transform=axes[0].transAxes, ha='right', va='center', color=INK, fontsize=11)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    t2 = pd.read_csv(ROOT / 'results' / 't2_gut_results.csv', index_col=0)
    lr = rs.lr_table()
    genes = sorted({g for _, l, r, _ in lr for g in l + r})
    data = {s: rs.load_sample(s, genes) for s in rs.G1 + rs.G2}
    K = {s: rs.kernel(xy) for s, (_, _, xy) in data.items()}
    locs = {p: local_scores(data, K, lr, p) for p in MAP_PAIRS}

    for p in MAP_PAIRS:                                  # one figure per pair (one slide build each)
        fig, axes = plt.subplots(1, 6, figsize=(14, 2.9))
        map_row(axes, data, locs[p], p, t2)
        fig.savefig(HERE / f't2_map_{p}.png'); plt.close(fig)

    fig, axes = plt.subplots(len(MAP_PAIRS), 6, figsize=(14, 2.7 * len(MAP_PAIRS)))
    for row, p in zip(axes, MAP_PAIRS):
        map_row(row, data, locs[p], p, t2)
    fig.tight_layout(h_pad=0.5)
    fig.suptitle('Local communication (ligand × receptor in neighbouring spots); darker = stronger',
                 color=INK2, y=1.01)
    fig.savefig(HERE / 't2_maps_grid.png'); plt.close(fig)

    d = t2.loc[DIFF_PAIRS, ['diff_spatialdm', 'diff_commot', 'diff_stlearn']]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(DIFF_PAIRS))
    for i, (col, lab, c) in enumerate(zip(d.columns, ['SpatialDM', 'COMMOT', 'stLearn'], [BLUE, ORANGE, AQUA])):
        ax.bar(x + (i - 1) * 0.26, d[col], 0.24, color=c, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(2.5, color=GRAY, lw=1, ls=':')
    ax.set_xticks(x, [p.replace('_', '–', 1) for p in DIFF_PAIRS])
    ax.set_ylabel('adult − fetal (standardised)')
    ax.text(2.4, ax.get_ylim()[1] * 0.92, 'higher in adult', ha='right', color=BLUE)
    ax.text(2.6, ax.get_ylim()[1] * 0.92, 'higher in fetal', ha='left', color=ORANGE)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.set_title('Each tool\'s difference between adult and fetal, for pairs called by T2 SAM')
    ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
    fig.savefig(HERE / 't2_tool_diffs.png'); plt.close(fig)
    print('written to', HERE)
