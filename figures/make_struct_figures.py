"""Presentation figures for the structured differential method, multi-condition version
(structured.multi_rank via mi_multi.py), on the myocardial infarction data, plus the
three-condition simulation benchmark (benchmark/metrics_multi.csv).
  struct_zone_heatmap.png : denoised effect per zone for the top 20 pairs
  struct_components.png   : zone loadings of the two CP components
  struct_map_<pair>.png   : local ligand x neighbouring receptor, two patients per zone
  struct_pair_<pair>.png  : the pair in every patient (top) and in the tissue (bottom)
  struct_bench.png        : AUPRC and false share of calls, three-condition benchmark
    python3 figures/make_struct_figures.py [--pairs-only]
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

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRAY = '#0b0b0b', '#52514e', '#b4b2a9'
ZONE_C = {'control': BLUE, 'RZ': AQUA, 'IZ': ORANGE, 'FZ': YELLOW}
ZONE_NAME = {'control': 'control', 'RZ': 'remote', 'IZ': 'ischemic', 'FZ': 'fibrotic'}
SEQ = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#b7d3f6', '#6da7ec', '#256abf', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', [BLUE, '#f0efec', ORANGE])
plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 200, 'font.size': 11, 'axes.titlesize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': GRAY,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})
ZONES = ['control', 'RZ', 'IZ', 'FZ']
MAP_PAIRS = ['SPP1_ITGA5_ITGB1', 'VEGFB_VEGFR1']
SHOW = {'control': ['control_P1', 'control_P7'], 'RZ': ['RZ_P6', 'RZ_P9'],
        'IZ': ['IZ_P3', 'IZ_P16'], 'FZ': ['FZ_P14', 'FZ_P18']}


def zone_heatmap(res):
    top = res.head(20)
    E = top[[f'effect_{z}' for z in ZONES]].to_numpy()
    m = np.abs(E).max()
    fig, ax = plt.subplots(figsize=(6.2, 7.2))
    im = ax.imshow(E, cmap=DIV, vmin=-m, vmax=m, aspect='auto')
    ax.set_xticks(range(4), [ZONE_NAME[z] for z in ZONES])
    ax.set_yticks(range(len(top)), [p.replace('_', '–', 1) for p in top.index], fontsize=9)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label('effect vs average of all zones')
    ax.set_title('Top 20 pairs: change per zone\n(orange = higher, blue = lower than average)')
    fig.savefig(HERE / 'struct_zone_heatmap.png'); plt.close(fig)


def components(comp):
    fig, axes = plt.subplots(1, len(comp), figsize=(5 * len(comp), 3.3), sharey=True)
    titles = {1: 'Component 1: ischemic-zone specific', 2: 'Component 2: healthy vs injured'}
    for ax, (_, r) in zip(np.atleast_1d(axes), comp.iterrows()):
        v = [r[f'zone_{z}'] for z in ZONES]
        ax.bar([ZONE_NAME[z] for z in ZONES], v, color=[ZONE_C[z] for z in ZONES], width=0.6)
        ax.axhline(0, color=INK2, lw=0.8)
        for i, x in enumerate(v):
            ax.text(i, x + (0.04 if x >= 0 else -0.1), f'{x:.2f}', ha='center', color=INK2, fontsize=10)
        ax.set_title(titles.get(int(r.component), f'Component {int(r.component)}'))
        ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
        ax.set_ylim(-0.7, 1.0)
    np.atleast_1d(axes)[0].set_ylabel('zone loading')
    fig.savefig(HERE / 'struct_components.png'); plt.close(fig)


def maps(res):
    import mi_structured as mis
    import real_structured as rs
    lr = rs.lr_table()
    samples = [s for z in ZONES for s in SHOW[z]]
    data = {s: mis.load(s) for s in samples}
    for pair in MAP_PAIRS:
        _, l, r, _ = next(p for p in lr if p[0] == pair)
        loc = {}
        for s, (X, P, xy, nn, _) in data.items():
            K = rs.kernel(xy, mis.L_SCALE * nn)
            loc[s] = rs.gene_expr(X, l) * (K @ rs.gene_expr(X, r))
        vmax = np.percentile(np.concatenate(list(loc.values())), 99)
        fig, axes = plt.subplots(1, len(samples), figsize=(17, 2.9))
        for ax, s in zip(axes, samples):
            xy, v = data[s][2], loc[s]
            o = np.argsort(v)
            ax.scatter(xy[o, 0], xy[o, 1], c=v[o], cmap=SEQ, vmin=0, vmax=vmax, s=1.2, linewidths=0)
            ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
            z = s.split('_')[0]
            ax.set_title(f'{ZONE_NAME[z]}\n{s.split("_")[1]}', color=ZONE_C[z], fontsize=10)
        row = res.loc[pair]
        q = '< 0.001' if row.q90 < 0.001 else f'= {row.q90:.3f}'
        axes[0].text(-0.1, 0.5, f'{pair.replace("_", "–", 1)}\nhighest: {ZONE_NAME[row.highest_in]}\n'
                     f'lowest: {ZONE_NAME[row.lowest_in]}\nq {q}', transform=axes[0].transAxes,
                     ha='right', va='center', color=INK, fontsize=10)
        fig.savefig(HERE / f'struct_map_{pair}.png'); plt.close(fig)


def pair_panel(res):
    """Per pair: mean communication per spot in every patient (top) and maps of two
    patients per zone (bottom); the value mapped is ligand x neighbouring receptor."""
    import mi_structured as mis
    import real_structured as rs
    from mi_multi import GROUPS
    lr = rs.lr_table()
    samples = [s for z in ZONES for s in GROUPS[z]]
    data = {s: mis.load(s) for s in samples}
    shown = [s for z in ZONES for s in SHOW[z]]
    rng = np.random.default_rng(0)
    for pair in MAP_PAIRS:
        _, l, r, _ = next(p for p in lr if p[0] == pair)
        loc = {}
        for s, (X, P, xy, nn, _) in data.items():
            K = rs.kernel(xy, mis.L_SCALE * nn)
            loc[s] = rs.gene_expr(X, l) * (K @ rs.gene_expr(X, r))
        vmax = np.percentile(np.concatenate([loc[s] for s in shown]), 99)
        fig = plt.figure(figsize=(17, 6.4))
        gs = fig.add_gridspec(2, len(shown), height_ratios=[1, 1.15], hspace=0.35)
        top = fig.add_subplot(gs[0, 1:len(shown) - 1])
        for i, z in enumerate(ZONES):
            v = np.array([loc[s].mean() for s in GROUPS[z]])
            top.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=70, color=ZONE_C[z],
                        edgecolor='white', lw=0.8, zorder=3)
            top.hlines(v.mean(), i - 0.28, i + 0.28, color=INK, lw=1.6)
        top.set_xticks(range(4), [f'{ZONE_NAME[z]}\n({len(GROUPS[z])} patients)' for z in ZONES])
        top.set_ylabel('mean per spot')
        row = res.loc[pair]
        q = '< 0.001' if row.q90 < 0.001 else f'= {row.q90:.3f}'
        top.set_title(f'{pair.replace("_", "–", 1).replace("_", "/")}: every patient  (q {q}; highest '
                      f'{ZONE_NAME[row.highest_in]}, lowest {ZONE_NAME[row.lowest_in]})')
        top.grid(axis='y', color='#e6e5e0', lw=0.6); top.set_axisbelow(True)
        top.set_ylim(bottom=0)
        for j, s in enumerate(shown):
            ax = fig.add_subplot(gs[1, j])
            xy, v = data[s][2], loc[s]
            o = np.argsort(v)
            ax.scatter(xy[o, 0], xy[o, 1], c=v[o], cmap=SEQ, vmin=0, vmax=vmax, s=1.2, linewidths=0)
            ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
            z = s.split('_')[0]
            ax.set_title(f'{ZONE_NAME[z]}\n{s.split("_")[1]}', color=ZONE_C[z], fontsize=10)
        fig.savefig(HERE / f'struct_pair_{pair}.png'); plt.close(fig)


def bench():
    m = pd.read_csv(ROOT / 'benchmark' / 'metrics_multi.csv')
    m = m[(m['eval'] == 'excl') & (m.scenario != 'multi_null')]
    names = {'multi cp r2': 'Structured (multi-condition)', 'pairwise v2': 'Pairwise two-group runs',
             'multiclass SAM': 'Multiclass SAM (no cell types)'}
    m = m[m.method.isin(names)].assign(method=lambda d: d.method.map(names))
    scen = {'multi_fold4': '4× change', 'multi_fold2': '2× change',
            'multi_comp': 'composition\ndiffers', 'multi_4each': '4 samples\nper group'}
    g = m.groupby(['method', 'scenario'])
    auprc = g['AUPRC'].mean().unstack()[list(scen)]
    fs = (g['FP@0.1'].sum() / g['calls@0.1'].sum()).unstack()[list(scen)]
    colors = dict(zip(names.values(), [BLUE, AQUA, YELLOW]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tab, title, yl in [(axes[0], auprc, 'Ranking of true changes (AUPRC)', 'AUPRC'),
                               (axes[1], fs, 'Share of calls that are false (q < 0.1)', 'false share')]:
        x = np.arange(len(scen))
        for i, meth in enumerate(names.values()):
            ax.bar(x + (i - 1) * 0.26, tab.loc[meth], 0.24, color=colors[meth], label=meth)
        ax.set_xticks(x, list(scen.values())); ax.set_ylim(0, 1); ax.set_ylabel(yl); ax.set_title(title)
        ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
    axes[1].axhline(0.1, color=INK2, lw=1, ls='--')
    axes[0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.18), ncol=3)
    fig.suptitle('Simulation: three conditions, 30 cohorts per scenario', color=INK2)
    fig.savefig(HERE / 'struct_bench.png'); plt.close(fig)
    return auprc, fs


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    res = pd.read_csv(ROOT / 'results' / 'mi_multi_pairs.csv', index_col=0)
    comp = pd.read_csv(ROOT / 'results' / 'mi_multi_components.csv')
    pair_panel(res)
    if '--pairs-only' in sys.argv:
        sys.exit(0)
    zone_heatmap(res)
    components(comp)
    maps(res)
    a, f = bench()
    print(a.round(2).to_string()); print(f.round(2).to_string())
    print('written to', HERE)
