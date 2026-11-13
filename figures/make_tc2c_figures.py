"""Presentation figures for the Tensor-cell2cell-style analysis of the myocardial
infarction data, from the saved outputs of tc2c_analysis.py (tc2c/mi/).
  tc2c_elbow.png        : elbow analysis (protocol Figure 5A)
  tc2c_factors.png      : loadings of every factor in every dimension (Figure 5B)
  tc2c_context_boxes.png: sample loadings by zone (Figure 6)
  tc2c_program_<f>.png  : top LR pairs and sender x receiver map of one factor
  tc2c_kegg.png         : KEGG GSEA per factor (Figure 7)
  tc2c_progeny.png      : PROGENy footprints per factor (Figure S3A)
    python3 figures/make_tc2c_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
D = ROOT / 'results' / 'tc2c' / 'mi'

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRAY = '#0b0b0b', '#52514e', '#b4b2a9'
ZONE_C = {'control': BLUE, 'RZ': AQUA, 'IZ': ORANGE, 'FZ': YELLOW}
ZONE_NAME = {'control': 'control', 'RZ': 'remote', 'IZ': 'ischemic', 'FZ': 'fibrotic'}
ZONES = list(ZONE_C)
SEQ = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#b7d3f6', '#6da7ec', '#256abf', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', [BLUE, '#f0efec', ORANGE])
plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 200, 'font.size': 11, 'axes.titlesize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': GRAY,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})
LABEL = {'Factor 1': 'healthy tissue', 'Factor 2': 'ischemic zone', 'Factor 3': 'healthy tissue',
         'Factor 4': 'healthy tissue', 'Factor 5': 'fibrotic zone'}


def load():
    meta = json.load(open(ROOT / 'results' / 'tc2c' / 'mi_meta.json'))
    L = {k: pd.read_csv(D / f'loadings_{k}.csv', index_col=0)
         for k in ('contexts', 'ligand-receptor', 'sender', 'receiver')}
    return meta, L, pd.read_csv(D / 'condition_tests.csv', index_col=0)


def elbow():
    e = pd.read_csv(D / 'elbow.csv')
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(e['rank'], e['error'], yerr=e['sd'], color=BLUE, marker='o', ms=5, lw=2, capsize=2)
    ax.scatter([5], e.set_index('rank').loc[5, 'error'], s=110, color=ORANGE, zorder=3, label='chosen: 5 factors')
    ax.set_xlabel('number of factors'); ax.set_ylabel('reconstruction error'); ax.legend()
    ax.grid(color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
    fig.savefig(HERE / 'tc2c_elbow.png'); plt.close(fig)


def factors(meta, L):
    cond = meta['condition']
    cols = list(L['contexts'].columns)
    order = sorted(L['contexts'].index, key=lambda s: ZONES.index(cond[s]))
    fig, axes = plt.subplots(len(cols), 4, figsize=(15, 1.7 * len(cols) + 0.8),
                             gridspec_kw={'width_ratios': [1.4, 2.6, 1.1, 1.1]})
    for i, f in enumerate(cols):
        for j, (k, title) in enumerate([('contexts', 'Samples'), ('ligand-receptor', 'LR pairs'),
                                        ('sender', 'Sender cells'), ('receiver', 'Receiver cells')]):
            ax = axes[i, j]
            v = L[k][f].reindex(order) if k == 'contexts' else L[k][f]
            c = [ZONE_C[cond[s]] for s in v.index] if k == 'contexts' else (GRAY if j == 1 else INK2)
            ax.bar(range(len(v)), v.values, color=c, width=1.0 if j == 1 else 0.8)
            ax.set_xticks([] if (j == 1 or i < len(cols) - 1) else range(len(v)))
            if i == len(cols) - 1 and j != 1:
                ax.set_xticklabels([s.split('_')[1] if k == 'contexts' else s for s in v.index],
                                   rotation=90, fontsize=8)
            ax.tick_params(axis='y', labelsize=8); ax.grid(False)
            if i == 0:
                ax.set_title(title)
        axes[i, 0].set_ylabel(f'{f}\n({LABEL[f]})', rotation=0, ha='right', va='center', color=INK, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ZONE_C[z]) for z in ZONES]
    fig.legend(handles, [ZONE_NAME[z] for z in ZONES], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(HERE / 'tc2c_factors.png'); plt.close(fig)


def context_boxes(meta, L, T):
    cond, C = meta['condition'], L['contexts']
    fig, axes = plt.subplots(1, len(C.columns), figsize=(3 * len(C.columns), 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, f in zip(axes, C.columns):
        data = [C[f][[cond[s] == z for s in C.index]].values for z in ZONES]
        b = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, z in zip(b['boxes'], ZONES):
            patch.set_facecolor(ZONE_C[z]); patch.set_alpha(0.3); patch.set_edgecolor(ZONE_C[z])
        for med in b['medians']:
            med.set_color(INK)
        for i, (d, z) in enumerate(zip(data, ZONES)):
            ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(d)), d, s=26, color=ZONE_C[z],
                       edgecolor='white', lw=0.6, zorder=3)
        ax.set_xticks(range(1, 5), [ZONE_NAME[z] for z in ZONES], rotation=35)
        p = T.loc[f, 'p_relabel']
        ax.set_title(f'{f}: {LABEL[f]}\np ' + ('< 0.001' if p < 0.001 else f'= {p:.3f}'), fontsize=10)
        ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel('sample loading')
    fig.savefig(HERE / 'tc2c_context_boxes.png'); plt.close(fig)


def program(L, f, n=10):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1.3, 1]})
    top = L['ligand-receptor'][f].nlargest(n)[::-1]
    a.barh(range(n), top.values, color=BLUE, height=0.7)
    a.set_yticks(range(n), [p.replace('_', '–', 1) for p in top.index], fontsize=10)
    a.set_xlabel('LR loading'); a.set_title(f'{f} ({LABEL[f]}): top LR pairs')
    a.grid(axis='x', color='#e6e5e0', lw=0.6); a.set_axisbelow(True)
    M = np.outer(L['sender'][f], L['receiver'][f])
    im = b.imshow(M, cmap=SEQ, vmin=0)
    cells = L['sender'].index
    b.set_xticks(range(len(cells)), cells, rotation=60, ha='right', fontsize=9)
    b.set_yticks(range(len(cells)), cells, fontsize=9)
    b.set_xlabel('receiver'); b.set_ylabel('sender'); b.set_title('sender × receiver'); b.grid(False)
    plt.colorbar(im, ax=b, shrink=0.8)
    fig.subplots_adjust(wspace=0.75)
    fig.savefig(HERE / f'tc2c_program_{f.replace(" ", "")}.png'); plt.close(fig)


def kegg(cols, q=0.25):
    g = pd.read_csv(D / 'gsea_kegg.csv')
    keep = g.loc[g.q < q, 'pathway'].unique()
    g = g[g.pathway.isin(keep)]
    paths = sorted(keep)
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 4.6, 0.45 * len(paths) + 1.6))
    m = np.nanmax(np.abs(g.NES))
    for _, r in g.iterrows():
        ax.scatter(cols.index(r.factor), paths.index(r.pathway), s=30 + 280 * min(-np.log10(r.p), 3) / 3,
                   c=[r.NES], cmap=DIV, vmin=-m, vmax=m, edgecolor=INK if r.q < 0.1 else 'none', lw=1)
    sc = ax.scatter([], [], c=[], cmap=DIV, vmin=-m, vmax=m)
    plt.colorbar(sc, ax=ax, label='enrichment (NES)', shrink=0.7)
    ax.set_xticks(range(len(cols)), [f'{c}\n{LABEL[c]}' for c in cols], fontsize=9)
    ax.set_yticks(range(len(paths)), paths)
    ax.set_xlim(-0.6, len(cols) - 0.4); ax.set_ylim(-0.6, len(paths) - 0.4)
    ax.grid(color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
    ax.set_title('KEGG pathways per factor (dot size = significance, black ring = q < 0.1)')
    fig.savefig(HERE / 'tc2c_kegg.png'); plt.close(fig)


def progeny(cols):
    t = pd.read_csv(D / 'progeny_t.csv', index_col=0)[cols]
    p = pd.read_csv(D / 'progeny_p.csv', index_col=0)[cols]
    m = np.abs(t.values).max()
    fig, ax = plt.subplots(figsize=(1.25 * len(cols) + 3, 0.42 * len(t) + 1.4))
    im = ax.imshow(t.values, cmap=DIV, vmin=-m, vmax=m, aspect='auto'); ax.grid(False)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f'{t.values[i, j]:.1f}' + ('*' if p.values[i, j] < 0.05 else ''),
                    ha='center', va='center', fontsize=9, color=INK)
    ax.set_xticks(range(len(cols)), [f'{c}\n{LABEL[c]}' for c in cols], fontsize=9)
    ax.set_yticks(range(len(t)), t.index)
    for s in ax.spines.values():
        s.set_visible(False)
    plt.colorbar(im, ax=ax, label='pathway activity (t)', shrink=0.8)
    ax.set_title('PROGENy pathway activity per factor (* p < 0.05)')
    fig.savefig(HERE / 'tc2c_progeny.png'); plt.close(fig)


if __name__ == '__main__':
    meta, L, T = load()
    cols = list(L['contexts'].columns)
    elbow()
    factors(meta, L)
    context_boxes(meta, L, T)
    for f in ('Factor 2', 'Factor 5', 'Factor 1'):
        program(L, f)
    kegg(cols)
    progeny(cols)
    print('written to', HERE)
