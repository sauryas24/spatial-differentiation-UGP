"""Presentation figures for the method comparison (from agreement.py --heart).
  agreement_gut.png      : gut, Spearman correlation of signed per-pair scores, all four methods
  agreement_programs_<gut|heart>.png : tensor programs vs the other methods: of each
                           program's top 20 LR pairs, how many does a method call higher in
                           the program's condition (bars) vs the number expected by chance
    python3 figures/make_agreement_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRAY = '#0b0b0b', '#52514e', '#b4b2a9'
DIV = LinearSegmentedColormap.from_list('div', [BLUE, '#f0efec', ORANGE])
plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 200, 'font.size': 11, 'axes.titlesize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': GRAY,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})
METHODS = ['SpatialDM', 'T2 SAM', 'Structured', 'Tensor']
METHOD_C = {'SpatialDM': YELLOW, 'T2 SAM': AQUA, 'Structured': BLUE}
COND_NAME = {'adult': 'adult', 'fetal': 'fetal', 'control': 'healthy', 'IZ': 'ischemic', 'FZ': 'fibrotic'}


def gut_heatmap():
    df = pd.read_csv(ROOT / 'results' / 'agreement_gut.csv')
    M = pd.DataFrame(np.nan, METHODS, METHODS)
    for _, r in df.iterrows():
        M.loc[r.method_2, r.method_1] = r.spearman
    n = len(METHODS)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(M.to_numpy(float), cmap=DIV, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M.iloc[i, j]):
                ax.text(j, i, f'{M.iloc[i, j]:.2f}', ha='center', va='center', fontsize=13,
                        color=INK, fontweight='bold')
    ax.set_xticks(range(n), METHODS, rotation=30, ha='right'); ax.set_yticks(range(n), METHODS)
    ax.set_xlim(-0.5, n - 1.5); ax.set_ylim(n - 0.5, 0.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('Spearman correlation\nof signed pair scores')
    ax.set_title('Gut: adult vs fetal')
    fig.savefig(HERE / 'agreement_gut.png'); plt.close(fig)


def programs():
    pr = pd.read_csv(ROOT / 'results' / 'agreement_programs.csv')
    for ds, name, title, size in [('gut', 'gut', 'Gut: adult vs fetal programs', (7.5, 4)),
                                  ('mi', 'heart', 'Heart: zone programs', (6.5, 4))]:
        fig, ax = plt.subplots(figsize=size)
        d = pr[pr.dataset == ds]
        progs = (d.drop_duplicates('program').sort_values(['p_condition', 'program']).program.tolist())
        meths = [m for m in METHOD_C if m in set(d.method)]
        w = 0.8 / len(meths)
        for i, p in enumerate(progs):
            for j, m in enumerate(meths):
                r = d[(d.program == p) & (d.method == m)].iloc[0]
                x = i + (j - (len(meths) - 1) / 2) * w
                ax.bar(x, r.top20_called, w * 0.9, color=METHOD_C[m], label=m if i == 0 else None)
                ax.hlines(r.expected, x - w * 0.45, x + w * 0.45, color=INK, lw=2,
                          label='expected by chance' if i == 0 and j == 0 else None)
                if r.p_enrichment < 0.05:
                    ax.text(x, r.top20_called + 0.4, '*', ha='center', fontsize=14, color=INK)
        labels = []
        for p in progs:
            r = d[d.program == p].iloc[0]
            sig = '' if r.p_condition <= 0.07 else '\n(not different)'
            labels.append(f'{p}\n{COND_NAME[r.condition]}{sig}')
        ax.set_xticks(range(len(progs)), labels, fontsize=10)
        ax.set_ylim(0, 21); ax.set_title(title)
        ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
        ax.set_ylabel('top 20 program pairs called\nhigher in that condition')
        ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.22))
        fig.savefig(HERE / f'agreement_programs_{name}.png'); plt.close(fig)


if __name__ == '__main__':
    gut_heatmap()
    programs()
    print('written to', HERE)
