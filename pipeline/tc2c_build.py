"""Build Tensor-cell2cell input tensors (contexts x LR pairs x senders x receivers)
from spatially restricted communication scores, for the gut and the MI data.

The scores replace LIANA in the Baghdassarian et al. (2024) protocol: for every
sample, pair and sender/receiver cell type, ligand in sender spots x kernel-weighted
receptor in neighbouring receiver spots (real_structured.lineage_slices). As in the
protocol, each sample's scores are turned into a magnitude rank (1 = the sample's
strongest interaction, 0 = no signal), the equivalent of LIANA's magnitude_rank
inverted with 1 - x. Ranks are within-sample, so section size and sequencing depth
drop out. Entries with no signal are biological zeros, not missing values.
    python3 tc2c_build.py [--score rank|log]
    python3 tc2c_build.py --consensus   (gut only; needs spot_scores.py for all three tools)
The --consensus mode follows LIANA's rank_aggregate instead: COMMOT, SpatialDM and
stLearn spot scores are split into sender x receiver cell types, ranked per method
within each sample, and combined with RobustRankAggregate; 1 - magnitude_rank fills the
tensor (also one tensor per tool, ranked alone).
Writes tc2c/<dataset>_tensor.npz and tc2c/<dataset>_meta.json.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parents[1]          # repository root
RESULTS, RESOURCES = HERE / 'results', HERE / 'resources'
GUT_DATA = HERE / 'data' / 'gut'                     # gut .h5ad files (not in the repository)
sys.path[:0] = [str(HERE), str(HERE / 'benchmark')]
import real_structured as rs  # noqa: E402
from samdiff import tensor as tc  # noqa: E402

OUT = RESULTS / 'tc2c'


def magnitude(T, score):
    """Per-sample transform of a samples x pairs x senders x receivers array."""
    out = np.zeros_like(T)
    for i, S in enumerate(T):
        pos = S > 0
        if score == 'rank':                    # 1 = strongest, -> 0 for the weakest positive
            r = rankdata(-S[pos], method='average')
            out[i][pos] = 1 - (r - 1) / pos.sum()
        else:                                  # log1p of signal per spot-pair, scaled per sample
            out[i] = np.log1p(S / np.median(S[pos]))
    return out


def assemble(slices, samples, pairs, keep):
    return np.stack([[slices[s][p][np.ix_(keep, keep)] for p in pairs] for s in samples])


def gut():
    lr = rs.lr_table()
    genes = sorted({g for _, l, r, _ in lr for g in l + r})
    samples = rs.G1 + rs.G2
    data = {s: rs.load_sample(s, genes) for s in samples}
    pairs = sorted(set.intersection(*[rs.detected(X, lr) for X, _, _ in data.values()]))
    sel = [p for p in lr if p[0] in set(pairs)]
    slices = {s: rs.lineage_slices(X, P, rs.kernel(xy), sel) for s, (X, P, xy) in data.items()}
    abund = pd.DataFrame({s: P.sum(0) for s, (_, P, _) in data.items()}, index=tc.LINEAGES).T
    keep = [i for i, t in enumerate(tc.LINEAGES) if abund[t].min() >= rs.MIN_ABUND]
    cond = {s: 'adult' if s in rs.G1 else 'fetal' for s in samples}
    ann = {n: a for n, _, _, a in lr}
    return assemble(slices, samples, pairs, keep), samples, pairs, [tc.LINEAGES[i] for i in keep], cond, ann


def mi():
    import mi_multi as mm
    import mi_structured as mis
    lr = rs.lr_table()
    samples = sum(mm.GROUPS.values(), [])
    data = {s: mis.load(s) for s in samples}
    pairs = sorted(set.intersection(*[rs.detected(data[s][0], lr) for s in samples]))
    sel = [p for p in lr if p[0] in set(pairs)]
    types = data[samples[0]][4]
    slices = {s: rs.lineage_slices(X, P, rs.kernel(xy, mis.L_SCALE * nn), sel)
              for s, (X, P, xy, nn, _) in data.items()}
    abund = pd.DataFrame({s: data[s][1].sum(0) for s in samples}, index=types).T
    keep = [i for i, t in enumerate(types) if abund[t].min() >= mis.MIN_ABUND]
    cond = {s: z for z, g in mm.GROUPS.items() for s in g}
    ann = {n: a for n, _, _, a in lr}
    return assemble(slices, samples, pairs, keep), samples, pairs, [types[i] for i in keep], cond, ann


TOOLS = ['commot', 'spatialdm', 'stlearn']


def tool_slices(S, P, K):
    """Spot scores (spots x pairs) -> pairs x senders x receivers: a spot's score goes to
    the sender types in the spot and the receiver types in its neighbourhood (kernel-
    weighted composition), each by its proportion."""
    Q = (K / K.sum(1, keepdims=True)) @ P
    return np.einsum('sp,sa,sb->pab', S, P, Q, optimize=True)


def norm_rank(C):
    """LIANA's normalised rank within one sample and one method: rank / number of
    interactions, 1 = strongest; ties (e.g. all the zeros) share their average rank."""
    return rankdata(-C.ravel(), method='average').reshape(C.shape) / C.size


def rank_aggregate(R):
    """RobustRankAggregate (Kolde et al. 2012) as in LIANA: for each interaction, the
    sorted normalised ranks r_(1) <= ... <= r_(m) of the m methods; beta_k is the chance
    that the k-th smallest of m uniform ranks is <= r_(k); rho = min_k beta_k, times m
    (Bonferroni), capped at 1. Small = consistently top-ranked."""
    from scipy.stats import beta
    m = R.shape[0]
    srt = np.sort(R, axis=0)
    b = np.stack([beta.cdf(srt[k - 1], k, m - k + 1) for k in range(1, m + 1)])
    return np.minimum(b.min(0) * m, 1.0)


def gut_tools():
    """Per-tool and consensus tensors from tc2c/spot_scores (spot_scores.py)."""
    meta = json.load(open(OUT / 'gut_meta.json'))
    lr = rs.lr_table()
    genes = sorted({g for _, l, r, _ in lr for g in l + r})
    keep = [tc.LINEAGES.index(c) for c in meta['cells']]
    per_tool = {t: [] for t in TOOLS}
    for s in meta['contexts']:
        _, P, xy = rs.load_sample(s, genes)
        K = rs.kernel(xy)
        K = K.toarray() if hasattr(K, 'toarray') else np.asarray(K)
        for t in TOOLS:
            d = np.load(OUT / 'spot_scores' / f'{t}_{s}.npz')
            assert list(d['pairs']) == meta['pairs']
            per_tool[t].append(tool_slices(d['scores'], P[:, keep], K))
    out = {}
    for t in TOOLS:
        out[t] = np.stack(per_tool[t])                        # raw per-tool scores
    ranks = {t: np.stack([norm_rank(C) for C in out[t]]) for t in TOOLS}
    signal = sum(out[t] for t in TOOLS) > 0
    mag = np.stack([rank_aggregate(np.stack([ranks[t][i] for t in TOOLS])) for i in range(len(meta['contexts']))])
    tensors = {f'gut_{t}': np.where(out[t] > 0, 1 - ranks[t], 0.0) for t in TOOLS}
    tensors['gut_consensus'] = np.where(signal, 1 - mag, 0.0)  # magnitude_rank inverted with 1 - x
    return tensors, meta


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--score', default='rank', choices=['rank', 'log'])
    ap.add_argument('--datasets', default='gut,mi')
    ap.add_argument('--consensus', action='store_true',
                    help='per-tool and LIANA-style consensus tensors from tc2c/spot_scores')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.consensus:
        tensors, meta = gut_tools()
        for tag, X in tensors.items():
            np.savez_compressed(OUT / f'{tag}_tensor.npz', tensor=X)
            json.dump({**meta, 'score': tag}, open(OUT / f'{tag}_meta.json', 'w'), indent=1)
            print(f'{tag}: tensor {X.shape}, {(X == 0).mean():.0%} zeros')
        sys.exit()
    for name in a.datasets.split(','):
        T, samples, pairs, types, cond, ann = {'gut': gut, 'mi': mi}[name]()
        X = magnitude(T, a.score)
        tag = name if a.score == 'rank' else f'{name}_{a.score}'
        np.savez_compressed(OUT / f'{tag}_tensor.npz', tensor=X)
        json.dump({'contexts': samples, 'pairs': pairs, 'cells': types, 'condition': cond,
                   'annotation': {p: ann[p] for p in pairs}, 'score': a.score},
                  open(OUT / f'{tag}_meta.json', 'w'), indent=1)
        print(f'{tag}: tensor {X.shape} (contexts x LR pairs x senders x receivers), '
              f'{(X == 0).mean():.0%} zeros')
