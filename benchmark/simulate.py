"""Two-condition spot-level simulator for benchmarking differential LR analysis.

Built on the spot-level design of Ku et al. (Genome Biology 2026, "Benchmarking
tools for deciphering cellular crosstalk in spatially-resolved transcriptomics"):
the same Visium layout (782 spots, 133-spot central interaction region), 12 cells
per spot drawn from the Baron human pancreas scRNA-seq reference (GSM2230759),
and per-cell-type negative-binomial expression fitted to that reference, with
the same sparsity rule (a gene with <= 10 UMIs or < 7 expressing cells in a type
is zero there).

What is added for a *differential* benchmark:
  * several samples per condition, each an independent draw of composition and
    expression on the same layout;
  * interaction programs planted by scaling the ligand in sender-type cells inside
    the central region and the receptor in receiver-type cells in the central
    region plus one ring of neighbours (the paper's paracrine embedding), active
    in condition A only, condition B only, or both (shared = not differential);
  * nuisances that real cohorts have: a per-sample sequencing-depth factor that
    can be confounded with condition, and a shift in cell-type composition.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TYPES = ['beta', 'delta', 'ductal', 'macrophage', 'activated_stellate',
         'quiescent_stellate', 'endothelial']

# sender -> receiver for each planted program
PROGRAMS = {'A': ('ductal', 'activated_stellate'),     # active in condition A only
            'B': ('beta', 'endothelial'),               # active in condition B only
            'shared': ('delta', 'ductal')}              # active in every sample

SPACING = 100.0     # um between neighbouring Visium spots


@dataclass
class Scenario:
    name: str
    n1: int = 3                  # samples in condition A
    n2: int = 3                  # samples in condition B
    fold: float = 4.0            # expression fold for planted genes
    n_diff: int = 10             # planted pairs per differential program (A and B)
    n_shared: int = 10           # planted pairs active in both conditions
    depth_shift: float = 1.0     # condition-A sequencing depth multiplier (1 = none)
    comp_shift: float = 0.0      # extra logit for the program-A sender type in condition A
    null: bool = False           # no differential programs at all
    scatter: bool = False        # each planted pair gets its own random sender/receiver
    n3: int = 0                  # samples in condition C (0 = two-condition design)
    active: tuple = None         # multi-condition designs: ((condition, (programs...)), ...)
    comp_strength: float = 1.0   # amplitude of the smooth spatial composition fields
    min_mu: float = 0.1          # minimum reference mean of planted genes in their type
    relocate: bool = False       # differential programs change co-localisation only (see simulate_sample)

    def samples(self):
        g = ([f'A{i + 1}' for i in range(self.n1)], [f'B{i + 1}' for i in range(self.n2)])
        return g + ([f'C{i + 1}' for i in range(self.n3)],) if self.n3 else g


SCENARIOS = {s.name: s for s in [
    Scenario('null', null=True),
    Scenario('fold2'),
    Scenario('fold4'),
    Scenario('fold4_2v4', n1=2, n2=4),
    Scenario('fold4_depth', depth_shift=1.6),
    Scenario('fold4_comp', comp_shift=0.8),
    Scenario('fold4_scatter', scatter=True),
    # three conditions: A = control, B = stage 1 (program A on), C = stage 2 (A and B on)
    Scenario('multi_null', n3=3, null=True),
    Scenario('multi_fold4', n3=3, active=(('A', ()), ('B', ('A',)), ('C', ('A', 'B')))),
    Scenario('multi_fold2', n3=3, fold=2.0, active=(('A', ()), ('B', ('A',)), ('C', ('A', 'B')))),
    Scenario('multi_comp', n3=3, comp_shift=0.8, active=(('A', ()), ('B', ('A',)), ('C', ('A', 'B')))),
    Scenario('multi_4each', n1=4, n2=4, n3=4, active=(('A', ()), ('B', ('A',)), ('C', ('A', 'B')))),
    # sensitivity of the SpatialDM comparison to the simulator's choices
    Scenario('sdm_null_flat', null=True, comp_strength=0.0),
    Scenario('sdm_flat', comp_strength=0.0),
    Scenario('sdm_weak', comp_strength=0.4),
    Scenario('sdm_hiexpr', min_mu=0.3, n_diff=8, n_shared=6),
    Scenario('sdm_n6', n1=6, n2=6),
    Scenario('sdm_reloc', relocate=True),
    Scenario('sdm_reloc_flat', relocate=True, comp_strength=0.0),
    Scenario('sdm_reloc_n6', relocate=True, n1=6, n2=6),
]}


# =========================================================================
# Reference
# =========================================================================

@dataclass
class Reference:
    genes: list
    mu: np.ndarray               # types x genes, mean UMI per cell
    size: np.ndarray             # types x genes, NB size (inf = Poisson)
    type_freq: np.ndarray        # reference cell-type frequencies


def fit_reference(csv_path, genes, types=TYPES):
    """Per-type NB fit by method of moments (the paper used MASS::fitdistr)."""
    header = pd.read_csv(csv_path, nrows=0).columns
    genes = [g for g in dict.fromkeys(genes) if g in header]
    df = pd.read_csv(csv_path, usecols=['assigned_cluster'] + genes)
    df = df[df['assigned_cluster'].isin(types)]
    mu = np.zeros((len(types), len(genes)))
    size = np.full_like(mu, np.inf)
    freq = np.zeros(len(types))
    for i, t in enumerate(types):
        X = df.loc[df['assigned_cluster'] == t, genes].to_numpy(float)
        freq[i] = len(X)
        m, v = X.mean(0), X.var(0, ddof=1)
        keep = (X.sum(0) > 10) & ((X > 0).sum(0) >= 7)
        mu[i] = np.where(keep, m, 0.0)
        over = keep & (v > m)
        size[i, over] = m[over] ** 2 / (v[over] - m[over])
    return Reference(genes, mu, size, freq / freq.sum())


def load_layout(coord_csv, region_csv):
    coord = pd.read_csv(coord_csv, index_col=0)[['x', 'y']].astype(float)
    xy = coord.to_numpy()
    from scipy.spatial import cKDTree
    nn = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
    xy = (xy - xy.min(0)) * SPACING / nn
    center = coord.index.isin(pd.read_csv(region_csv)['interest_region_Spot_IDs'])
    tree = cKDTree(xy)
    ring = np.zeros(len(xy), bool)
    for i in np.where(center)[0]:
        ring[tree.query_ball_point(xy[i], 1.5 * SPACING)] = True
    return pd.DataFrame(xy, index=coord.index, columns=['x', 'y']), center, ring


# =========================================================================
# Planted pairs
# =========================================================================

def choose_programs(ref, lr_pairs, scenario, rng):
    """Pick distinct single-chain pairs whose ligand is expressed by the program's
    sender type and receptor by its receiver type (as the paper required)."""
    g = {x: i for i, x in enumerate(ref.genes)}
    ti = {t: i for i, t in enumerate(TYPES)}
    used, out = set(), {}
    plan = [('shared', scenario.n_shared)] + ([] if scenario.null else
                                              [('A', scenario.n_diff), ('B', scenario.n_diff)])
    def eligible(s, r):
        return [(p, l, rc, s, r) for p, l, rc in lr_pairs
                if l in g and rc in g and p not in used
                and ref.mu[ti[s], g[l]] >= scenario.min_mu and ref.mu[ti[r], g[rc]] >= scenario.min_mu]

    for prog, n in plan:
        if scenario.scatter and prog != 'shared':
            pick = []
            while len(pick) < n:
                s, r = (TYPES[i] for i in rng.choice(len(TYPES), 2))
                ok = eligible(s, r)
                if ok:
                    pick.append(ok[rng.integers(len(ok))])
                    used.add(pick[-1][0])
        else:
            ok = eligible(*PROGRAMS[prog])
            if len(ok) < n:
                raise ValueError(f'only {len(ok)} eligible pairs for program {prog}')
            pick = [ok[i] for i in rng.choice(len(ok), n, replace=False)]
            used.update(x[0] for x in pick)
        out[prog] = pick
    return out


# =========================================================================
# One sample
# =========================================================================

def _smooth_fields(xy, n_types, rng, n_centers=4, width=3.0 * SPACING):
    F = np.zeros((len(xy), n_types))
    for t in range(n_types):
        c = xy[rng.choice(len(xy), n_centers, replace=False)]
        d2 = ((xy[:, None, :] - c[None]) ** 2).sum(-1)
        F[:, t] = np.exp(-d2 / (2 * width ** 2)).sum(1)
    return (F - F.mean(0)) / F.std(0)


def _far_region(xy, ring, min_gap=400.0):
    """A compact patch the size of the receptor region (centre + one ring), as far
    from it as the tissue allows and at least min_gap away (out of kernel range)."""
    from scipy.spatial import cKDTree
    tree, to_ring = cKDTree(xy), cKDTree(xy[ring])
    best, far = -1.0, None
    for seed in np.where(to_ring.query(xy)[0] > min_gap)[0]:
        idx = tree.query(xy[seed], k=int(ring.sum()))[1]
        gap = to_ring.query(xy[idx])[0].min()
        if gap > best:
            best, far = gap, idx
    if best < min_gap:
        raise ValueError('no patch out of kernel range')
    out = np.zeros(len(xy), bool)
    out[far] = True
    return out


def simulate_sample(ref, layout, programs, active, scenario, condition, rng,
                    cells_per_spot=12, depth_sd=0.15):
    """Counts (spots x genes) and true composition (spots x types) for one sample."""
    xy, center, ring = layout
    n_spots, T = len(xy), len(TYPES)

    logits = np.log(ref.type_freq) + scenario.comp_strength * _smooth_fields(xy.to_numpy(), T, rng)
    if condition == 'A' and scenario.comp_shift:
        logits[:, TYPES.index(PROGRAMS['A'][0])] += scenario.comp_shift
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    counts = np.stack([rng.multinomial(cells_per_spot, p) for p in P])     # spots x types

    depth = rng.lognormal(0, depth_sd) * (scenario.depth_shift if condition == 'A' else 1.0)
    g = {x: i for i, x in enumerate(ref.genes)}
    # relocate: every program is planted in every sample with the same amounts, but an
    # inactive program boosts its receptor in a distant region instead of around the ligand
    planted = list(programs) if scenario.relocate else active
    far = _far_region(xy.to_numpy(), ring) if scenario.relocate else None
    X = np.zeros((n_spots, len(ref.genes)))
    for t in range(T):
        n = counts[:, t].astype(float)
        mu = np.broadcast_to(ref.mu[t] * depth, (n_spots, len(ref.genes))).copy()
        for prog in planted:
            for _, l, rc, s, r in programs[prog]:
                if TYPES[t] == s:
                    mu[center, g[l]] *= scenario.fold
                if TYPES[t] == r:
                    mu[ring if prog in active else far, g[rc]] *= scenario.fold
        size = ref.size[t]
        has = (n > 0)[:, None] & (mu > 0)
        # a sum of n iid NB(mu, size) draws is NB(n*mu, n*size)
        with np.errstate(invalid='ignore'):
            k = np.where(np.isfinite(size), n[:, None] * size[None, :], np.inf)
        lam = n[:, None] * mu
        pois = ~np.isfinite(k)
        draw = np.zeros_like(lam)
        nb = has & ~pois
        draw[nb] = rng.negative_binomial(k[nb], k[nb] / (k[nb] + lam[nb]))
        draw[has & pois] = rng.poisson(lam[has & pois])
        X += draw
    comp = pd.DataFrame(counts / cells_per_spot, index=xy.index, columns=TYPES)
    return X, comp, depth


def simulate_cohort(ref, layout, lr_pairs, scenario, seed):
    """All samples of one replicate, plus the ground truth."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    programs = choose_programs(ref, lr_pairs, scenario, rng)
    groups = scenario.samples()
    act = dict(scenario.active) if scenario.active else {}
    adatas = {}
    for name in sum(groups, []):
        cond = name[0]
        if scenario.null:
            active = ['shared']
        elif scenario.active:
            active = ['shared'] + list(act[cond])
        else:
            active = ['shared', cond]
        X, comp, depth = simulate_sample(ref, layout, programs, active, scenario, cond, rng)
        a = ad.AnnData(X.astype(np.float32), obs=comp.copy(),
                       var=pd.DataFrame(index=ref.genes))
        a.obsm['spatial'] = layout[0].to_numpy()
        a.uns['depth'] = depth
        adatas[name] = a
    truth = pd.DataFrame([{'pair': p, 'program': prog, 'sender': s, 'receiver': r,
                           'direction': ({'A': 1, 'B': 1, 'shared': 0} if scenario.n3 else
                                         {'A': 1, 'B': -1, 'shared': 0})[prog]}
                          for prog, lst in programs.items() for p, _, _, s, r in lst])
    return adatas, truth, groups


# =========================================================================
# LR universe and gene set
# =========================================================================

def single_chain_secreted(interaction_csv, complex_csv):
    """(pair, ligand, receptor) for CellChat secreted pairs with no complexes."""
    i = pd.read_csv(interaction_csv, index_col=0)
    comp = set(pd.read_csv(complex_csv, index_col=0).index)
    i = i[i['annotation'] == 'Secreted Signaling']
    out = []
    for name, row in i.iterrows():
        l, r = row['ligand'], row['receptor']
        if l in comp or r in comp or '_' in r or '_' in l:
            continue
        out.append((f'{l}_{r}', l, r))
    return out


def gene_universe(interaction_csv, complex_csv, ref_csv, n_background=1500, seed=0):
    """All CellChat LR genes (with complex subunits) plus random background genes,
    so library-size normalisation sees a realistic transcriptome slice."""
    i = pd.read_csv(interaction_csv, index_col=0)
    comp = pd.read_csv(complex_csv, index_col=0)
    genes = set()
    for col in ('ligand', 'receptor'):
        for x in i[col]:
            genes.update(comp.loc[x].dropna().tolist() if x in comp.index else [x])
    header = [c for c in pd.read_csv(ref_csv, nrows=0).columns
              if c not in ('Unnamed: 0', 'barcode', 'assigned_cluster')]
    rest = sorted(set(header) - genes)
    rng = np.random.default_rng(seed)
    bg = list(rng.choice(rest, n_background, replace=False))
    return sorted(g for g in genes if g in set(header)) + bg
