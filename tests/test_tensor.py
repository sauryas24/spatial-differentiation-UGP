"""Sanity checks for samdiff. Run from the project root: python3 tests/test_tensor.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from samdiff import sam, tensor as tc

G1, G2 = ['A1', 'A2'], ['A3', 'A4', 'A8', 'A9']


def test_sam_matches_notebook():
    # final.ipynb: 918 complete pairs, s0 = 2.255, 162 pairs with q < 0.1
    tab = pd.read_csv(Path(__file__).resolve().parents[1] / 'results' / 'spatialdm_results.csv', index_col=0)
    res = sam.sam(tab, G1, G2)
    assert len(res) == 918, len(res)
    assert abs(res.attrs['s0'] - 2.255) < 1e-3, res.attrs['s0']
    assert (res['q'] < 0.1).sum() == 162, (res['q'] < 0.1).sum()


def test_recovers_planted_patterns():
    tabs, truth = tc.simulate_tables(effect=4.0, seed=1)
    ct = tc.build_tensor_a(tabs, G1 + G2, min_obs=6)
    dec = tc.decompose(ct, rank=3, n_runs=5)
    test = tc.loading_test(dec, G1, G2)
    lr = dec.factors['LR pairs']
    # the factor most adult-weighted should be enriched for g1 pairs, and vice versa
    f_ad = test['t'].idxmax()
    f_fe = test['t'].idxmin()
    top_ad = set(lr[f_ad].nlargest(30).index)
    top_fe = set(lr[f_fe].nlargest(30).index)
    assert (truth[list(top_ad)] == 'g1').sum() >= 10
    assert (truth[list(top_fe)] == 'g2').sum() >= 10
    assert np.isclose(test['p_floor'].iloc[0], 1 / 15)
    assert (test['p_perm'] >= 1 / 15 - 1e-12).all()


def test_mask_ignores_missing_entries():
    tabs, _ = tc.simulate_tables(seed=2)
    tabs['method1'].iloc[:50, 2:] = np.nan          # pairs scored in only 2 samples
    ct = tc.build_tensor_a(tabs, G1 + G2, min_obs=4)
    assert ct.mask[:, :, 0].sum() == 250 * 6        # those 50 rows dropped for method1
    assert ct.shape[1] == 300                        # but kept via the other methods
    dec = tc.decompose(ct, rank=2, n_runs=2)
    assert np.isfinite(dec.error) and dec.error < 1


def test_permutation_on_null_is_uniform_ish():
    # no planted effect: the max-T p-value should rarely be at its floor
    hits = 0
    for seed in range(20):
        tabs, _ = tc.simulate_tables(n_planted=0, seed=100 + seed)
        dec = tc.decompose(tc.build_tensor_a(tabs, G1 + G2), rank=2, n_runs=2, seed=seed)
        hits += (tc.loading_test(dec, G1, G2)['p_maxT'] <= 1 / 15 + 1e-12).any()
    assert hits <= 6, hits                           # expected about 20 * 1/15 ≈ 1.3


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
