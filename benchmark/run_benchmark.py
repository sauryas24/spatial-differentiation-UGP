"""Simulate every scenario x replicate and run the three tools on each sample.

Resumable: cohorts already on disk are reused and finished samples are skipped.
    python3 run_benchmark.py --reps 3 --workers 8
Outputs
    sim_data/<scenario>/rep<r>/<sample>.h5ad, truth.csv, commot_db.csv
    tool_out/<scenario>/rep<r>/<sample>.pkl   {'spatialdm', 'commot', 'commot_ct', 'stlearn'}
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[v] = '1'

import argparse
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simulate as sim  # noqa: E402

DATA = HERE / 'data_dl'
ROOT = HERE.parent
LR_ARGS = (ROOT / 'resources' / 'interaction_input_CellChatDB.csv', ROOT / 'resources' / 'complex_input_CellChatDB.csv')


def cohort_dir(scenario, rep):
    return HERE / 'sim_data' / scenario / f'rep{rep}'


def out_path(scenario, rep, sample):
    return HERE / 'tool_out' / scenario / f'rep{rep}' / f'{sample}.pkl'


def seed_for(scenario, rep):
    return 1000 * (list(sim.SCENARIOS).index(scenario) + 1) + rep


def make_cohorts(reps):
    import run_tools as rt
    ref = pickle.load(open(HERE / 'ref.pkl', 'rb'))
    lr = sim.single_chain_secreted(*LR_ARGS)
    layout = sim.load_layout(DATA / 'coord.csv', DATA / 'interest_region_Spot_IDs.csv')
    jobs = []
    for name, sc in sim.SCENARIOS.items():
        for rep in range(reps):
            d = cohort_dir(name, rep)
            if not (d / 'truth.csv').exists():
                adatas, truth, _ = sim.simulate_cohort(ref, layout, lr, sc, seed_for(name, rep))
                d.mkdir(parents=True, exist_ok=True)
                for s, a in adatas.items():
                    a.write_h5ad(d / f'{s}.h5ad')
                db = rt.commot_db(list(adatas.values()), lr)
                db.to_csv(d / 'commot_db.csv')
                truth.to_csv(d / 'truth.csv', index=False)
            for f in sorted(d.glob('*.h5ad')):
                if not out_path(name, rep, f.stem).exists():
                    jobs.append((name, rep, f.stem))
    return jobs


def run_one(job):
    import anndata as ad
    import run_tools as rt
    warnings.filterwarnings('ignore')
    name, rep, sample = job
    d = cohort_dir(name, rep)
    a = ad.read_h5ad(d / f'{sample}.h5ad')
    db = pd.read_csv(d / 'commot_db.csv', index_col=0)
    db.columns = range(db.shape[1])
    lr = [(f'{l}_{r}', l, r) for l, r in zip(db[0], db[1])]
    t = time.time()
    res = {'spatialdm': rt.run_spatialdm(a)}
    res['stlearn'] = rt.run_stlearn(a, lr, seed=seed_for(name, rep))
    res['commot'], res['commot_ct'] = rt.run_commot(a, db, sim.TYPES)
    res['seconds'] = time.time() - t
    p = out_path(name, rep, sample)
    p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(p, 'wb'))
    return job, res['seconds']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    jobs = make_cohorts(args.reps)
    print(f'{len(jobs)} samples to run', flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                job, sec = f.result()
                print(f'[{i}/{len(jobs)}] {job} {sec:.0f}s', flush=True)
            except Exception:
                print(f'[{i}/{len(jobs)}] FAILED {futs[f]}\n{traceback.format_exc()}', flush=True)
