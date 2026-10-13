"""Re-run the power grid behind docs/power_analysis.md (about 20 minutes on 2 cores)."""
import itertools, json, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "src")
from lookahead import sim

NS, GAMMAS, NSIM = [50, 100, 200], [0.0, 0.05, 0.1, 0.2, 0.4], 60

def job(a):
    n, g, s = a
    r = sim.one_run(1000 + s, n, g, "coverage", True, True)
    return dict(n=n, gamma=g, seed=s, **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in r.items()})

if __name__ == "__main__":
    with Pool(2) as p, open("results/power_grid.jsonl", "w") as f:
        for r in p.imap_unordered(job, itertools.product(NS, GAMMAS, range(NSIM)), chunksize=4):
            f.write(json.dumps(r) + "\n"); f.flush()
