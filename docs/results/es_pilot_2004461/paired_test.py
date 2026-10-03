"""Paired-by-seed comparison of mean reward between the three networks in this folder.
Run from the repo root: python docs/results/es_pilot_2004461/paired_test.py
Critical values for df = 19 (20 seeds), two-sided: 2.093 at 0.05, 2.625 at 0.05/3 (Bonferroni, three tests),
looked up with scipy.stats.t.ppf, which is not a repo dependency."""
import numpy as np
import pandas as pd

D = "docs/results/es_pilot_2004461/"
es = pd.read_csv(D + "eval_2004461.csv").query("policy == 'mlp'").set_index("seed").reward
cl = pd.read_csv(D + "eval_2004551.csv").query("policy == 'mlp'").set_index("seed").reward
rl = pd.read_csv(D + "eval_2004551.csv").query("policy == 'rule'").set_index("seed").reward
for name, a, b in [("ES checkpoint - clone", es, cl), ("clone - rule", cl, rl), ("ES checkpoint - rule", es, rl)]:
    d = (a - b).to_numpy()
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    print(f"{name}: mean diff {d.mean():.4f}, se {se:.4f}, t {d.mean() / se:.1f} (df {n - 1}), "
          f"higher on {int((d > 0).sum())}/{n} seeds")
