"""Paired-by-seed comparison of mean reward: the seed-1 replication (jobs 2004980, 2005005) against rule and the
seed-0 pilot (jobs 2004461, 2004551; files in docs/results/es_pilot_2004461/).
Run from the repo root: python docs/results/es_replication_2004980/paired_test.py
Critical values, df = 19 (20 seeds), two-sided, from scipy.stats.t.ppf (not a repo dependency):
2.093 at 0.05; 2.759 at 0.05/4 (Bonferroni for the four tests below)."""
import numpy as np
import pandas as pd

R, P = "docs/results/es_replication_2004980/", "docs/results/es_pilot_2004461/"


def reward(path, policy="mlp"):
    return pd.read_csv(path).query("policy == @policy").set_index("seed").reward


es1 = reward(R + "eval_2004980_mlp.csv")      # seed 1, best checkpoint of the 20-generation run
cl1 = reward(R + "eval_2005005_mlp.csv")      # seed 1, cloned network, no ES updates
rule = reward(P + "eval_2004551.csv", "rule")
es0 = reward(P + "eval_2004461.csv")          # seed 0, best checkpoint (the pilot)
for name, a, b in [("seed-1 ES - seed-1 clone", es1, cl1), ("seed-1 clone - rule", cl1, rule),
                   ("seed-1 ES - rule", es1, rule), ("seed-1 ES - seed-0 ES", es1, es0)]:
    d = (a - b).to_numpy()
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    print(f"{name}: mean diff {d.mean():+.4f}, se {se:.4f}, t {d.mean() / se:.1f} (df {n - 1}), "
          f"first higher on {int((d > 0).sum())}/{n} seeds")
