"""Generation-aware analysis (step B): what changed across generations, and was it selection or drift?

    python -m lifesim.generations --seeds 8 --years 150 --out runs/gen
    python -m lifesim.generations --arm no_gate birth_reserve_gate=False --arm assort assortative_mating=3

Each arm is a config override run over several seeds with births on. For every run it measures

  * gene means per generation, and each gene's change from the founders to the last generation that is big
    enough, in units of the gene's allowed range;
  * a drift null: genes the policy never reads (the marker, and e.g. `patience` under the rule policy) are
    inherited and mutated exactly like the others, so their change is what chance alone does. A gene that the
    policy reads is compared with the neutral genes of the same runs, paired by run;
  * the rank correlation between a child's real net worth at age 40 and their parents' (mean of the two);
  * inequality over time: the Gini of real net worth among living adults (debts count as zero).

Honest limits: with a handful of seeds the standard errors are large, nothing is corrected for multiple
comparisons, wealth at age 40 only exists for people who lived to 40 (survivorship), and the neutral genes are
only a valid null because the model has no linkage between genes.
"""
# ruff: noqa: E402
import os

# One process per core is the parallelism here, so keep every worker's BLAS single-threaded. This only takes
# effect if numpy is not imported yet (true for `python -m lifesim.generations`); users' own settings win.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .economy import Prices, update_prices
from .family import GENE_RANGES
from .parallel import default_workers
from .policy import genes_read, make_policy
from .run import parse_overrides
from .world import World

AGE_REF = 40.0          # age at which parent and child wealth are compared
DEFAULT_ARMS = {
    "baseline": {},
    "no_reserve_gate": {"birth_reserve_gate": False},
    "assortative": {"assortative_mating": 3.0},
}


# --------------------------------------------------------------------------------------------- measures

def price_idx_on(cfg: Config, day: int) -> float:
    """Price index of the day lived when a snapshot labelled `day` was taken (snapshots are taken after the
    tick, and prices are set at the start of it). Uses the simulation's own `update_prices`, so the two cannot
    drift apart; without an economy prices never move."""
    if not cfg.economy:
        return 1.0
    prices = Prices()
    update_prices(prices, cfg, day - 1)
    return prices.price_idx


def gini(values) -> float:
    """Gini of non-negative values (debts are floored at 0 by the caller). 0 = equal, (n-1)/n = one has all."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return float("nan") if n == 0 else 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def spearman(a, b) -> float:
    """Rank correlation with average ranks for ties (no scipy needed)."""
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    if len(a) < 3:
        return float("nan")
    ra, rb = a.rank().to_numpy(), b.rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def wealth_at_age(snaps: pd.DataFrame, cfg: Config, age: float = AGE_REF, tol: float | None = None) -> pd.Series:
    """Real (day-0) cash of each person at the snapshot nearest `age`, if one is within `tol` years.
    The default tolerance is half the snapshot interval plus a margin, so it follows `log_every`: whoever lived
    through `age` is within reach of a snapshot. People who died before `age` have none (survivorship), which is
    documented, not hidden."""
    if snaps.empty:
        return pd.Series(dtype=float)
    tol = cfg.log_every / 730 + 0.1 if tol is None else tol
    s = snaps.assign(gap=(snaps["age"] - age).abs())
    s = s[s["gap"] <= tol].sort_values(["id", "gap"]).drop_duplicates("id")
    idx = {int(d): price_idx_on(cfg, int(d)) for d in s["day"].unique()}
    real = s["cash"].to_numpy() / np.array([idx[int(d)] for d in s["day"]])
    return pd.Series(real, index=s["id"].to_numpy())


def genes_by_generation(lineage: pd.DataFrame, genes) -> pd.DataFrame:
    """Per generation: number of adults and mean/sd of every gene. Minors are not adults yet."""
    adults = lineage[lineage["status"] == "adult"]
    rows = []
    for gen, grp in adults.groupby("generation"):
        for g in genes:
            rows.append({"generation": int(gen), "gene": g, "n": len(grp),
                         "mean": float(grp[g].mean()), "sd": float(grp[g].std(ddof=0))})
    return pd.DataFrame(rows)


def inequality_series(snaps: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if snaps.empty:                                   # a run shorter than log_every has no snapshots
        return pd.DataFrame(columns=["day", "n_adults", "gini", "share_in_debt"])
    rows = []
    for day, grp in snaps.groupby("day"):
        real = grp["cash"].to_numpy() / price_idx_on(cfg, int(day))
        rows.append({"day": int(day), "n_adults": len(grp), "gini": gini(np.maximum(real, 0.0)),
                     "share_in_debt": float((real < 0).mean())})
    return pd.DataFrame(rows)


def parent_child_pairs(lineage: pd.DataFrame, wealth40: pd.Series) -> pd.DataFrame:
    """One row per adult whose own wealth at 40 and at least one parent's are known (parents averaged)."""
    adults = lineage[(lineage["status"] == "adult") & lineage["mother_id"].notna()]
    rows = []
    for r in adults.itertuples():
        if r.id not in wealth40.index:
            continue
        pw = [wealth40[p] for p in (r.mother_id, r.father_id) if p in wealth40.index]
        if pw:
            rows.append({"child_id": int(r.id), "generation": int(r.generation),
                         "child_wealth": float(wealth40[r.id]), "parent_wealth": float(np.mean(pw)),
                         "n_parents_known": len(pw)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------- one run

def analyze_world(world: World, policy: str) -> dict:
    """Everything the aggregation needs from one finished run (small, picklable)."""
    cfg = world.cfg
    if not cfg.reproduction:
        raise ValueError("analyze_world needs a run with reproduction=True")
    genes = sorted(next(iter(world.agents)).genes)
    lineage = pd.DataFrame(world.lineage())
    snaps = pd.DataFrame(world.snapshots)
    return {"genes": genes, "read_genes": [g for g in genes if g in genes_read(policy)],
            "ranges": {g: getattr(cfg, GENE_RANGES[g]) for g in genes},
            "by_generation": genes_by_generation(lineage, genes),
            "gini": inequality_series(snaps, cfg),
            "pairs": parent_child_pairs(lineage, wealth_at_age(snaps, cfg)),
            "population": pd.DataFrame(world.population_log),
            "minors_by_generation": {int(g): int(n) for g, n in
                                     lineage[lineage["status"] == "minor"].groupby("generation").size().items()},
            "years": cfg.years,
            "max_generation": int(lineage[lineage["status"] == "adult"]["generation"].max())}


def validate_arms(arms: dict) -> None:
    """Every arm runs with births on; an arm that says otherwise would silently be a duplicate baseline."""
    for name, overrides in arms.items():
        if "reproduction" in overrides and not overrides["reproduction"]:
            raise ValueError(f"arm {name!r} sets reproduction=False, but lifesim.generations always runs "
                             f"with births on. Use lifesim.ablate for fixed-population comparisons.")


def run_one(arm: str, overrides: dict, seed: int, agents: int, years: int, policy: str) -> dict:
    validate_arms({arm: overrides})
    cfg = Config(seed=seed, n_agents=agents, years=years,
                 **{"log_every": 365, **overrides, "reproduction": True})
    world = World(cfg, lambda rng, genes: make_policy(policy, rng, genes))
    world.run()
    res = analyze_world(world, policy)
    res.update(arm=arm, seed=seed)
    return res


def _task(args):
    return run_one(*args)


# ---------------------------------------------------------------------------------------- aggregation

def gene_shifts(run: dict, min_n: int = 10) -> pd.DataFrame:
    """Per gene: (mean in the last generation with >= min_n adults) - (founder mean), in range units."""
    bg = run["by_generation"]
    if bg.empty:
        return pd.DataFrame(columns=["gene", "shift", "final_generation"])
    sizes = bg.drop_duplicates("generation").set_index("generation")["n"]
    waiting = run.get("minors_by_generation", {})
    # A generation still has children waiting to come of age is only its earliest-maturing members: skip it.
    ok = [g for g in sizes.index if g > 0 and sizes[g] >= min_n and waiting.get(int(g), 0) == 0]
    if 0 not in sizes.index or not ok:
        return pd.DataFrame(columns=["gene", "shift", "final_generation"])
    final = max(ok)
    rows = []
    for gene in run["genes"]:
        lo, hi = run["ranges"][gene]
        sub = bg[bg["gene"] == gene].set_index("generation")["mean"]
        rows.append({"gene": gene, "shift": float((sub[final] - sub[0]) / (hi - lo)),
                     "final_generation": int(final), "read": gene in run["read_genes"]})
    return pd.DataFrame(rows)


# Two-sided 5% critical values of Student's t by degrees of freedom. With a handful of seeds the normal
# value 1.96 is far too lenient (2.36 at 8 seeds, 12.7 at 2). Between table entries the next lower df is used.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_crit_95(df: int) -> float:
    if df < 1:
        return float("nan")
    keys = [k for k in _T95 if k <= df]
    return _T95[max(keys)] if df <= 120 else 1.96


def _mean_se(x) -> tuple:
    x = np.asarray([v for v in x if v == v], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(se), len(x)


def summarize_arm(runs: list, min_n: int = 10, last_years: float = 50.0) -> pd.DataFrame:
    """Long table of metric, value, se, n_runs for one arm."""
    rows = []

    # -- gene change, and the drift null
    per_run = [(r, gene_shifts(r, min_n)) for r in runs]
    # Which generation the shifts are measured at, so an empty gene section is explained, not silent.
    finals = [float(s["final_generation"].iloc[0]) if not s.empty else float("nan") for _, s in per_run]
    m, se, n = _mean_se(finals)
    rows.append({"metric": "last complete generation used for gene shifts (none: too few or still filling)",
                 "value": m, "se": se, "n_runs": n})
    per_run = [(r, s) for r, s in per_run if not s.empty]
    genes = sorted({g for _, s in per_run for g in s["gene"]})
    paired = {g: [] for g in genes}
    for r, s in per_run:
        neutral = s[~s["read"]]["shift"]
        for _, row in s.iterrows():
            if row["read"] and len(neutral):
                paired[row["gene"]].append(row["shift"] - float(neutral.mean()))
    for g in genes:
        shifts = [float(s[s["gene"] == g]["shift"].iloc[0]) for _, s in per_run]
        is_read = bool(per_run[0][1][per_run[0][1]["gene"] == g]["read"].iloc[0])
        m, se, n = _mean_se(shifts)
        rows.append({"metric": f"shift[{g}]" + ("" if is_read else " (neutral)"), "value": m, "se": se, "n_runs": n})
        if is_read:
            dm, dse, dn = _mean_se(paired[g])
            t = dm / dse if dse and dse == dse else float("nan")
            rows.append({"metric": f"shift[{g}] minus neutral genes", "value": dm, "se": dse, "n_runs": dn})
            # paired t-test over runs of H0: this gene moves like the neutral genes of the same run
            rows.append({"metric": f"t[{g}] vs neutral genes", "value": t, "se": float("nan"), "n_runs": dn})
            rows.append({"metric": f"t_crit_95[{g}] (two-sided, df={dn - 1}): beyond drift if |t| exceeds it",
                         "value": t_crit_95(dn - 1), "se": float("nan"), "n_runs": dn})

    # -- parent-child wealth persistence
    rho = [spearman(r["pairs"]["parent_wealth"], r["pairs"]["child_wealth"]) if len(r["pairs"]) >= 20
           else float("nan") for r in runs]
    m, se, n = _mean_se(rho)
    rows.append({"metric": "spearman(child, parent wealth at 40)", "value": m, "se": se, "n_runs": n})
    rows.append({"metric": "pairs per run", "value": float(np.mean([len(r["pairs"]) for r in runs])),
                 "se": float("nan"), "n_runs": len(runs)})
    both = [float((r["pairs"]["n_parents_known"] == 2).mean()) if len(r["pairs"]) else float("nan") for r in runs]
    m, se, n = _mean_se(both)
    rows.append({"metric": "share of pairs with both parents' wealth known", "value": m, "se": se, "n_runs": n})

    # -- inequality and population
    def late_mean(r, col):
        g = r["gini"]
        if g.empty:
            return float("nan")
        return float(g[g["day"] >= g["day"].max() - last_years * 365][col].mean())
    for col, name in (("gini", f"gini of real net worth (last {last_years:g} years)"),
                      ("share_in_debt", f"share of adults in debt (last {last_years:g} years)")):
        m, se, n = _mean_se([late_mean(r, col) for r in runs])
        rows.append({"metric": name, "value": m, "se": se, "n_runs": n})
    for key, name in (("max_generation", "last generation reached"),):
        m, se, n = _mean_se([r[key] for r in runs])
        rows.append({"metric": name, "value": m, "se": se, "n_runs": n})
    tot = [float(r["population"]["population_total"].iloc[-1]) if len(r["population"]) else float("nan") for r in runs]
    m, se, n = _mean_se(tot)
    rows.append({"metric": "final population (adults + children)", "value": m, "se": se, "n_runs": n})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- CLI

def run_arms(arms: dict, seeds: list, agents: int, years: int, policy: str, workers: int) -> dict:
    validate_arms(arms)
    tasks = [(name, ov, s, agents, years, policy) for name, ov in arms.items() for s in seeds]
    if workers <= 1:
        results = [_task(t) for t in tasks]
    else:
        with mp.Pool(workers) as pool:
            results = pool.map(_task, tasks, chunksize=1)
    out = {name: [] for name in arms}
    for r in results:
        out[r["arm"]].append(r)
    return out


def write_outputs(results: dict, out: Path, min_n: int) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    tables = {"genes_by_generation": [], "gini": [], "parent_child": [], "population": []}
    summary = []
    for arm, runs in results.items():
        s = summarize_arm(runs, min_n)
        s.insert(0, "arm", arm)
        summary.append(s)
        for r in runs:
            for key, src in (("genes_by_generation", "by_generation"), ("gini", "gini"),
                             ("parent_child", "pairs"), ("population", "population")):
                tables[key].append(r[src].assign(arm=arm, seed=r["seed"]))
    for key, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(out / f"{key}.csv", index=False)
    summary = pd.concat(summary, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--agents", type=int, default=100)
    p.add_argument("--years", type=int, default=150)
    p.add_argument("--policy", choices=["rule", "utility"], default="rule")
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--min-generation-size", type=int, default=10,
                   help="smallest number of adults for a generation to count as the final one")
    p.add_argument("--arm", nargs="+", action="append", metavar=("NAME", "KEY=VALUE"),
                   help="an experiment arm: its name, then config overrides (default: baseline, "
                        "no_reserve_gate, assortative). Repeat for more arms.")
    p.add_argument("--out", default="runs/generations")
    a = p.parse_args()
    arms = ({n[0]: parse_overrides(n[1:]) for n in a.arm} if a.arm else DEFAULT_ARMS)
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    print(f"arms={list(arms)} seeds={len(seeds)} agents={a.agents} years={a.years} workers={a.workers}", flush=True)
    try:
        validate_arms(arms)                           # fail before any slow run starts
    except ValueError as e:
        raise SystemExit(str(e)) from e
    results = run_arms(arms, seeds, a.agents, a.years, a.policy, a.workers)
    summary = write_outputs(results, Path(a.out), a.min_generation_size)
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(summary.round(3).to_string(index=False))
    print(f"wrote {a.out}/")


if __name__ == "__main__":
    main()
