"""Held-out comparison: learned network vs rule / random (/ utility) on fresh seeds.

python -m lifesim.rl.evaluate --theta runs/rl/best.npz --seeds 20 --agents 40 --years 30
"""
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

from ..config import Config
from ..parallel import default_workers
from ..policy import make_policy
from ..run import parse_overrides
from ..world import World
from .es import check_supported
from .mlp import load
from .reward import Reward

_G: dict = {}


def _one(args):
    name, seed = args
    cfg = Config(seed=seed, n_agents=_G["agents"], years=_G["years"], **_G["overrides"])
    if name == "mlp":
        brain = load(_G["theta"])
        factory = lambda rng, genes: brain  # noqa: E731
    else:
        factory = lambda rng, genes: make_policy(name, rng, genes)  # noqa: E731
    w = World(cfg, factory)
    w.run()
    outcomes = w.outcomes()
    o = pd.DataFrame(outcomes)
    ev = pd.DataFrame(w.events)
    ill = ev[ev.event == "illness"] if len(ev) else ev
    parts = {f"reward_{k}": float(v.mean()) for k, v in _G["reward"].components(outcomes, cfg.years).items()}
    return {"policy": name, "seed": seed,
            "reward": float(_G["reward"].per_agent(outcomes, cfg.years).mean()), **parts,
            "alive": o.alive.mean(),
            "restricted_mean_years": o.years_lived.mean(),   # E[min(T, horizon)], not life expectancy
           
            "median_cash": o.cash.median(), "mean_health": o.health.mean(),
            "illness_treated": ill.treated.mean() if len(ill) else np.nan}


def _init(theta, agents, years, overrides, reward):
    _G.update(theta=theta, agents=agents, years=years, overrides=overrides, reward=reward)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--theta", required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--agents", type=int, default=40)
    p.add_argument("--years", type=int, default=30)
    p.add_argument("--policies", nargs="+", default=["mlp", "rule", "random"])
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--reward", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", default=None, help="write the per-seed table to this CSV")
    a = p.parse_args()
    reward = Reward(**{k: float(v) for k, v in (kv.split("=") for kv in a.reward)})
    try:
        check_supported(parse_overrides(a.set))
    except ValueError as e:
        raise SystemExit(str(e)) from e
    seeds = [9_000_000 + i for i in range(a.seeds)]       # never seen in training
    jobs = [(name, s) for name in a.policies for s in seeds]
    with mp.Pool(a.workers, initializer=_init,
                 initargs=(a.theta, a.agents, a.years, parse_overrides(a.set), reward)) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    summary = df.groupby("policy").agg(["mean", "std"]).drop(columns="seed")
    print(summary.round(3).to_string())
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
