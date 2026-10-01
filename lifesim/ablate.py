"""Compare a baseline config against an override over several seeds.

python -m lifesim.ablate --set life_stages=False   # vs default (life_stages=True)
"""
import argparse

import numpy as np
import pandas as pd

from .config import Config
from .policy import make_policy
from .run import parse_overrides
from .world import World


def summarize(cfg_kwargs: dict, seeds: int, agents: int, years: int, policy: str) -> dict:
    rows = []
    for seed in range(seeds):
        w = World(Config(seed=seed, n_agents=agents, years=years, **cfg_kwargs),
                  lambda rng, genes: make_policy(policy, rng, genes))
        w.run()
        rows.append(pd.DataFrame(w.outcomes()))
    o = pd.concat(rows)
    dead = o[~o["alive"]]
    return {"agents": len(o), "alive_frac": o["alive"].mean(),
            # Restricted mean survival: estimates E[min(T, horizon)], the mean years lived
            # inside the simulated window. It is NOT the unrestricted life expectancy.
            "restricted_mean_years": o["years_lived"].mean(),
            "median_age_at_death": dead["age"].median() if len(dead) else np.nan,
            "median_final_cash": o["cash"].median(),
            "median_wealth_per_year": o["wealth_per_year"].median(),
            "illnesses_per_agent": o["n_illness"].mean(),
            "job_losses_per_agent": o["n_job_losses"].mean(),
            "chronic_frac": o["chronic"].mean()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", nargs="+", required=True, metavar="KEY=VALUE")
    p.add_argument("--base", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--agents", type=int, default=60)
    p.add_argument("--years", type=int, default=50)
    p.add_argument("--policy", default="rule")
    a = p.parse_args()
    base, alt = parse_overrides(a.base), {**parse_overrides(a.base), **parse_overrides(a.set)}
    res = pd.DataFrame({"base": summarize(base, a.seeds, a.agents, a.years, a.policy),
                        "override": summarize(alt, a.seeds, a.agents, a.years, a.policy)})
    print(f"override: {a.set}  ({a.seeds} seeds x {a.agents} agents x {a.years} y)")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
