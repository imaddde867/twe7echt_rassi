"""Compare a baseline config against an override over several seeds.

python -m lifesim.ablate --set life_stages=False   # vs default (life_stages=True)
"""
import argparse

import numpy as np
import pandas as pd

from .config import Config
from .guards import require_fixed_population
from .policy import make_policy
from .run import parse_overrides
from .world import World


def summarize(cfg_kwargs: dict, seeds: int, agents: int, years: int, policy: str) -> dict:
    require_fixed_population(cfg_kwargs, "lifesim.ablate",
                             "Use the per-run outputs of lifesim.run (lineage.csv, population.csv) instead.")
    rows = []
    for seed in range(seeds):
        w = World(Config(seed=seed, n_agents=agents, years=years, **cfg_kwargs),
                  lambda rng, genes: make_policy(policy, rng, genes))
        w.run()
        o_seed = pd.DataFrame(w.outcomes())
        # Money is reported the same way in every arm, whatever its money convention: nominal (the legacy
        # figures, unchanged) and real (day-0 money, each agent valued at its own death-day or final prices).
        # An arm with real_money on and one with it off are then comparable.
        p_end = pd.Series([a.end_price_idx if a.end_price_idx is not None else w.prices.price_idx
                           for a in w.agents], index=o_seed.index)
        start = pd.Series([a.start_cash for a in w.agents], index=o_seed.index)
        years_lived = o_seed["years_lived"].clip(lower=1e-9)
        o_seed["wealth_nominal"] = (o_seed["cash"] - start) / years_lived
        o_seed["cash_real_all"] = o_seed["cash"] / p_end
        o_seed["wealth_real"] = (o_seed["cash_real_all"] - start) / years_lived
        rows.append(o_seed)
    o = pd.concat(rows)
    dead = o[~o["alive"]]
    return {"agents": len(o), "alive_frac": o["alive"].mean(),
            # Restricted mean survival: estimates E[min(T, horizon)], the mean years lived
            # inside the simulated window. It is NOT the unrestricted life expectancy.
            "restricted_mean_years": o["years_lived"].mean(),
            "median_age_at_death": dead["age"].median() if len(dead) else np.nan,
            "median_final_cash": o["cash"].median(),                     # nominal
            "median_wealth_per_year": o["wealth_nominal"].median(),      # nominal
            "median_final_cash_real": o["cash_real_all"].median(),       # day-0 money
            "median_wealth_per_year_real": o["wealth_real"].median(),     # day-0 money
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
    try:
        for cfg_kwargs in (base, alt):             # check both up front: fail before the slow base run
            require_fixed_population(cfg_kwargs, "lifesim.ablate",
                                     "Use the per-run outputs of lifesim.run (lineage.csv, population.csv) instead.")
    except ValueError as e:
        raise SystemExit(str(e)) from e
    res = pd.DataFrame({"base": summarize(base, a.seeds, a.agents, a.years, a.policy),
                        "override": summarize(alt, a.seeds, a.agents, a.years, a.policy)})
    print(f"override: {a.set}  ({a.seeds} seeds x {a.agents} agents x {a.years} y)")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
