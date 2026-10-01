"""python -m lifesim.run --agents 100 --years 50 --policy rule --out runs/demo"""
import argparse
import ast
from dataclasses import fields
from pathlib import Path

import pandas as pd

from .config import Config
from .plot import make_plots
from .policy import make_policy
from .world import World

# Set by dedicated CLI flags in every entry point, so --set must not also carry them.
RESERVED = {"seed": "--seed/--seeds", "n_agents": "--agents", "years": "--years"}


def parse_overrides(pairs: list[str]) -> dict:
    """['retire_age=70', 'life_stages=False'] -> {'retire_age': 70, 'life_stages': False}"""
    valid = {f.name for f in fields(Config)}
    out = {}
    for pair in pairs:
        key, _, val = pair.partition("=")
        if key in RESERVED:
            raise SystemExit(f"{key!r} has its own flag ({RESERVED[key]}); do not pass it via --set")
        if key not in valid:
            raise SystemExit(f"unknown config field {key!r}; valid: {sorted(valid)}")
        out[key] = ast.literal_eval(val)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=100)
    p.add_argument("--years", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=["random", "rule", "utility", "mlp"], default="rule")
    p.add_argument("--theta", default=None, help="trained network (.npz) for --policy mlp")
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override a Config field (not seed/agents/years, which have flags), e.g. --set retire_age=70 life_stages=False")
    a = p.parse_args()

    cfg = Config(seed=a.seed, n_agents=a.agents, years=a.years, **parse_overrides(a.set))
    if a.policy == "mlp":
        from .rl.mlp import load
        brain = load(a.theta)                     # one shared brain for every agent
        factory = lambda rng, genes: brain        # noqa: E731
    else:
        factory = lambda rng, genes: make_policy(a.policy, rng, genes)  # noqa: E731
    world = World(cfg, factory)
    world.run()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    snaps, deaths = pd.DataFrame(world.snapshots), pd.DataFrame(world.deaths)
    snaps.to_csv(out / "snapshots.csv", index=False)
    deaths.to_csv(out / "deaths.csv", index=False)
    outcomes = pd.DataFrame(world.outcomes())
    outcomes.to_csv(out / "outcomes.csv", index=False)
    pd.DataFrame(world.events).to_csv(out / "events.csv", index=False)
    make_plots(snaps, deaths, outcomes, out / "summary.png", title=f"policy={a.policy}")

    alive = sum(x.alive for x in world.agents)
    print(f"policy={a.policy} seed={a.seed}: {len(deaths)} died, {alive} alive "
          f"after {world.day / 365:.1f} years")
    if len(deaths):
        print(f"median age at death: {deaths['age'].median():.1f}; "
              f"causes: {deaths['cause'].value_counts().to_dict()}")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
