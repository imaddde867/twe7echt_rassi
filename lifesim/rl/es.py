"""Evolution strategies (OpenAI-ES): antithetic sampling, rank shaping, Adam.

Every candidate network is tried as the shared brain of a whole world of agents;
fitness is the mean reward of those agents. All candidates in a generation see
the same world seeds (common random numbers) so differences come from the
network, not from luck. Worlds are independent, so this scales across cores.
"""
import argparse
import csv
import dataclasses
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..guards import require_fixed_population
from ..run import parse_overrides
from ..world import World
from . import clone
from .mlp import MLPPolicy, save
from .reward import Reward

_G: dict = {}


def check_supported(cfg_kwargs: dict) -> None:
    """The reward assumes a fixed population. With reproduction, descendants exist for only part of
    the horizon (so a whole-horizon `years_lived` penalises them) and the dead keep cash that was
    already passed on in an estate. Until a dynamic-population reward exists, refuse."""
    require_fixed_population(cfg_kwargs, "the ES trainer or evaluator",
                             "Run generations with lifesim.run instead.")


def _init(cfg_kwargs, reward, hidden):
    _G.update(cfg_kwargs=cfg_kwargs, reward=reward, hidden=hidden)


def evaluate(theta: np.ndarray, seeds: list[int], cfg_kwargs=None, reward=None, hidden=None) -> dict:
    cfg_kwargs = cfg_kwargs if cfg_kwargs is not None else _G["cfg_kwargs"]
    check_supported(cfg_kwargs)
    reward = reward or _G["reward"]
    hidden = hidden or _G["hidden"]
    policy = MLPPolicy.from_flat(theta, hidden)
    rs, alive, yrs = [], [], []
    for s in seeds:
        cfg = Config(seed=s, **cfg_kwargs)
        w = World(cfg, lambda rng, genes: policy)   # one shared brain
        w.run()
        o = w.outcomes()
        rs.append(reward.per_agent(o, cfg.years).mean())
        alive.append(np.mean([x["alive"] for x in o]))
        yrs.append(np.mean([x["years_lived"] for x in o]))
    return {"fitness": float(np.mean(rs)), "alive": float(np.mean(alive)),
            "years": float(np.mean(yrs))}


def _task(args):
    theta, seeds = args
    return evaluate(theta, seeds)["fitness"]


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Rank-shape to [-0.5, 0.5]. Ties share their average rank, so equal fitness gets equal
    reward (no fake gradient from arbitrary tie-breaking); all-equal gives all zeros."""
    x = np.asarray(x, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    avg = np.cumsum(counts) - (counts + 1) / 2.0   # mean 0-based rank of each tie group
    return avg[inv.reshape(-1)] / (len(x) - 1) - 0.5


def default_workers() -> int:
    try:
        return len(os.sched_getaffinity(0))   # respects the Slurm allocation
    except AttributeError:
        return os.cpu_count() or 1


def gen_seeds(gen: int, n: int) -> list[int]:
    """World seeds for one generation: shared by every candidate in it."""
    return [1_000_000 + gen * 100 + j for j in range(n)]


class _Serial:
    """Stand-in for a Pool when workers <= 1 (tests, tiny runs)."""

    def __init__(self, initargs):
        _init(*initargs)

    def map(self, fn, items, chunksize=1):
        return [fn(x) for x in items]

    def apply(self, fn, args):
        return fn(*args)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _pool(workers, initargs):
    return _Serial(initargs) if workers <= 1 else mp.Pool(workers, initializer=_init, initargs=initargs)


def _rng_state(rng) -> str:
    return json.dumps(rng.bit_generator.state)


def _set_rng_state(rng, blob: str) -> None:
    rng.bit_generator.state = json.loads(blob)


def _manifest(cfg_kwargs, reward, hidden, pairs, seeds_per_eval, sigma, lr, weight_decay) -> dict:
    """Everything that gives fitness and the stored optimiser state their meaning.
    generations, workers, clone_worlds and seed are deliberately absent: changing them on resume
    is safe. Round-tripped through JSON so tuples and lists compare equal."""
    m = {"cfg": cfg_kwargs, "reward": dataclasses.asdict(reward), "hidden": list(hidden),
         "pairs": pairs, "seeds_per_eval": seeds_per_eval, "sigma": sigma, "lr": lr,
         "weight_decay": weight_decay}
    return json.loads(json.dumps(m, sort_keys=True, default=list))


def _check_manifest(path: Path, new: dict) -> None:
    if not path.exists():
        raise SystemExit(f"cannot resume: {path} is missing, so the checkpoint's config is unknown")
    old = json.loads(path.read_text())
    diff = [f"{k}: checkpoint={old.get(k)!r} now={new.get(k)!r}"
            for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
    if diff:
        raise SystemExit("cannot resume with a different training setup (the stored fitness would "
                         "not be comparable):\n  " + "\n  ".join(diff))


def train(out, cfg_kwargs, reward, hidden=(32, 32), generations=200, pairs=32, seeds_per_eval=2,
          sigma=0.05, lr=0.02, weight_decay=0.005, workers=1, clone_worlds=10, seed=0,
          resume=False, say=print, on_generation=None) -> dict:
    """Run ES. `generations` is the total to reach, so resuming continues towards it."""
    check_supported(cfg_kwargs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hidden = tuple(hidden)
    rng = np.random.default_rng(seed)
    latest, best_path = out / "latest.npz", out / "best.npz"
    manifest_path = out / "manifest.json"
    manifest = _manifest(cfg_kwargs, reward, hidden, pairs, seeds_per_eval, sigma, lr, weight_decay)

    gen0, m, v, t = 0, None, None, 0
    best_fit, best_theta, best_gen = -np.inf, None, -1
    if resume and latest.exists():
        _check_manifest(manifest_path, manifest)
        d = np.load(latest)
        theta, m, v, t, gen0 = d["theta"], d["m"], d["v"], int(d["t"]), int(d["gen"]) + 1
        _set_rng_state(rng, str(d["rng_state"]))         # continue the same perturbation sequence
        if best_path.exists():                           # best.npz is the source of truth for the best
            b = np.load(best_path)
            best_fit, best_theta, best_gen = float(b["fitness"]), b["theta"].copy(), int(b["gen"])
        say(f"resumed at generation {gen0}; best so far {best_fit:.4f} (generation {best_gen})")
    elif clone_worlds > 0:
        t0 = time.time()
        pol, losses, acc = clone.clone_rule_policy(cfg_kwargs, clone_worlds, hidden=hidden, seed=seed)
        theta = pol.get_flat()
        say(f"behaviour cloning: loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
            f"accuracy {acc:.2f}, {time.time() - t0:.0f}s")
    else:
        theta = MLPPolicy(hidden, rng=rng).get_flat()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    n = len(theta)
    m = np.zeros(n) if m is None else m
    v = np.zeros(n) if v is None else v

    log_path = out / "log.csv"
    new_log = not (resume and log_path.exists())
    with open(log_path, "w" if new_log else "a", newline="") as logf:
        log = csv.writer(logf)
        if new_log:
            log.writerow(["gen", "theta_fitness", "theta_alive", "theta_years",
                          "pop_mean", "pop_max", "seconds"])
        with _pool(workers, (cfg_kwargs, reward, hidden)) as pool:
            for gen in range(gen0, generations):
                t0 = time.time()
                seeds = gen_seeds(gen, seeds_per_eval)
                eps = rng.normal(size=(pairs, n))
                cands = np.concatenate([theta + sigma * eps, theta - sigma * eps])
                fits = np.array(pool.map(_task, [(c, seeds) for c in cands], chunksize=1))
                evaluated = theta.copy()                 # exactly the weights that get scored below
                ref = pool.apply(evaluate, (evaluated, seeds))
                if on_generation:
                    on_generation(gen, evaluated.copy(), ref)
                if ref["fitness"] > best_fit:
                    best_fit, best_theta, best_gen = ref["fitness"], evaluated, gen
                    save(best_path, best_theta, hidden, fitness=best_fit, gen=best_gen)
                # gradient estimate from rank-shaped fitness, then an Adam ascent step
                shaped = centered_ranks(fits)
                grad = (shaped[:pairs] - shaped[pairs:]) @ eps / (2 * pairs * sigma)
                grad -= weight_decay * theta
                t += 1
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad * grad
                theta = theta + lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

                log.writerow([gen, f"{ref['fitness']:.4f}", f"{ref['alive']:.3f}", f"{ref['years']:.2f}",
                              f"{fits.mean():.4f}", f"{fits.max():.4f}", f"{time.time() - t0:.1f}"])
                logf.flush()
                say(f"gen {gen:4d}  fitness {ref['fitness']:.3f}  alive {ref['alive']:.2f}  "
                    f"years {ref['years']:.1f}  pop max {fits.max():.3f}  {time.time() - t0:.0f}s")
                save(latest, theta, hidden, m=m, v=v, t=t, gen=gen,
                     rng_state=_rng_state(rng), best_fitness=best_fit, best_gen=best_gen)
    say(f"done. best noisy fitness {best_fit:.3f} at generation {best_gen}. Evaluate on held-out seeds "
        f"with python -m lifesim.rl.evaluate --theta {out}/best.npz")
    return {"best_fitness": best_fit, "best_gen": best_gen, "theta": theta}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/rl")
    p.add_argument("--generations", type=int, default=200, help="total to reach (resume continues towards it)")
    p.add_argument("--pairs", type=int, default=32, help="antithetic pairs (population = 2x)")
    p.add_argument("--seeds-per-eval", type=int, default=2, help="worlds per candidate")
    p.add_argument("--agents", type=int, default=20)
    p.add_argument("--years", type=int, default=20)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.005)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--clone-worlds", type=int, default=10, help="0 = random init")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Config overrides, e.g. start_age=\"(18,25)\"")
    p.add_argument("--reward", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Reward weights, e.g. w_wealth=1.0")
    a = p.parse_args()
    cfg_kwargs = {"n_agents": a.agents, "years": a.years, **parse_overrides(a.set)}
    reward = Reward(**{k: float(v) for k, v in (kv.split("=") for kv in a.reward)})
    print(f"workers={a.workers} pairs={a.pairs} seeds/eval={a.seeds_per_eval} "
          f"agents={a.agents} years={a.years}\nreward={reward}", flush=True)
    train(a.out, cfg_kwargs, reward, a.hidden, a.generations, a.pairs, a.seeds_per_eval, a.sigma,
          a.lr, a.weight_decay, a.workers, a.clone_worlds, a.seed, a.resume,
          say=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
