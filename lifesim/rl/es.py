"""Evolution strategies (OpenAI-ES): antithetic sampling, rank shaping, Adam.

Every candidate network is tried as the shared brain of a whole world of agents;
fitness is the mean reward of those agents. All candidates in a generation see
the same world seeds (common random numbers) so differences come from the
network, not from luck. Worlds are independent, so this scales across cores.
"""
import argparse
import csv
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..run import parse_overrides
from ..world import World
from . import clone
from .mlp import MLPPolicy, save
from .reward import Reward

_G: dict = {}


def _init(cfg_kwargs, reward, hidden):
    _G.update(cfg_kwargs=cfg_kwargs, reward=reward, hidden=hidden)


def evaluate(theta: np.ndarray, seeds: list[int], cfg_kwargs=None, reward=None, hidden=None) -> dict:
    cfg_kwargs = cfg_kwargs if cfg_kwargs is not None else _G["cfg_kwargs"]
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
    r = np.empty(len(x))
    r[x.argsort()] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def default_workers() -> int:
    try:
        return len(os.sched_getaffinity(0))   # respects the Slurm allocation
    except AttributeError:
        return os.cpu_count() or 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/rl")
    p.add_argument("--generations", type=int, default=200)
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

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hidden = tuple(a.hidden)
    cfg_kwargs = {"n_agents": a.agents, "years": a.years, **parse_overrides(a.set)}
    reward = Reward(**{k: float(v) for k, v in (kv.split("=") for kv in a.reward)})
    rng = np.random.default_rng(a.seed)
    print(f"workers={a.workers} pairs={a.pairs} seeds/eval={a.seeds_per_eval} "
          f"agents={a.agents} years={a.years}\nreward={reward}", flush=True)

    # ---- init / resume ---------------------------------------------------
    gen0, m, v, t = 0, None, None, 0
    latest = out / "latest.npz"
    if a.resume and latest.exists():
        d = np.load(latest)
        theta, m, v, t, gen0 = d["theta"], d["m"], d["v"], int(d["t"]), int(d["gen"]) + 1
        rng = np.random.default_rng(a.seed + gen0)
        print(f"resumed at generation {gen0}", flush=True)
    elif a.clone_worlds > 0:
        t0 = time.time()
        pol, losses, acc = clone.clone_rule_policy(cfg_kwargs, a.clone_worlds, hidden=hidden, seed=a.seed)
        theta = pol.get_flat()
        print(f"behaviour cloning: loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
              f"accuracy {acc:.2f}, {time.time() - t0:.0f}s", flush=True)
    else:
        theta = MLPPolicy(hidden, rng=rng).get_flat()
    n = len(theta)
    m = np.zeros(n) if m is None else m
    v = np.zeros(n) if v is None else v

    log_path = out / "log.csv"
    new_log = not (a.resume and log_path.exists())
    logf = open(log_path, "a" if not new_log else "w", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["gen", "theta_fitness", "theta_alive", "theta_years",
                      "pop_mean", "pop_max", "seconds"])
    best = -np.inf

    # ---- main loop -----------------------------------------------------------
    with mp.Pool(a.workers, initializer=_init, initargs=(cfg_kwargs, reward, hidden)) as pool:
        for gen in range(gen0, a.generations):
            t0 = time.time()
            seeds = [1_000_000 + gen * 100 + j for j in range(a.seeds_per_eval)]   # shared by all candidates
            eps = rng.normal(size=(a.pairs, n))
            cands = np.concatenate([theta + a.sigma * eps, theta - a.sigma * eps])
            fits = np.array(pool.map(_task, [(c, seeds) for c in cands], chunksize=1))
            ref = pool.apply(evaluate, (theta, seeds))
            # gradient estimate from rank-shaped fitness
            shaped = centered_ranks(fits)
            grad = (shaped[:a.pairs] - shaped[a.pairs:]) @ eps / (2 * a.pairs * a.sigma)
            grad -= a.weight_decay * theta
            t += 1                                   # Adam ascent step
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            theta = theta + a.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

            log.writerow([gen, f"{ref['fitness']:.4f}", f"{ref['alive']:.3f}", f"{ref['years']:.2f}",
                          f"{fits.mean():.4f}", f"{fits.max():.4f}", f"{time.time() - t0:.1f}"])
            logf.flush()
            print(f"gen {gen:4d}  fitness {ref['fitness']:.3f}  alive {ref['alive']:.2f}  "
                  f"years {ref['years']:.1f}  pop max {fits.max():.3f}  {time.time() - t0:.0f}s", flush=True)
            save(latest, theta, hidden, m=m, v=v, t=t, gen=gen)
            if ref["fitness"] > best:
                best = ref["fitness"]
                save(out / "best.npz", theta, hidden, fitness=best, gen=gen)
    logf.close()
    print(f"done. best noisy fitness {best:.3f}. Evaluate on held-out seeds with "
          f"python -m lifesim.rl.evaluate --theta {out}/best.npz", flush=True)


if __name__ == "__main__":
    main()
