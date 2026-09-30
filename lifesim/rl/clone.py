"""Warm start: copy the rule policy into the network so ES begins at a policy
that already survives, instead of from random weights that all die in a year."""
import numpy as np

from ..config import Config
from ..policy import make_policy
from ..world import World
from .mlp import MLPPolicy, featurize


class _Recorder:
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def act(self, obs):
        a = self.inner.act(obs)
        self.sink.append((featurize(obs), int(a)))
        return a


def collect(cfg_kwargs: dict, n_worlds: int, seed0: int = 50_000):
    sink: list = []
    for i in range(n_worlds):
        cfg = Config(seed=seed0 + i, **cfg_kwargs)
        World(cfg, lambda rng, genes: _Recorder(make_policy("rule", rng, genes), sink)).run()
    X = np.array([s[0] for s in sink])
    y = np.array([s[1] for s in sink])
    return X, y


def clone_rule_policy(cfg_kwargs: dict, n_worlds=10, epochs=30, hidden=(32, 32), seed=0):
    X, y = collect(cfg_kwargs, n_worlds)
    pol = MLPPolicy(hidden, rng=np.random.default_rng(seed))
    losses = pol.fit(X, y, epochs=epochs, rng=np.random.default_rng(seed))
    acc = float((np.argmax(pol.logits(X), 1) == y).mean())
    return pol, losses, acc
