"""A policy maps an observation vector to an Action. Swap in an RL net later."""
from typing import Protocol

import numpy as np

from .actions import N_ACTIONS, Action
from .agent import OBS


class Policy(Protocol):
    def act(self, obs: np.ndarray) -> Action: ...


class RandomPolicy:
    """Baseline: if RL cannot beat this, it learned nothing."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, obs: np.ndarray) -> Action:
        return Action(int(self.rng.integers(N_ACTIONS)))


class RulePolicy:
    """Hand-coded needs-first routine: survive, then earn, then improve."""

    def __init__(self, study_frac: float = 0.3, cash_buffer: float = 150.0,
                 rng: np.random.Generator | None = None):
        self.study_frac = study_frac
        self.cash_buffer = cash_buffer
        self.rng = rng or np.random.default_rng()

    def act(self, obs: np.ndarray) -> Action:
        if obs[OBS["satiety"]] < 40:
            return Action.EAT
        if obs[OBS["energy"]] < 30:
            return Action.REST
        if obs[OBS["mood"]] < 30:
            return Action.SOCIALIZE
        if obs[OBS["cash"]] < self.cash_buffer:
            return Action.WORK
        if self.rng.random() < self.study_frac:
            return Action.STUDY
        return Action.WORK


def make_policy(name: str, rng: np.random.Generator, genes: dict | None = None) -> Policy:
    genes = genes or {}
    if name == "random":
        return RandomPolicy(rng)
    if name == "rule":
        return RulePolicy(rng=rng, **genes)
    raise ValueError(f"unknown policy {name!r}")
