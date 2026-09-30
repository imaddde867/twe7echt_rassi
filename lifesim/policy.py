"""A policy maps an observation vector to an Action. Swap in an RL net later."""
import math
from typing import Protocol

import numpy as np

from .actions import N_ACTIONS, Action
from .agent import OBS, Agent
from .actions import apply


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
        if obs[OBS["sick"]]:
            return Action.REST
        if obs[OBS["unemployed"]]:
            return Action.STUDY      # use the time off; cannot work
        if obs[OBS["retired"]]:
            return Action.SOCIALIZE if obs[OBS["mood"]] < 70 else Action.REST
        if obs[OBS["cash"]] < self.cash_buffer:
            return Action.WORK
        if self.rng.random() < self.study_frac:
            return Action.STUDY
        return Action.WORK


class UtilityPolicy:
    """One-step lookahead: try every action on a scratch copy of yourself and
    pick the one leading to the best state, as scored by a utility function.

    No hand-written "if hungry then eat": eating wins when hunger is costly
    enough. The weights are genes, so agents differ in what they value:
      patience        how much future-oriented stats (education, career) count
      wealth_weight   how much money counts (concave: the first euro matters most)
    Needs `bind(world)` for the config and live prices.
    """

    def __init__(self, patience: float = 1.0, wealth_weight: float = 1.0,
                 rng: np.random.Generator | None = None):
        self.patience = patience
        self.wealth_weight = wealth_weight
        self.rng = rng or np.random.default_rng()
        self.cfg = None
        self.prices = None

    def bind(self, world) -> None:
        self.cfg, self.prices = world.cfg, world.prices

    def utility(self, s: Agent) -> float:
        cfg, p = self.cfg, self.prices
        # deficits are penalised quadratically: a little hunger is cheap, starving is not
        needs = -(40 * (max(0.0, 50 - s.satiety) / 50) ** 2
                  + 20 * (max(0.0, 35 - s.energy) / 35) ** 2
                  + 8 * (max(0.0, 30 - s.mood) / 30) ** 2)
        horizon = max(0.0, cfg.retire_age - s.age) / (cfg.retire_age - 18)
        work_days = horizon * (cfg.retire_age - 18) * 365
        scale = 1000.0 * p.price_idx
        cash = max(0.0, s.cash)
        # cash matters more as retirement approaches; concave, so the first euros matter most
        money_w = 10.0 * self.wealth_weight * (0.5 + (1 - horizon))
        # human capital: future pay from education and career, valued at today's
        # marginal utility of cash (linearised, so it does not flatten the cash term)
        human_capital = 0.5 * work_days * (1.0 * s.education + 0.4 * s.career) * p.wage_idx
        money = money_w * (math.log1p(cash / scale)
                           + self.patience * human_capital / (scale + cash))
        # A one-step planner cannot see daily costs coming, so give it an emergency
        # fund target (days of spending, scaled by wealth_weight) and a wall at the
        # credit limit. Below the fund, every missing euro hurts.
        burn = (cfg.daily_living_cost + cfg.eat_cost) * p.price_idx
        fund = 40 * self.wealth_weight * burn
        wall = cfg.credit_limit * p.price_idx
        debt = (0.4 * max(0.0, fund - s.cash)
                + 30.0 * (max(0.0, -s.cash - 0.25 * wall) / (0.75 * wall)) ** 2)
        return needs + 0.05 * s.health + 0.01 * s.mood + money - debt

    def act(self, obs: np.ndarray) -> Action:
        best, best_u = Action.REST, -math.inf
        for action in Action:
            s = Agent.from_obs(obs)
            apply(action, s, self.cfg, self.prices)
            u = self.utility(s)
            if u > best_u:
                best, best_u = action, u
        return best


_GENES = {"rule": ("study_frac", "cash_buffer"),
          "utility": ("patience", "wealth_weight")}


def make_policy(name: str, rng: np.random.Generator, genes: dict | None = None) -> Policy:
    genes = {k: v for k, v in (genes or {}).items() if k in _GENES.get(name, ())}
    if name == "random":
        return RandomPolicy(rng)
    if name == "rule":
        return RulePolicy(rng=rng, **genes)
    if name == "utility":
        return UtilityPolicy(rng=rng, **genes)
    raise ValueError(f"unknown policy {name!r}")
