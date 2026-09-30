import math

import numpy as np

from . import actions
from .agent import Agent
from .config import Config
from .policy import Policy


class World:
    def __init__(self, cfg: Config, policy_factory):
        """policy_factory(rng) -> Policy; one policy instance shared by all agents."""
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.policy: Policy = policy_factory(self.rng)
        self.day = 0
        lo, hi = cfg.start_age
        self.agents = [
            Agent(id=i,
                  sex=str(self.rng.choice(["M", "F"])),
                  age_days=int(self.rng.integers(lo * 365, hi * 365)),
                  cash=cfg.start_cash)
            for i in range(cfg.n_agents)
        ]
        self.snapshots: list[dict] = []
        self.deaths: list[dict] = []

    # -- one day ---------------------------------------------------------
    def tick(self) -> None:
        cfg = self.cfg
        for a in self.agents:
            if not a.alive:
                continue
            for _ in range(cfg.slots_per_day):
                actions.apply(self.policy.act(a.observe()), a, cfg)
            self._end_of_day(a)
        self.day += 1
        if self.day % cfg.log_every == 0:
            self._snapshot()

    def _end_of_day(self, a: Agent) -> None:
        cfg = self.cfg
        a.age_days += 1
        a.cash -= cfg.daily_living_cost
        a.satiety -= 15
        a.energy += 30           # sleeping
        a.mood += (50 - a.mood) * 0.02   # drift back toward neutral
        # health dynamics
        if a.satiety < 10:
            a.health -= 3
        if a.energy < 10:
            a.health -= 1
        if a.cash < 0:
            a.health -= 1        # money stress
        if a.mood < 20:
            a.health -= 0.5
        if a.satiety > 50 and a.energy > 50:
            a.health += 0.3
        if a.age > 40:
            a.health -= (a.age - 40) * 0.0002
        a.clamp()
        # death
        if a.health <= 0:
            self._die(a, "health")
        else:
            annual = (cfg.gompertz_a * math.exp(cfg.gompertz_b * a.age)
                      * cfg.sex_mortality_mult[a.sex]
                      * (1 + (100 - a.health) / 25))
            if self.rng.random() < 1 - math.exp(-annual / 365):
                self._die(a, "age")

    def _die(self, a: Agent, cause: str) -> None:
        a.alive = False
        self.deaths.append({"day": self.day, "id": a.id, "sex": a.sex,
                            "age": a.age, "cause": cause, "cash": a.cash,
                            "education": a.education, "career": a.career})

    def _snapshot(self) -> None:
        for a in self.agents:
            if a.alive:
                self.snapshots.append({
                    "day": self.day, "id": a.id, "sex": a.sex, "age": a.age,
                    "health": a.health, "energy": a.energy,
                    "satiety": a.satiety, "mood": a.mood, "cash": a.cash,
                    "education": a.education, "career": a.career})

    def run(self) -> None:
        for _ in range(self.cfg.days):
            self.tick()
            if not any(a.alive for a in self.agents):
                break
