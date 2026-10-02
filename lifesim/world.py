import math

import numpy as np

from . import actions, economy, family, shocks
from . import rng as R
from .agent import Agent
from .config import Config
from .policy import Policy


class World:
    def __init__(self, cfg: Config, policy_factory):
        """policy_factory(rng, genes) -> Policy, called once per agent.

        Return the same object every time for a shared policy (e.g. one RL net).
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)       # initial conditions only
        self.policy_rng = np.random.default_rng([cfg.seed, 2])   # what policies may consume
        self.env = R.EnvRng(cfg.seed)                    # all exogenous events; see rng.py
        self.day = 0
        self.prices = economy.Prices()
        lo, hi = cfg.start_age
        ages = self.rng.integers(lo * 365, hi * 365, size=cfg.n_agents)
        self.agents = [
            Agent(id=i,
                  sex=str(self.rng.choice(["M", "F"])),
                  age_days=int(ages[i]), start_age_days=int(ages[i]),
                  cash=cfg.start_cash, start_cash=cfg.start_cash,
                  genes={"study_frac": float(self.rng.uniform(*cfg.study_frac_range)),
                         "cash_buffer": float(self.rng.uniform(*cfg.cash_buffer_range))})
            for i in range(cfg.n_agents)
        ]
        # extra genes come from their own stream so older runs keep their numbers
        gene_rng = np.random.default_rng([cfg.seed, 1])
        for a in self.agents:
            a.education = float(gene_rng.uniform(*cfg.start_education))
            a.genes["patience"] = float(gene_rng.uniform(*cfg.patience_range))
            a.genes["wealth_weight"] = float(gene_rng.uniform(*cfg.wealth_weight_range))
        if cfg.economy:
            for a in self.agents:
                a.job_level = economy.job_level_for(a, cfg)
        self._policy_factory = policy_factory
        self.policies: dict[int, Policy] = {}
        self._bound: set[int] = set()
        self.by_id: dict[int, Agent] = {}
        self._dying: list[Agent] = []                # deaths awaiting estate settlement
        self.pending: list = []                      # queued children (see family.py)
        self.next_id = len(self.agents)
        for a in self.agents:
            self._attach_policy(a)
        self.snapshots: list[dict] = []
        self.deaths: list[dict] = []
        self.events: list[dict] = []
        self.population_log: list[dict] = []        # reproduction only: adults vs queued children over time

    def _attach_policy(self, a: Agent) -> None:
        pol = self._policy_factory(self.policy_rng, a.genes)
        self.policies[a.id] = pol
        self.by_id[a.id] = a
        if hasattr(pol, "bind") and id(pol) not in self._bound:
            self._bound.add(id(pol))
            pol.bind(self)

    def register(self, a: Agent) -> None:
        """Add an agent born during the run (a child reaching adulthood)."""
        self.agents.append(a)
        self._attach_policy(a)

    # -- one day ---------------------------------------------------------
    def tick(self) -> None:
        cfg = self.cfg
        self.env.begin_day(self.day)
        if cfg.economy:
            economy.update_prices(self.prices, cfg, self.day)
        for a in self.agents:
            if not a.alive:
                continue
            for _ in range(cfg.slots_per_day):
                actions.apply(self.policies[a.id].act(a.observe()), a, cfg, self.prices)
            self._end_of_day(a)
        if cfg.reproduction:
            self.settle_deaths()
            family.step(self)
        self.day += 1
        if self.day % cfg.log_every == 0:
            self._snapshot()

    def _end_of_day(self, a: Agent) -> None:
        cfg = self.cfg
        a.age_days += 1
        if not cfg.economy:
            a.cash -= cfg.daily_living_cost * economy.household_factor(a, cfg)
        if cfg.life_stages:
            if not a.retired and a.age >= cfg.retire_age:
                a.retired = True
                active_days = max(1, a.age_days - a.start_age_days)
                a.pension_daily = cfg.pension_frac * a.lifetime_earnings / active_days
                a.pension_real_daily = cfg.pension_frac * a.lifetime_earnings_real / active_days
            if cfg.uses_real_money and a.retired:
                # real-money mode: based on real career earnings and indexed to prices all through
                # retirement, so a retiree does not lose a third of their income to 20 years of inflation
                a.pension_daily = a.pension_real_daily * self.prices.price_idx
            a.cash += a.pension_daily
        if cfg.economy:
            economy.end_of_day(self, a)
        if cfg.shocks:
            shocks.resolve(self, a)
        a.satiety -= 15
        recovery = cfg.elder_energy_recovery if a.retired else 30
        a.energy += recovery * (0.5 if a.sick_days > 0 else 1.0)   # sleeping
        a.mood += (50 - a.mood) * 0.02   # drift back toward neutral
        # health dynamics
        if a.satiety < 10:
            a.health -= 3
        if a.energy < 10:
            a.health -= 1
        if a.cash < 0:
            a.health -= cfg.debt_stress if cfg.economy else 1.0   # money stress
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
            if self.env.u(a.id, R.MORTALITY) < 1 - math.exp(-annual / 365):
                self._die(a, "age")

    def settle_deaths(self) -> None:
        """Estates and widowing for today's deaths, in id order. Runs after every agent has
        lived the day, so the result does not depend on the order of `self.agents`."""
        for a in sorted(self._dying, key=lambda x: x.id):
            family.on_death(self, a)
        self._dying.clear()

    def log_event(self, a: Agent, kind: str, **detail) -> None:
        self.events.append({"day": self.day, "id": a.id, "event": kind, **detail})

    def _die(self, a: Agent, cause: str) -> None:
        a.alive = False
        a.end_price_idx = self.prices.price_idx
        self.deaths.append({"day": self.day, "id": a.id, "sex": a.sex,
                            "age": a.age, "cause": cause, "cash": a.cash,
                            "education": a.education, "career": a.career})
        if self.cfg.reproduction:
            self._dying.append(a)       # settled after the whole day, see settle_deaths

    def _snapshot(self) -> None:
        if self.cfg.reproduction:
            self.population_log.append({"day": self.day, **self.population()})
        for a in self.agents:
            if a.alive:
                self.snapshots.append({
                    "day": self.day, "id": a.id, "sex": a.sex, "age": a.age,
                    "health": a.health, "energy": a.energy,
                    "satiety": a.satiety, "mood": a.mood, "cash": a.cash,
                    "education": a.education, "career": a.career})

    def outcomes(self) -> list[dict]:
        """One row per agent: genes, how long they lived, and what they ended with.
        Family columns appear only when `reproduction` is on, so the schema is unchanged otherwise."""
        rows = []
        for a in self.agents:
            years = (a.age_days - a.start_age_days) / 365
            row = {"id": a.id, "sex": a.sex, "alive": a.alive, **a.genes,
                   "years_lived": years, "age": a.age, "cash": a.cash,
                   "education": a.education, "career": a.career,
                   "health": a.health,
                   "job_level": a.job_level, "taxes_paid": a.taxes_paid,
                   "n_illness": a.n_illness, "n_job_losses": a.n_job_losses,
                   "chronic": a.chronic,
                   "wealth_per_year": (a.cash - a.start_cash) / max(years, 1e-9)}
            if self.cfg.reproduction:
                row.update(generation=a.generation, n_children=len(a.children),
                           mother_id=a.parent_ids[0] if a.parent_ids else None,
                           father_id=a.parent_ids[1] if a.parent_ids else None,
                           inherited=a.inherited)
            if self.cfg.uses_real_money:
                # Real (day-0) money, so figures from different decades are comparable. This follows the
                # money convention, not reproduction: a real-money run without births reports the same way.
                p_end = a.end_price_idx if a.end_price_idx is not None else self.prices.price_idx
                cash_real = a.cash / p_end
                # Accumulation excluding transfers; undefined (NaN) until there is enough adult exposure
                # for an annual rate to mean anything. Inheritance stays visible in `inherited_real`.
                row["wealth_per_year"] = ((cash_real - a.start_cash - a.inherited_real) / years
                                          if years >= self.cfg.min_rate_years else float("nan"))
                row.update(inherited_real=a.inherited_real, cash_real=cash_real, price_idx_end=p_end)
            elif self.cfg.reproduction:
                # Nominal convention with births: same exposure guard, transfers excluded, in nominal money.
                row["wealth_per_year"] = ((a.cash - a.start_cash - a.inherited) / years
                                          if years >= self.cfg.min_rate_years else float("nan"))
            rows.append(row)
        return rows

    def lineage(self) -> list[dict]:
        """Everyone born into this world: adults (alive or dead) and children still waiting to come of
        age (`status` = "minor"). Leaving the minors out would drop every child born in the last
        `maturity_age` years while they still occupy population slots and are listed by their parents."""
        rows = [{"id": a.id, "status": "adult", "generation": a.generation,
                 "mother_id": a.parent_ids[0] if a.parent_ids else None,
                 "father_id": a.parent_ids[1] if a.parent_ids else None,
                 "birth_day": a.birth_day, "entered_day": a.entered_day, "sex": a.sex,
                 "alive": a.alive, "n_children": len(a.children),
                 "inherited": a.inherited, "inherited_real": a.inherited_real, **a.genes}
                for a in self.agents]
        rows += [{"id": c.id, "status": "minor", "generation": c.generation,
                  "mother_id": c.mother_id, "father_id": c.father_id,
                  "birth_day": c.birth_day, "entered_day": None, "sex": c.sex,
                  "alive": True, "n_children": 0,
                  "inherited": c.trust, "inherited_real": c.trust / self.prices.price_idx, **c.genes}
                 for c in self.pending]
        return rows

    def population(self) -> dict:
        """Adults and queued children counted separately; the cap uses their sum."""
        adults = sum(1 for a in self.agents if a.alive)
        return {"alive_adults": adults, "pending_children": len(self.pending),
                "population_total": adults + len(self.pending)}

    def run(self) -> None:
        for _ in range(self.cfg.days):
            self.tick()
            if not any(a.alive for a in self.agents) and not self.pending:
                break
