"""Couples, births and inheritance (step A of generations). Off unless cfg.reproduction.

Design: children are not simulated as agents. A birth enters a queue, the parents
pay for the child every day, and at `maturity_age` the child appears as an adult
carrying mixed genes, schooling bought by the parents' spending, and any inheritance
held in trust. Every random choice is keyed (see rng.py), so runs stay paired.
"""
import math
from dataclasses import dataclass, field

from . import economy
from . import rng as R
from .agent import Agent

# gene name -> Config attribute holding its allowed range
GENE_RANGES = {"study_frac": "study_frac_range", "cash_buffer": "cash_buffer_range",
               "patience": "patience_range", "wealth_weight": "wealth_weight_range"}


@dataclass
class Child:
    id: int
    sex: str
    birth_day: int
    mother_id: int
    father_id: int
    generation: int
    genes: dict
    spent_real: float = 0.0     # parents' spending so far, in day-0 money
    trust: float = 0.0          # inheritance held until adulthood, nominal money
    parents: tuple = field(init=False)

    def __post_init__(self):
        self.parents = (self.mother_id, self.father_id)


def related(a: Agent, b: Agent) -> bool:
    """Siblings, half-siblings, parent and child do not pair."""
    return bool(set(a.parent_ids) & set(b.parent_ids)) or a.id in b.parent_ids or b.id in a.parent_ids


def _pairable(a: Agent, cfg) -> bool:
    lo, hi = cfg.fertile_age
    return a.alive and a.partner_id is None and lo <= a.age <= hi


# -- matching -------------------------------------------------------------------

def match(world) -> None:
    cfg, env = world.cfg, world.env
    women = [a for a in world.agents if a.sex == "F" and _pairable(a, cfg)]
    men = [a for a in world.agents if a.sex == "M" and _pairable(a, cfg)]
    free = {m.id for m in men}
    for w in women:
        if env.u(w.id, R.MATCH) >= cfg.meet_rate:
            continue
        cands = [m for m in men if m.id in free and abs(m.age - w.age) <= cfg.max_age_gap
                 and not related(w, m)]
        if not cands:
            continue
        weights = [math.exp(-cfg.assortative_mating * abs(m.education - w.education) / 20.0)
                   for m in cands]
        target = env.u(w.id, R.MATCH_PICK) * sum(weights)
        acc, chosen = 0.0, cands[-1]
        for m, wt in zip(cands, weights):
            acc += wt
            if target < acc:
                chosen = m
                break
        free.discard(chosen.id)
        w.partner_id, chosen.partner_id = chosen.id, w.id
        world.log_event(w, "pair", partner=chosen.id)


# -- births -------------------------------------------------------------------------

def _crowding(world) -> float:
    n = sum(1 for a in world.agents if a.alive) + len(world.pending)
    return max(0.0, 1.0 - n / world.cfg.max_population)


def births(world) -> None:
    cfg, env = world.cfg, world.env
    crowd = _crowding(world)
    if crowd <= 0:
        return
    p_day = 1 - math.exp(-cfg.birth_rate * crowd / 365)
    reserve = cfg.child_reserve_days * cfg.child_cost * world.prices.price_idx
    lo, hi = cfg.fertile_age
    for mom in world.agents:
        if not mom.alive or mom.sex != "F" or mom.partner_id is None:
            continue
        dad = world.by_id[mom.partner_id]
        if not dad.alive or not (lo <= mom.age <= hi):
            continue
        if min(mom.health, dad.health) <= cfg.min_parent_health:
            continue
        if world.day - mom.last_birth_day < cfg.birth_spacing_years * 365:
            continue
        if cfg.fertility_needs_wealth and mom.cash + dad.cash < reserve:
            continue
        if env.u(mom.id, R.BIRTH) < p_day:
            _birth(world, mom, dad)


def _birth(world, mom: Agent, dad: Agent) -> None:
    cfg, env = world.cfg, world.env
    genes = {}
    for i, key in enumerate(sorted(mom.genes)):
        parent = mom if env.u(mom.id, R.CROSSOVER, i) < 0.5 else dad
        glo, ghi = getattr(cfg, GENE_RANGES[key])
        v = parent.genes[key] + cfg.mutation_sigma * (ghi - glo) * env.normal(mom.id, R.MUTATION, i)
        genes[key] = min(ghi, max(glo, v))
    child = Child(id=world.next_id, sex="F" if env.u(mom.id, R.CHILD_SEX) < 0.5 else "M",
                  birth_day=world.day, mother_id=mom.id, father_id=dad.id,
                  generation=max(mom.generation, dad.generation) + 1, genes=genes)
    world.next_id += 1
    world.pending.append(child)
    mom.last_birth_day = dad.last_birth_day = world.day
    mom.children.append(child.id)
    dad.children.append(child.id)
    world.log_event(mom, "birth", child=child.id, father=dad.id, generation=child.generation)


# -- raising children ---------------------------------------------------------------

def pay_child_costs(world) -> None:
    cfg, p = world.cfg, world.prices
    for ch in world.pending:
        parents = [a for a in (world.by_id[ch.mother_id], world.by_id[ch.father_id]) if a.alive]
        if not parents:
            continue                                  # orphan: nobody left to pay
        wealth_real = max(0.0, sum(a.cash for a in parents) / len(parents)) / p.price_idx
        invest = min(cfg.child_invest_cap, cfg.child_invest_frac * wealth_real / 365)
        spend_real = cfg.child_cost + invest
        for a in parents:
            a.cash -= spend_real * p.price_idx / len(parents)
        ch.spent_real += spend_real


def mature(world) -> None:
    cfg = world.cfg
    age_days = int(cfg.maturity_age * 365)
    for ch in [c for c in world.pending if world.day - c.birth_day >= age_days]:
        world.pending.remove(ch)
        lo, hi = cfg.start_education
        edu = lo + (hi - lo) * (1 - math.exp(-ch.spent_real / cfg.edu_spend_scale))
        a = Agent(id=ch.id, sex=ch.sex, age_days=age_days, start_age_days=age_days,
                  cash=cfg.start_cash + ch.trust, start_cash=cfg.start_cash + ch.trust,
                  education=edu, genes=dict(ch.genes), parent_ids=ch.parents,
                  generation=ch.generation, birth_day=ch.birth_day, entered_day=world.day)
        if cfg.economy:
            a.job_level = economy.job_level_for(a, cfg)
        world.register(a)
        world.log_event(a, "adult", generation=a.generation, education=edu, trust=ch.trust)


# -- death ------------------------------------------------------------------------------

def on_death(world, a: Agent) -> None:
    """Widow the partner and settle the estate. `a.cash` is left as the value at death."""
    cfg = world.cfg
    spouse = world.by_id.get(a.partner_id) if a.partner_id is not None else None
    if spouse is not None:
        spouse.partner_id = None
        if not spouse.alive:
            spouse = None
    a.partner_id = None

    estate = max(a.cash, 0.0)                         # debt is forgiven
    adults = [world.by_id[c] for c in a.children if c in world.by_id and world.by_id[c].alive]
    minors = [c for c in world.pending if c.id in a.children]
    n_heirs = len(adults) + len(minors)
    if estate <= 0:
        return
    to_children = 0.0
    if n_heirs and spouse is not None:
        to_children = cfg.estate_to_children_with_spouse * estate
    elif n_heirs:
        to_children = estate
    to_spouse = (estate - to_children) if spouse is not None else 0.0
    if n_heirs:
        share = to_children / n_heirs
        for h in adults:
            h.cash += share
        for c in minors:
            c.trust += share
    if spouse is not None:
        spouse.cash += to_spouse
    world.log_event(a, "estate", estate=estate, heirs=n_heirs, to_children=to_children,
                    to_spouse=to_spouse, lost=estate - to_children - to_spouse)


def step(world) -> None:
    """Once per day, after every agent has lived its day."""
    pay_child_costs(world)
    if world.day % world.cfg.meet_interval == 0:
        match(world)
    births(world)
    mature(world)
