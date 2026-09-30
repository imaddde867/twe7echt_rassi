from enum import IntEnum

from . import economy
from .agent import Agent
from .config import Config
from .economy import NEUTRAL, Prices


class Action(IntEnum):
    WORK = 0
    STUDY = 1
    EAT = 2
    REST = 3
    SOCIALIZE = 4


N_ACTIONS = len(Action)


def wage(agent: Agent, cfg: Config, prices: Prices = NEUTRAL) -> float:
    # These rules ARE the assumptions of the simulation (see economy.wage).
    return economy.wage(agent, cfg, prices)


def _refresh_job_level(agent: Agent, cfg: Config) -> None:
    """Employed agents move up a tier the moment they qualify (laid-off ones, at -1, wait for rehire)."""
    if cfg.economy and agent.job_level >= 0:
        agent.job_level = economy.job_level_for(agent, cfg)


def apply(action: Action, agent: Agent, cfg: Config, prices: Prices = NEUTRAL) -> None:
    if action == Action.WORK and not agent.can_work:
        return  # retired, sick or laid off: the slot is wasted
    if action == Action.WORK:
        earned = wage(agent, cfg, prices)
        agent.earned_today += earned
        agent.lifetime_earnings += earned
        agent.cash += earned
        agent.energy -= 25
        agent.satiety -= 10
        agent.mood -= 2
        agent.career += 0.004 * (1 - agent.career / 100)
        _refresh_job_level(agent, cfg)
    elif action == Action.STUDY:
        agent.education += 0.03 * (1 - agent.education / 100)
        agent.energy -= 20
        agent.mood -= 2
        _refresh_job_level(agent, cfg)
    elif action == Action.EAT:
        cost = cfg.eat_cost * prices.price_idx
        if cfg.economy and agent.cash - cost < -cfg.credit_limit:
            return  # cannot borrow any more: no food
        agent.cash -= cost
        agent.satiety += 50
        agent.energy += 5
        agent.health += 1
    elif action == Action.REST:
        agent.energy += 40
        agent.health += 0.5
        agent.mood += 1
    elif action == Action.SOCIALIZE:
        agent.cash -= cfg.social_cost * prices.price_idx
        agent.mood += 10
        agent.energy -= 10
    agent.clamp()
