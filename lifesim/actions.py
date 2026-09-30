from enum import IntEnum

from .agent import Agent
from .config import Config


class Action(IntEnum):
    WORK = 0
    STUDY = 1
    EAT = 2
    REST = 3
    SOCIALIZE = 4


N_ACTIONS = len(Action)


def wage(agent: Agent, cfg: Config) -> float:
    # Education and experience both raise pay. These rules ARE the assumptions
    # of the simulation: change them and every downstream result changes.
    return cfg.base_wage * (1 + agent.education / 50) * (1 + agent.career / 100)


def apply(action: Action, agent: Agent, cfg: Config) -> None:
    if action == Action.WORK and agent.retired:
        return  # retired agents cannot work: the slot is wasted
    if action == Action.WORK:
        earned = wage(agent, cfg)
        agent.lifetime_earnings += earned
        agent.cash += earned
        agent.energy -= 25
        agent.satiety -= 10
        agent.mood -= 2
        agent.career += 0.004 * (1 - agent.career / 100)
    elif action == Action.STUDY:
        agent.education += 0.03 * (1 - agent.education / 100)
        agent.energy -= 20
        agent.mood -= 2
    elif action == Action.EAT:
        agent.cash -= cfg.eat_cost
        agent.satiety += 50
        agent.energy += 5
        agent.health += 1
    elif action == Action.REST:
        agent.energy += 40
        agent.health += 0.5
        agent.mood += 1
    elif action == Action.SOCIALIZE:
        agent.cash -= cfg.social_cost
        agent.mood += 10
        agent.energy -= 10
    agent.clamp()
