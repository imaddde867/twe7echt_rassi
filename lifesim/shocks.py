"""Random life events (layer 2): illness, job loss, windfall."""
import math

from . import economy
from .agent import Agent
from .config import Config


def _p(annual_rate: float) -> float:
    """Annual rate -> probability that it happens on a given day."""
    return 1 - math.exp(-annual_rate / 365)


def resolve(world, a: Agent) -> None:
    """Advance ongoing conditions by a day, then roll for new events."""
    cfg: Config = world.cfg
    rng = world.rng

    # ongoing conditions
    if a.sick_days > 0:
        a.sick_days -= 1
    if a.unemployed_days > 0:
        a.unemployed_days -= 1
        a.cash += cfg.unemployment_benefit * world.prices.wage_idx
    if a.chronic:
        a.cash -= cfg.chronic_daily_cost * world.prices.price_idx
        a.health -= cfg.chronic_daily_health_drain

    # back at work: the job tier reflects current education and career
    if cfg.economy and a.job_level < 0 and a.unemployed_days == 0:
        a.job_level = economy.job_level_for(a, cfg)

    # new events: one draw each, always consumed so runs stay comparable
    r_ill, r_job, r_win = rng.random(3)

    # illness: likelier when old or already unhealthy
    rate = (cfg.illness_rate * (1 + max(0.0, a.age - 40) / 30)
            * (1 + (100 - a.health) / 50))
    if r_ill < _p(rate):
        lo, hi = cfg.illness_severity
        severity = float(rng.uniform(lo, hi))
        cost = severity * cfg.healthcare_cost_per_severity * world.prices.price_idx
        treated = a.cash >= cost
        if treated:
            a.cash -= cost           # money buys less damage
        a.health -= severity * (cfg.treatment_health_fraction if treated else 1.0)
        a.sick_days = max(a.sick_days, int(severity))
        a.n_illness += 1
        world.log_event(a, "illness", severity=severity, treated=treated)
        if severity >= cfg.chronic_severity and not a.chronic \
                and rng.random() < cfg.chronic_prob:
            a.chronic = True
            world.log_event(a, "chronic")

    # job loss
    if not a.retired and a.unemployed_days == 0 and r_job < _p(cfg.job_loss_rate):
        lo, hi = cfg.unemployed_days
        a.unemployed_days = int(rng.integers(lo, hi + 1))
        a.career *= cfg.career_loss_on_layoff
        a.job_level = -1
        a.n_job_losses += 1
        world.log_event(a, "job_loss", days=a.unemployed_days)

    # windfall
    if r_win < _p(cfg.windfall_rate):
        amount = float(rng.exponential(cfg.windfall_mean))
        a.cash += amount
        world.log_event(a, "windfall", amount=amount)
