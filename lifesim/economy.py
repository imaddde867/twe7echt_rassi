"""Layer 3: jobs, prices, tax, lifestyle spending, savings and debt."""
from dataclasses import dataclass

from .agent import Agent
from .config import Config


@dataclass
class Prices:
    price_idx: float = 1.0   # what things cost, relative to day 0
    wage_idx: float = 1.0    # what pay is, relative to day 0


NEUTRAL = Prices()


def update_prices(prices: Prices, cfg: Config, day: int) -> None:
    years = day / 365
    prices.price_idx = (1 + cfg.inflation) ** years
    prices.wage_idx = (1 + cfg.wage_growth) ** years


def job_level_for(a: Agent, cfg: Config) -> int:
    """Highest job tier whose education and career requirements the agent meets."""
    level = 0
    for i, (min_edu, min_career, _mult) in enumerate(cfg.job_tiers):
        if a.education >= min_edu and a.career >= min_career:
            level = i
    return level


def wage(a: Agent, cfg: Config, prices: Prices = NEUTRAL) -> float:
    if not cfg.economy:
        # v1 rule: education and experience raise pay continuously
        return cfg.base_wage * (1 + a.education / 50) * (1 + a.career / 100)
    mult = cfg.job_tiers[max(a.job_level, 0)][2]
    return cfg.base_wage * mult * (1 + a.career / 100) * prices.wage_idx


def tax(gross_real: float, cfg: Config) -> float:
    """Progressive tax on one day's earnings (in day-0 money)."""
    owed, lower = 0.0, 0.0
    for upper, rate in cfg.tax_brackets:
        if gross_real > lower:
            owed += (min(gross_real, upper) - lower) * rate
        lower = upper
    return owed


def credit_wall(cfg: Config, prices: Prices = NEUTRAL) -> float:
    """Borrowing limit in current money. Inflation-adjusted, like every other price here;
    the environment and the utility planner both use this one definition."""
    return cfg.credit_limit * prices.price_idx


def end_of_day(world, a: Agent) -> None:
    """Tax, living costs (with lifestyle creep), interest. Replaces the flat rent."""
    cfg, p = world.cfg, world.prices
    gross = a.earned_today
    owed = tax(gross / p.price_idx, cfg) * p.price_idx   # brackets are in day-0 money
    a.cash -= owed
    a.taxes_paid += owed
    a.earned_today = 0.0

    net_income = gross - owed + a.pension_daily
    a.income_ema = 0.99 * a.income_ema + 0.01 * net_income

    base = cfg.daily_living_cost
    creep = cfg.lifestyle_frac * max(0.0, a.income_ema / p.price_idx - base)
    a.cash -= (base + creep) * p.price_idx

    if a.cash < -credit_wall(cfg, p):   # safety net: keeps people alive, not comfortable
        a.cash += cfg.welfare_daily * p.price_idx

    rate = cfg.savings_rate if a.cash > 0 else cfg.debt_rate
    a.cash += a.cash * rate / 365
