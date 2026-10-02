"""Pension in real-money mode: based on real career earnings, indexed to prices through retirement."""
import math

from lifesim.actions import Action, apply
from lifesim.agent import Agent
from lifesim.config import Config
from lifesim.economy import Prices
from lifesim.world import World


class _Rest:
    def act(self, obs):
        return Action.REST


def _retiree_world(real_money, earnings_nominal=730_000.0, earnings_real=730_000.0):
    """One agent a day short of retirement, with a 40 year career on record."""
    w = World(Config(seed=1, n_agents=1, years=1, shocks=False, real_money=real_money),
              lambda r, g: _Rest())
    a = w.agents[0]
    a.start_age_days = int(25 * 365)
    a.age_days = int(65 * 365) - 1
    a.lifetime_earnings, a.lifetime_earnings_real = earnings_nominal, earnings_real
    a.health = 100.0
    return w, a


def _retire_at(w, a, price_idx):
    w.prices.price_idx = price_idx
    w._end_of_day(a)
    assert a.retired
    return a.pension_daily


def test_earnings_are_recorded_both_nominally_and_deflated():
    cfg = Config()
    a = Agent(id=0, sex="M", age_days=30 * 365, job_level=2)
    apply(Action.WORK, a, cfg, Prices(price_idx=2.0, wage_idx=2.0))
    assert a.lifetime_earnings > 0
    assert math.isclose(a.lifetime_earnings_real, a.lifetime_earnings / 2.0)
    b = Agent(id=1, sex="M", age_days=30 * 365, job_level=2)
    apply(Action.WORK, b, cfg)                                     # neutral prices: the two coincide
    assert b.lifetime_earnings_real == b.lifetime_earnings


def test_real_money_pension_has_the_same_purchasing_power_whenever_you_retire():
    w1, a1 = _retiree_world(real_money=True)
    w2, a2 = _retiree_world(real_money=True)
    early, late = _retire_at(w1, a1, 1.0), _retire_at(w2, a2, 4.0)
    assert math.isclose(late / early, 4.0)                         # same real career, 4x the prices, 4x the payment
    active_days = a1.age_days - a1.start_age_days
    assert math.isclose(early, w1.cfg.pension_frac * 730_000.0 / active_days)      # exact, at price index 1


def test_real_money_pension_keeps_up_with_prices_during_retirement():
    w, a = _retiree_world(real_money=True)
    p1 = _retire_at(w, a, 1.0)
    w.prices.price_idx = 3.0
    w._end_of_day(a)
    assert math.isclose(a.pension_daily, 3.0 * p1)
    w.prices.price_idx = 3.0 * 1.02 ** 10
    w._end_of_day(a)
    assert math.isclose(a.pension_daily, 3.0 * 1.02 ** 10 * p1)


def test_nominal_mode_pension_is_the_old_flat_amount():
    w, a = _retiree_world(real_money=False)
    p1 = _retire_at(w, a, 1.0)
    active_days = a.age_days - a.start_age_days
    assert math.isclose(p1, w.cfg.pension_frac * 730_000.0 / active_days)       # the original formula
    w.prices.price_idx = 3.0
    w._end_of_day(a)
    assert a.pension_daily == p1                                   # never indexed
    # and it ignores the real record entirely
    w2, a2 = _retiree_world(real_money=False, earnings_real=1.0)
    assert math.isclose(_retire_at(w2, a2, 1.0), p1, rel_tol=1e-9)


def test_default_config_without_reproduction_keeps_the_nominal_pension():
    cfg = Config()
    assert cfg.uses_real_money is False
    w = World(Config(seed=1, n_agents=1, years=1, shocks=False), lambda r, g: _Rest())
    a = w.agents[0]
    a.start_age_days, a.age_days = int(25 * 365), int(65 * 365) - 1
    a.lifetime_earnings, a.lifetime_earnings_real = 500_000.0, 1.0
    a.health = 100.0
    p = _retire_at(w, a, 2.0)
    assert math.isclose(p, w.cfg.pension_frac * 500_000.0 / (a.age_days - a.start_age_days))


def test_reproduction_turns_real_money_pension_on_automatically():
    w = World(Config(seed=1, n_agents=1, years=1, shocks=False, reproduction=True), lambda r, g: _Rest())
    a = w.agents[0]
    a.start_age_days, a.age_days = int(25 * 365), int(65 * 365) - 1
    a.lifetime_earnings, a.lifetime_earnings_real = 1.0, 730_000.0     # only the real record matters now
    a.health = 100.0
    p = _retire_at(w, a, 5.0)
    assert math.isclose(p, 5.0 * w.cfg.pension_frac * 730_000.0 / (a.age_days - a.start_age_days))
