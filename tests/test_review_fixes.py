"""Regression tests for the PR review: job tiers, ES checkpoints, paired randomness, reward."""
import numpy as np

from lifesim import rng as R
from lifesim.actions import Action, apply, wage
from lifesim.agent import Agent
from lifesim.config import Config
from lifesim.rl.es import evaluate, gen_seeds, train
from lifesim.rl.mlp import save
from lifesim.rl.reward import Reward
from lifesim.world import World

# ---- 1. job tier follows education and career without a layoff ----------------

def test_study_crosses_tier_threshold_and_pay_rises_immediately():
    cfg = Config()                                       # tier 2 needs education >= 35
    a = Agent(id=0, sex="M", age_days=30 * 365, education=34.99, job_level=1)
    assert a.job_level == 1
    apply(Action.STUDY, a, cfg)
    assert a.education >= 35 and a.job_level == 2
    assert wage(a, cfg) == cfg.base_wage * cfg.job_tiers[2][2] * (1 + a.career / 100)


def test_work_crosses_career_threshold_and_pay_rises_immediately():
    cfg = Config()                                       # tier 3 needs education 60 and career 10
    a = Agent(id=0, sex="M", age_days=30 * 365, education=70.0, career=9.999, job_level=2)
    apply(Action.WORK, a, cfg)
    assert a.career >= 10 and a.job_level == 3
    assert wage(a, cfg) == cfg.base_wage * cfg.job_tiers[3][2] * (1 + a.career / 100)


def test_laid_off_agent_keeps_no_tier_until_rehired():
    cfg = Config()
    a = Agent(id=0, sex="M", age_days=30 * 365, education=34.99, job_level=-1, unemployed_days=5)
    apply(Action.STUDY, a, cfg)
    assert a.job_level == -1                             # still laid off: no promotion out of thin air


def test_observation_exposes_the_current_tier():
    from lifesim.agent import OBS
    cfg = Config()
    a = Agent(id=0, sex="F", age_days=30 * 365, education=34.99, job_level=1)
    apply(Action.STUDY, a, cfg)
    assert a.observe()[OBS["job_level"]] == 2


# ---- 2. ES checkpoints --------------------------------------------------------

def _tiny(out, generations, resume=False, **kw):
    return train(out, {"n_agents": 4, "years": 1}, Reward(), hidden=(8,), generations=generations,
                 pairs=2, seeds_per_eval=2, workers=1, clone_worlds=0, seed=5, resume=resume,
                 say=lambda s: None, **kw)


def test_best_npz_theta_is_exactly_the_evaluated_theta_of_the_best_generation(tmp_path):
    seen = {}
    _tiny(tmp_path, 4, on_generation=lambda g, th, ref: seen.update({g: (th, ref["fitness"])}))
    b = np.load(tmp_path / "best.npz")
    gen, fit = int(b["gen"]), float(b["fitness"])
    assert fit == max(f for _, f in seen.values())
    assert np.array_equal(b["theta"], seen[gen][0]) and fit == seen[gen][1]


def test_best_npz_holds_the_weights_that_earned_the_recorded_fitness(tmp_path):
    res = _tiny(tmp_path, 3)
    b = np.load(tmp_path / "best.npz")
    rescored = evaluate(b["theta"], gen_seeds(int(b["gen"]), 2),
                        {"n_agents": 4, "years": 1}, Reward(), (8,))
    assert rescored["fitness"] == float(b["fitness"])
    assert res["best_fitness"] == float(b["fitness"])


def test_resume_cannot_overwrite_a_better_historical_best(tmp_path):
    _tiny(tmp_path, 2)
    sentinel = np.full(len(np.load(tmp_path / "best.npz")["theta"]), 0.123)
    save(tmp_path / "best.npz", sentinel, (8,), fitness=99.0, gen=1)
    _tiny(tmp_path, 4, resume=True)
    b = np.load(tmp_path / "best.npz")
    assert float(b["fitness"]) == 99.0 and np.array_equal(b["theta"], sentinel)


def test_resume_does_replace_a_worse_best(tmp_path):
    _tiny(tmp_path, 2)
    th = np.load(tmp_path / "best.npz")["theta"]
    save(tmp_path / "best.npz", th, (8,), fitness=-5.0, gen=0)
    _tiny(tmp_path, 4, resume=True)
    assert float(np.load(tmp_path / "best.npz")["fitness"]) > -5.0


def test_interrupted_and_resumed_training_matches_uninterrupted(tmp_path):
    _tiny(tmp_path / "full", 4)
    _tiny(tmp_path / "split", 2)
    _tiny(tmp_path / "split", 4, resume=True)
    a, b = np.load(tmp_path / "full" / "latest.npz"), np.load(tmp_path / "split" / "latest.npz")
    assert np.array_equal(a["theta"], b["theta"])
    assert str(a["rng_state"]) == str(b["rng_state"])
    assert float(np.load(tmp_path / "full" / "best.npz")["fitness"]) == \
        float(np.load(tmp_path / "split" / "best.npz")["fitness"])


# ---- 3. environmental randomness is paired ------------------------------------

class _Survivor:
    """Same actions whatever it draws; `burn` makes it consume policy-side randomness."""

    def __init__(self, rng, burn):
        self.rng, self.burn = rng, burn

    def act(self, obs):
        if self.burn:
            self.rng.random(1000)
        return Action.EAT if obs[2] < 50 else Action.REST      # index 2 = satiety


def _world(burn=False, **cfg):
    base = dict(seed=3, n_agents=6, years=2, start_age=(20, 30), illness_severity=(1, 2))
    base.update(cfg)
    return World(Config(**base), lambda rng, genes: _Survivor(rng, burn))


def test_policy_side_randomness_does_not_change_exogenous_events():
    a, b = _world(burn=False, illness_rate=3.0), _world(burn=True, illness_rate=3.0)
    a.run()
    b.run()
    assert len(a.events) > 0
    assert a.events == b.events and a.deaths == b.deaths


def test_one_shock_firing_does_not_shift_unrelated_draws():
    quiet, busy = _world(illness_rate=0.0), _world(illness_rate=3.0)
    quiet.run()
    busy.run()
    assert not any(e["event"] == "illness" for e in quiet.events)
    assert any(e["event"] == "illness" for e in busy.events)
    for kind in ("job_loss", "windfall"):
        q = [e for e in quiet.events if e["event"] == kind]
        w = [e for e in busy.events if e["event"] == kind]
        assert q == w and len(q) > 0, kind


def test_draws_are_pure_functions_of_seed_day_agent_kind():
    e1, e2 = R.EnvRng(9), R.EnvRng(9)
    e1.begin_day(40)
    e2.begin_day(40)
    first = e1.u(3, R.MORTALITY)
    for kind in (R.ILLNESS, R.SEVERITY, R.WINDFALL_AMOUNT):      # other draws in between, any order
        e2.u(3, kind)
        e2.u(4, kind)
    assert e2.u(3, R.MORTALITY) == first
    e2.begin_day(41)
    assert e2.u(3, R.MORTALITY) != first
    assert R.EnvRng(10).u(3, R.MORTALITY) in (R.EnvRng(10).u(3, R.MORTALITY),)   # seeded, repeatable
    xs = [e1.u(i, R.ILLNESS) for i in range(2000)]
    assert 0.45 < float(np.mean(xs)) < 0.55 and min(xs) >= 0 and max(xs) < 1


# ---- 4. the dead earn no health reward ----------------------------------------

def test_dead_agent_gets_zero_health_reward_even_with_high_terminal_health():
    only_health = Reward(w_survival=0.0, w_wealth=0.0, w_health=1.0)
    dead = {"years_lived": 10, "cash": 0.0, "health": 90.0, "alive": False}    # died of old age, healthy
    alive = {"years_lived": 10, "cash": 0.0, "health": 90.0, "alive": True}
    v = only_health.per_agent([dead, alive], years=20)
    assert v[0] == 0.0 and v[1] == 0.9


# ---- 4. second review: ties, resume manifest, tax/credit units, reserved overrides ----

def test_tied_fitness_gets_equal_centered_ranks():
    from lifesim.rl.es import centered_ranks
    r = centered_ranks(np.array([1.0, 5.0, 5.0, 3.0, 1.0]))
    assert r[0] == r[4] and r[1] == r[2] and r[0] < r[3] < r[1]
    assert r.min() == -0.5 + 0.5 / 4 and np.isclose(r.sum(), 0.0)    # tied groups share average ranks


def test_all_equal_fitness_gives_zero_ranks():
    from lifesim.rl.es import centered_ranks
    assert not centered_ranks(np.full(8, 2.5)).any()


def test_all_equal_fitness_gives_zero_es_step(tmp_path, monkeypatch):
    from lifesim.rl import es
    monkeypatch.setattr(es, "_task", lambda args: 1.0)               # every candidate ties
    th0 = es.train(tmp_path / "a", {"n_agents": 4, "years": 1}, Reward(), hidden=(8,), generations=1,
                   pairs=2, seeds_per_eval=1, workers=1, clone_worlds=0, seed=5, weight_decay=0.0,
                   say=lambda s: None)["theta"]
    start = np.load(tmp_path / "a" / "best.npz")["theta"]            # gen 0 weights, before the step
    assert np.array_equal(th0, start)


def _resume_with(tmp_path, **changed):
    kw = dict(cfg={"n_agents": 4, "years": 1}, reward=Reward(), hidden=(8,), generations=3, workers=1)
    kw.update(changed)
    return train(tmp_path, kw["cfg"], kw["reward"], hidden=kw["hidden"], generations=kw["generations"],
                 pairs=2, seeds_per_eval=2, workers=kw["workers"], clone_worlds=0, seed=5, resume=True,
                 say=lambda s: None)


def test_resume_rejects_changed_reward(tmp_path):
    import pytest
    _tiny(tmp_path, 2)
    with pytest.raises(SystemExit, match="reward"):
        _resume_with(tmp_path, reward=Reward(w_wealth=2.0))


def test_resume_rejects_changed_world_config(tmp_path):
    import pytest
    _tiny(tmp_path, 2)
    with pytest.raises(SystemExit, match="cfg_kwargs"):
        _resume_with(tmp_path, cfg={"n_agents": 4, "years": 1, "inflation": 0.1})


def test_resume_allows_more_generations_and_workers(tmp_path):
    _tiny(tmp_path, 2)
    assert _resume_with(tmp_path, generations=3, workers=1)["best_gen"] >= 0


def test_resume_without_manifest_fails_clearly(tmp_path):
    import pytest
    _tiny(tmp_path, 2)
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="manifest"):
        _resume_with(tmp_path)


def test_tax_brackets_are_day0_money_deflated_by_prices_not_wages():
    from lifesim import economy
    w = World(Config(n_agents=1, years=1, lifestyle_frac=0.0), lambda rng, genes: _Survivor(rng, False))
    a = w.agents[0]
    w.prices.price_idx, w.prices.wage_idx = 1.0, 2.0                 # wages doubled, prices did not
    a.earned_today, a.cash = 100.0, 0.0
    economy.end_of_day(w, a)
    assert a.taxes_paid == economy.tax(100.0, w.cfg)


def test_credit_wall_scales_with_prices_for_world_and_planner():
    from lifesim import economy
    from lifesim.economy import Prices
    cfg = Config()
    prices = Prices(price_idx=2.0)
    assert economy.credit_wall(cfg, prices) == 2 * cfg.credit_limit
    a = Agent(id=0, sex="M", age_days=1, cash=-1.5 * cfg.credit_limit, satiety=20.0)
    apply(Action.EAT, a, cfg, prices)                                # inside the inflated wall: allowed
    assert a.satiety > 20.0
    b = Agent(id=0, sex="M", age_days=1, cash=-2.0 * cfg.credit_limit, satiety=20.0)
    apply(Action.EAT, b, cfg, prices)                                # at the wall: blocked
    assert b.satiety == 20.0


def test_reserved_keys_are_rejected_in_overrides():
    import pytest

    from lifesim.run import parse_overrides
    for k in ("seed=4", "years=10", "n_agents=5"):
        with pytest.raises(SystemExit, match="own flag"):
            parse_overrides([k])
    assert parse_overrides(["retire_age=70"]) == {"retire_age": 70}
