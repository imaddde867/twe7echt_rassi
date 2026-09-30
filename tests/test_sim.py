from lifesim.actions import Action, apply
from lifesim.agent import Agent, OBS_FIELDS
from lifesim.config import Config
from lifesim.policy import make_policy
from lifesim.world import World


def run(seed, policy="rule", agents=30, years=10):
    w = World(Config(seed=seed, n_agents=agents, years=years),
              lambda rng, genes: make_policy(policy, rng, genes))
    w.run()
    return w


def test_deterministic_given_seed():
    a, b = run(3), run(3)
    assert a.deaths == b.deaths and a.snapshots == b.snapshots


def test_stats_stay_in_bounds():
    w = run(1, "random")
    for s in w.snapshots:
        for k in ("health", "energy", "satiety", "mood", "education", "career"):
            assert 0 <= s[k] <= 100


def test_actions_change_expected_stats():
    cfg, a = Config(), Agent(id=0, sex="F", age_days=20 * 365)
    cash0 = a.cash
    apply(Action.WORK, a, cfg)
    assert a.cash > cash0
    e0 = a.education
    apply(Action.STUDY, a, cfg)
    assert a.education > e0


def test_observation_shape():
    assert Agent(id=0, sex="M", age_days=1).observe().shape == (len(OBS_FIELDS),)


def test_rule_policy_beats_random():
    rule = sum(len(run(s, "rule").deaths) for s in range(3))
    rnd = sum(len(run(s, "random").deaths) for s in range(3))
    assert rule < rnd


def test_agents_have_distinct_genes_and_outcomes_table():
    w = run(2, agents=20, years=2)
    genes = {round(a.genes["study_frac"], 6) for a in w.agents}
    assert len(genes) == 20
    rows = w.outcomes()
    assert len(rows) == 20 and "wealth_per_year" in rows[0]


def test_retired_agent_cannot_work_and_gets_pension():
    cfg = Config()
    a = Agent(id=0, sex="M", age_days=70 * 365, retired=True)
    cash0 = a.cash
    apply(Action.WORK, a, cfg)
    assert a.cash == cash0 and a.lifetime_earnings == 0

    w = World(Config(seed=0, n_agents=1, years=1, start_age=(64, 65)),
              lambda rng, genes: make_policy("rule", rng, genes))
    w.run()
    ag = w.agents[0]
    assert ag.retired and ag.pension_daily > 0


def test_life_stages_switch_restores_old_behaviour():
    w = World(Config(seed=0, n_agents=5, years=3, start_age=(64, 65), life_stages=False),
              lambda rng, genes: make_policy("rule", rng, genes))
    w.run()
    assert not any(a.retired for a in w.agents)


def _w(**kw):
    return World(Config(seed=1, n_agents=30, years=10, **kw),
                 lambda rng, genes: make_policy("rule", rng, genes))


def test_shocks_fire_and_are_logged():
    w = _w(illness_rate=5.0, job_loss_rate=2.0, windfall_rate=2.0)
    w.run()
    kinds = {e["event"] for e in w.events}
    assert {"illness", "job_loss", "windfall"} <= kinds


def test_shocks_off_means_no_events():
    w = _w(shocks=False)
    w.run()
    assert w.events == [] and all(a.n_illness == 0 for a in w.agents)


def test_laid_off_or_sick_agent_cannot_work():
    cfg = Config()
    for kw in ({"unemployed_days": 10}, {"sick_days": 10}):
        a = Agent(id=0, sex="F", age_days=30 * 365, **kw)
        cash0 = a.cash
        apply(Action.WORK, a, cfg)
        assert a.cash == cash0


def test_treatment_reduces_damage():
    # same illness, rich agent pays and loses less health than a broke one
    def hit(cash):
        w = _w(illness_rate=1e6, illness_severity=(20, 20), job_loss_rate=0.0,
               windfall_rate=0.0, chronic_prob=0.0)
        a = Agent(id=0, sex="M", age_days=30 * 365, cash=cash, health=100.0)
        from lifesim import shocks
        shocks.resolve(w, a)
        return 100.0 - a.health
    assert hit(1e6) < hit(0.0)
