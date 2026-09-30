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
