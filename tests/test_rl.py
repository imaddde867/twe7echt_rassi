import numpy as np

from lifesim.actions import Action
from lifesim.agent import Agent
from lifesim.rl.es import centered_ranks, evaluate
from lifesim.rl.mlp import MLPPolicy, featurize, load, save
from lifesim.rl.reward import Reward


def test_flat_roundtrip_and_determinism():
    p = MLPPolicy(rng=np.random.default_rng(1))
    q = MLPPolicy.from_flat(p.get_flat())
    obs = Agent(id=0, sex="M", age_days=30 * 365).observe()
    assert p.n_params == len(p.get_flat())
    assert np.allclose(p.logits(featurize(obs)), q.logits(featurize(obs)))
    assert isinstance(p.act(obs), Action) and p.act(obs) == q.act(obs)


def test_save_load(tmp_path):
    p = MLPPolicy(hidden=(8,), rng=np.random.default_rng(2))
    save(tmp_path / "m.npz", p.get_flat(), (8,))
    q = load(tmp_path / "m.npz")
    x = featurize(Agent(id=0, sex="F", age_days=9000).observe())
    assert np.allclose(p.logits(x), q.logits(x))


def test_featurize_keeps_inputs_small_even_for_huge_cash():
    a = Agent(id=0, sex="M", age_days=30 * 365, cash=5e6)
    assert np.abs(featurize(a.observe())).max() <= 2.0


def test_supervised_fit_learns_a_simple_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 13))
    y = (X[:, 2] > 0).astype(int) + 2 * (X[:, 0] > 0.5)    # 4 classes from two features
    p = MLPPolicy(hidden=(16,), rng=rng)
    losses = p.fit(X, y, epochs=60, lr=1e-2, rng=rng)
    acc = (np.argmax(p.logits(X), 1) == y).mean()
    assert losses[-1] < losses[0] * 0.5 and acc > 0.85


def test_centered_ranks():
    r = centered_ranks(np.array([3.0, 1.0, 2.0]))
    assert r.min() == -0.5 and r.max() == 0.5 and r[1] < r[2] < r[0]


def test_evaluate_is_deterministic_for_same_seeds():
    th = MLPPolicy(rng=np.random.default_rng(3)).get_flat()
    kw = {"n_agents": 5, "years": 1}
    a = evaluate(th, [7], kw, Reward(), (32, 32))
    b = evaluate(th, [7], kw, Reward(), (32, 32))
    assert a == b


def test_reward_prefers_long_healthy_rich_lives():
    r = Reward()
    good = {"years_lived": 20, "cash": 50000, "health": 95, "alive": True}
    bad = {"years_lived": 3, "cash": -2000, "health": 0, "alive": False}
    v = r.per_agent([good, bad], years=20)
    assert v[0] > v[1]


def test_centered_ranks_ties_share_a_rank():
    r = centered_ranks(np.array([1.0, 2.0, 2.0, 3.0]))
    assert r[1] == r[2] and r[0] < r[1] < r[3]
    assert r.min() == -0.5 and r.max() == 0.5 and abs(r.sum()) < 1e-12


def test_all_equal_fitness_gives_zero_ranks_and_zero_gradient():
    r = centered_ranks(np.full(8, 0.37))
    assert np.all(r == 0.0)
    pairs, eps = 4, np.random.default_rng(0).normal(size=(4, 10))
    assert np.all((r[:pairs] - r[pairs:]) @ eps == 0.0)
