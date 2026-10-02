"""Step B: marker gene, generation-aware measures, drift-versus-selection summary, runner."""
import math
import sys

import numpy as np
import pandas as pd
import pytest

from lifesim import generations as gen
from lifesim.actions import Action
from lifesim.config import Config
from lifesim.family import GENE_INDEX, GENE_RANGES
from lifesim.policy import genes_read
from lifesim.world import World

# ---------------------------------------------------------------------------------- the measures

def test_gini_known_values():
    assert gen.gini([1, 1, 1, 1]) == 0.0
    assert math.isclose(gen.gini([0, 0, 0, 1]), 0.75)            # one person has everything: (n-1)/n
    assert math.isclose(gen.gini([1, 2, 3, 4]), 0.25)
    assert gen.gini([0, 0, 0]) == 0.0
    assert math.isnan(gen.gini([]))
    assert math.isclose(gen.gini([5, 5, 100]), gen.gini([50, 50, 1000]))   # scale free


def test_spearman_uses_ranks_ties_and_degrades_to_nan():
    assert math.isclose(gen.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(gen.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
    assert math.isclose(gen.spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0)          # monotone, not linear
    ranks = np.corrcoef([1, 2.5, 2.5, 4], [1, 2, 3, 4])[0, 1]
    assert math.isclose(gen.spearman([1, 2, 2, 3], [1, 2, 3, 4]), ranks)         # average ranks for ties
    assert math.isnan(gen.spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    assert math.isnan(gen.spearman([1, 2], [1, 2]))


def test_price_index_matches_the_day_just_lived_and_is_neutral_without_an_economy():
    assert gen.price_idx_on(Config(), 1) == 1.0
    assert math.isclose(gen.price_idx_on(Config(), 366), 1.02)
    assert gen.price_idx_on(Config(economy=False), 5000) == 1.0


def test_wealth_at_age_picks_the_nearest_snapshot_converts_to_real_and_skips_people_who_never_reached_it():
    cfg = Config()
    snaps = pd.DataFrame([
        {"day": 366, "id": 1, "age": 39.8, "cash": 1020.0},
        {"day": 731, "id": 1, "age": 40.2, "cash": 2000.0},      # 39.8 is 0.2 away, 40.2 is 0.2: take the first (stable)
        {"day": 366, "id": 2, "age": 35.0, "cash": 500.0},       # never within tolerance
        {"day": 731, "id": 3, "age": 40.5, "cash": 1.0404 * 700.0},
    ])
    w = gen.wealth_at_age(snaps, cfg)
    assert set(w.index) == {1, 3}
    assert math.isclose(w[1], 1000.0)                            # 1020 / 1.02
    assert math.isclose(w[3], 700.0, rel_tol=1e-9)               # 1.0404 = 1.02^2
    assert gen.wealth_at_age(pd.DataFrame(columns=["day", "id", "age", "cash"]), cfg).empty


def _lineage():
    rows = [
        {"id": 0, "status": "adult", "generation": 0, "mother_id": None, "father_id": None, "g": 0.2},
        {"id": 1, "status": "adult", "generation": 0, "mother_id": None, "father_id": None, "g": 0.4},
        {"id": 2, "status": "adult", "generation": 1, "mother_id": 0, "father_id": 1, "g": 0.9},
        {"id": 3, "status": "adult", "generation": 1, "mother_id": 0, "father_id": 9, "g": 0.7},
        {"id": 4, "status": "minor", "generation": 2, "mother_id": 2, "father_id": 3, "g": 0.0},
    ]
    return pd.DataFrame(rows)


def test_genes_by_generation_counts_adults_only():
    t = gen.genes_by_generation(_lineage(), ["g"])
    assert set(t["generation"]) == {0, 1}                         # the minor's generation does not appear
    g0 = t[t["generation"] == 0].iloc[0]
    assert g0["n"] == 2 and math.isclose(g0["mean"], 0.3) and math.isclose(g0["sd"], 0.1)
    assert math.isclose(t[t["generation"] == 1].iloc[0]["mean"], 0.8)


def test_parent_child_pairs_average_known_parents_and_drop_people_without_data():
    w = pd.Series({0: 100.0, 1: 300.0, 2: 250.0})                 # id 3 has no wealth at 40; id 9 unknown
    p = gen.parent_child_pairs(_lineage(), w)
    assert list(p["child_id"]) == [2]                             # 3 has no own wealth; the minor is not an adult
    assert p.iloc[0]["parent_wealth"] == 200.0 and p.iloc[0]["n_parents_known"] == 2
    w2 = pd.Series({0: 100.0, 3: 50.0})                           # child 3: mother known, father 9 unknown
    p2 = gen.parent_child_pairs(_lineage(), w2)
    assert p2.iloc[0]["parent_wealth"] == 100.0 and p2.iloc[0]["n_parents_known"] == 1


def test_inequality_floors_debt_at_zero_and_reports_the_share_in_debt():
    snaps = pd.DataFrame({"day": [366] * 4, "id": range(4), "cash": [-5.0 * 1.0, 0.0, 10.0, 10.0]})
    s = gen.inequality_series(snaps, Config())
    assert len(s) == 1 and math.isclose(s.iloc[0]["gini"], 0.5) and s.iloc[0]["share_in_debt"] == 0.25


# -------------------------------------------------------------- drift versus selection arithmetic

def _fake_run(shifts, n=20, final_gen=2, ranges=None, read=("study_frac",)):
    ranges = ranges or {g: (0.0, 1.0) for g in shifts}
    rows = []
    for g, sh in shifts.items():
        for generation, mean in ((0, 0.5), (1, 0.5 + sh / 2), (final_gen, 0.5 + sh)):
            rows.append({"generation": generation, "gene": g, "n": n, "mean": mean, "sd": 0.1})
    return {"genes": list(shifts), "read_genes": list(read), "ranges": ranges,
            "by_generation": pd.DataFrame(rows), "pairs": pd.DataFrame(columns=["parent_wealth", "child_wealth"]),
            "gini": pd.DataFrame({"day": [365 * y for y in range(1, 61)], "n_adults": 50, "gini": 0.4, "share_in_debt": 0.1}),
            "population": pd.DataFrame({"population_total": [10, 20, 30]}), "max_generation": final_gen}


def test_gene_shift_is_in_range_units_and_uses_the_last_big_enough_generation():
    run = _fake_run({"study_frac": 0.2}, ranges={"study_frac": (0.0, 2.0)})
    s = gen.gene_shifts(run, min_n=10)
    assert math.isclose(s.iloc[0]["shift"], 0.1) and s.iloc[0]["final_generation"] == 2       # 0.2 over a width of 2
    small = _fake_run({"study_frac": 0.2}, n=5)
    assert gen.gene_shifts(small, min_n=10).empty                                              # nothing is big enough
    no_founders = _fake_run({"study_frac": 0.2})
    no_founders["by_generation"] = no_founders["by_generation"].query("generation != 0")
    assert gen.gene_shifts(no_founders).empty


def _arm(read_shift, neutral_sd, n_runs=12, seed=0):
    rng = np.random.default_rng(seed)
    return [_fake_run({"study_frac": read_shift + rng.normal(0, 0.01), "marker": rng.normal(0, neutral_sd),
                       "patience": rng.normal(0, neutral_sd)}) for _ in range(n_runs)]


def _metric(table, prefix):
    return table[table["metric"].str.startswith(prefix)].iloc[0]


def test_a_gene_that_moves_beyond_the_neutral_genes_gets_a_large_z_and_a_drifting_one_does_not():
    selected = gen.summarize_arm(_arm(read_shift=0.15, neutral_sd=0.02), min_n=10)
    z_sel = _metric(selected, "z[study_frac]")["value"]
    assert abs(z_sel) > 2
    assert _metric(selected, "shift[study_frac] minus neutral")["value"] > 0.1
    drifting = gen.summarize_arm(_arm(read_shift=0.0, neutral_sd=0.02, seed=3), min_n=10)
    assert abs(_metric(drifting, "z[study_frac]")["value"]) < 2.5
    assert _metric(selected, "shift[marker]")["metric"].endswith("(neutral)")


def _common_mode_arm(common, n_runs=12, seed=7):
    """Every gene drifts by about `common` (e.g. clipping at the range edges or a shared bottleneck)."""
    rng = np.random.default_rng(seed)
    return [_fake_run({"study_frac": common + rng.normal(0, 0.01), "marker": common + rng.normal(0, 0.01),
                       "patience": common + rng.normal(0, 0.01)}) for _ in range(n_runs)]


def test_a_shift_shared_by_every_gene_is_not_mistaken_for_selection():
    t = gen.summarize_arm(_common_mode_arm(0.1), min_n=10)
    assert math.isclose(_metric(t, "shift[study_frac]")["value"], 0.1, abs_tol=0.02)         # the raw shift is large
    assert abs(_metric(t, "shift[study_frac] minus neutral")["value"]) < 0.02               # but it is the neutral level
    assert abs(_metric(t, "z[study_frac]")["value"]) < 2.5
    both = gen.summarize_arm(_arm(read_shift=0.15, neutral_sd=0.0), min_n=10)
    assert math.isclose(_metric(both, "shift[study_frac] minus neutral")["value"], 0.15, abs_tol=0.02)


def test_summary_has_the_demographic_and_inequality_rows():
    t = gen.summarize_arm(_arm(0.1, 0.02, n_runs=3), min_n=10)
    metrics = " | ".join(t["metric"])
    for needle in ("spearman(child, parent wealth at 40)", "gini of real net worth", "share of adults in debt",
                   "last generation reached", "final population"):
        assert needle in metrics
    assert math.isclose(_metric(t, "gini")["value"], 0.4) and _metric(t, "final population")["value"] == 30.0


# -------------------------------------------------------------------------------- the marker gene

class _Survivor:
    def act(self, obs):
        return Action.EAT if obs[2] < 50 else Action.REST


def _bred(**cfg):
    base = dict(seed=3, n_agents=25, years=40, reproduction=True, shocks=False, meet_rate=0.5,
                birth_rate=1.0, birth_reserve_gate=False, log_every=365)
    base.update(cfg)
    w = World(Config(**base), lambda rng, genes: _Survivor())
    w.run()
    return w


def test_marker_exists_only_with_reproduction_and_every_gene_has_a_fixed_draw_index():
    on = World(Config(seed=1, n_agents=3, years=1, reproduction=True), lambda r, g: _Survivor())
    off = World(Config(seed=1, n_agents=3, years=1), lambda r, g: _Survivor())
    assert "marker" in on.agents[0].genes and "marker" not in off.agents[0].genes
    lo, hi = Config().marker_range
    assert all(lo <= a.genes["marker"] <= hi for a in on.agents)
    assert set(GENE_INDEX) == set(GENE_RANGES) == set(on.agents[0].genes)
    # the original four keep the index they had as positions in the sorted gene list
    assert [GENE_INDEX[g] for g in ("cash_buffer", "patience", "study_frac", "wealth_weight")] == [0, 1, 2, 3]
    assert len(set(GENE_INDEX.values())) == len(GENE_INDEX)


def test_marker_does_not_disturb_the_other_founder_genes():
    on = World(Config(seed=5, n_agents=8, years=1, reproduction=True), lambda r, g: _Survivor())
    off = World(Config(seed=5, n_agents=8, years=1), lambda r, g: _Survivor())
    for a, b in zip(on.agents, off.agents):
        for g in ("study_frac", "cash_buffer", "patience", "wealth_weight"):
            assert a.genes[g] == b.genes[g]


def test_marker_is_inherited_mutated_clipped_and_read_by_no_policy():
    w = _bred(mutation_sigma=0.0)
    kids = [a for a in w.agents if a.parent_ids]
    assert kids
    for k in kids:
        mom, dad = (w.by_id[i] for i in k.parent_ids)
        assert k.genes["marker"] in (mom.genes["marker"], dad.genes["marker"])      # no noise: a parent's value
    noisy = _bred(mutation_sigma=3.0)
    lo, hi = Config().marker_range
    assert all(lo <= a.genes["marker"] <= hi for a in noisy.agents)
    assert any(a.genes["marker"] not in (w.by_id[a.parent_ids[0]].genes["marker"], w.by_id[a.parent_ids[1]].genes["marker"])
               for a in noisy.agents if a.parent_ids)                             # and with noise it does change
    for name in ("rule", "utility"):
        assert "marker" not in genes_read(name)


def test_changing_the_marker_range_cannot_change_anything_that_happens():
    a, b = _bred(), _bred(marker_range=(5.0, 9.0))
    assert a.events == b.events and a.deaths == b.deaths           # the marker is purely a label
    assert [x.cash for x in a.agents] == [x.cash for x in b.agents]


def test_widowhood_clears_the_partnership_day_on_both_sides():
    w = _bred(years=1, n_agents=2, meet_rate=1.0)
    f, m = w.agents
    f.sex, m.sex = "F", "M"
    f.partner_id, m.partner_id, f.pair_day, m.pair_day = m.id, f.id, 10, 10
    w._die(m, "age")
    w.settle_deaths()
    assert f.partner_id is None and f.pair_day is None and m.pair_day is None


# ------------------------------------------------------------------------------------------- runner

_TINY = {"meet_rate": 0.5, "birth_rate": 1.0, "birth_reserve_gate": False}


def test_analyze_world_needs_reproduction_and_reports_adult_generations():
    with pytest.raises(ValueError, match="reproduction"):
        gen.analyze_world(World(Config(seed=1, n_agents=3, years=1), lambda r, g: _Survivor()), "rule")
    run = gen.run_one("t", _TINY, 1, 14, 45, "rule")
    assert {"genes", "read_genes", "by_generation", "gini", "pairs", "population", "max_generation"} <= set(run)
    assert run["max_generation"] >= 1 and run["read_genes"] == ["cash_buffer", "study_frac"]
    assert "marker" in run["genes"] and run["arm"] == "t" and run["seed"] == 1


def test_runs_are_reproducible_and_the_parallel_path_matches_the_serial_one():
    arms = {"baseline": _TINY}
    serial = gen.run_arms(arms, [0, 1], 10, 24, "rule", workers=1)
    again = gen.run_arms(arms, [0, 1], 10, 24, "rule", workers=1)
    parallel = gen.run_arms(arms, [0, 1], 10, 24, "rule", workers=2)
    ref = gen.summarize_arm(serial["baseline"], min_n=2)
    pd.testing.assert_frame_equal(ref, gen.summarize_arm(again["baseline"], min_n=2))
    pd.testing.assert_frame_equal(ref, gen.summarize_arm(parallel["baseline"], min_n=2))


def test_cli_writes_the_tables_for_default_and_custom_arms(monkeypatch, tmp_path, capsys):
    base = ["generations", "--seeds", "2", "--agents", "10", "--years", "24", "--workers", "1",
            "--min-generation-size", "2", "--out"]
    monkeypatch.setattr(sys, "argv", [*base, str(tmp_path / "custom"), "--arm", "fast", "meet_rate=0.5",
                                      "birth_rate=1.0", "birth_reserve_gate=False", "--arm", "slow", "birth_rate=0.2"])
    gen.main()
    out = capsys.readouterr().out
    summary = pd.read_csv(tmp_path / "custom" / "summary.csv")
    assert set(summary["arm"]) == {"fast", "slow"} and {"metric", "value", "se", "n_runs"} <= set(summary.columns)
    for name in ("genes_by_generation", "gini", "parent_child", "population"):
        assert (tmp_path / "custom" / f"{name}.csv").exists()
    assert "z[study_frac]" in out or "shift[study_frac]" in out
    genes = pd.read_csv(tmp_path / "custom" / "genes_by_generation.csv")
    assert {"arm", "seed", "generation", "gene", "n", "mean", "sd"} <= set(genes.columns)


def test_default_arms_are_the_ones_the_review_asked_for():
    assert set(gen.DEFAULT_ARMS) == {"baseline", "no_reserve_gate", "assortative"}
    assert gen.DEFAULT_ARMS["no_reserve_gate"] == {"birth_reserve_gate": False}
    assert gen.DEFAULT_ARMS["assortative"]["assortative_mating"] > 0


def test_ablate_points_at_the_generation_tool():
    from lifesim import ablate
    with pytest.raises(ValueError, match="lifesim.generations"):
        ablate.summarize({"reproduction": True}, seeds=1, agents=2, years=1, policy="rule")
