"""Couples, births, inheritance (family.py). All behind cfg.reproduction."""
import math

import numpy as np

from lifesim import family
from lifesim import rng as R
from lifesim.actions import Action
from lifesim.agent import Agent
from lifesim.config import Config
from lifesim.policy import make_policy
from lifesim.rl.reward import Reward
from lifesim.world import World


class _Rest:
    def act(self, obs):
        return Action.EAT if obs[2] < 50 else Action.REST


def _world(n=2, **cfg):
    base = dict(seed=1, n_agents=n, years=1, reproduction=True, shocks=False)
    base.update(cfg)
    w = World(Config(**base), lambda rng, genes: _Rest())
    for a in w.agents:                      # tests set the stage by hand
        a.age_days = 30 * 365
        a.health = 90.0
        a.partner_id = None
        a.children = []
    return w


def _couple(w, cash=50_000.0):
    mom, dad = w.agents[0], w.agents[1]
    mom.sex, dad.sex = "F", "M"
    mom.partner_id, dad.partner_id = dad.id, mom.id
    mom.cash = dad.cash = cash
    return mom, dad


def _force_birth(w, mom, dad):
    family._birth(w, mom, dad)
    return w.pending[-1]


# ---- off by default ---------------------------------------------------------------

def test_reproduction_is_off_by_default_and_changes_nothing():
    assert Config().reproduction is False
    w = World(Config(seed=2, n_agents=20, years=3, shocks=False), lambda rng, g: _Rest())
    w.run()
    assert not w.pending and len(w.agents) == 20
    assert not any(e["event"] in ("pair", "birth", "adult", "estate") for e in w.events)


# ---- matching ---------------------------------------------------------------------------

def test_matching_pairs_opposite_sex_within_age_gap_and_never_relatives():
    w = _world(40, meet_rate=1.0)
    for i, a in enumerate(w.agents):
        a.sex = "F" if i % 2 else "M"
        a.age_days = int((22 + (i % 10)) * 365)
    sib_f, sib_m = w.agents[1], w.agents[0]
    sib_f.parent_ids = sib_m.parent_ids = (900, 901)         # siblings
    w.day = 0
    family.match(w)
    paired = [a for a in w.agents if a.partner_id is not None]
    assert len(paired) > 10
    for a in paired:
        b = w.by_id[a.partner_id]
        assert b.partner_id == a.id and a.sex != b.sex
        assert abs(a.age - b.age) <= w.cfg.max_age_gap
        assert not family.related(w, a, b)
    assert sib_f.partner_id != sib_m.id


def _lone_pair(parents_f, parents_m, age_gap_years=0.0):
    w = _world(2, meet_rate=1.0)
    f, m = w.agents
    f.sex, m.sex = "F", "M"
    f.age_days, m.age_days = 25 * 365, int((25 + age_gap_years) * 365)
    f.parent_ids, m.parent_ids = parents_f, parents_m
    family.match(w)
    return f.partner_id == m.id


def test_the_only_eligible_man_and_woman_pair_unless_they_are_related():
    assert _lone_pair((), ())                                  # unrelated founders pair
    assert _lone_pair((10, 11), (12, 13))                      # unrelated families pair
    assert not _lone_pair((10, 11), (10, 11))                  # full siblings
    assert not _lone_pair((10, 11), (10, 99))                  # half siblings
    w = _world(2, meet_rate=1.0)
    f, m = w.agents
    f.sex, m.sex = "F", "M"
    f.age_days = m.age_days = 25 * 365
    m.parent_ids = (f.id, 77)                                  # she is his mother
    family.match(w)
    assert f.partner_id is None


def test_assortative_mating_prefers_similar_education():
    def mean_gap(strength):
        w = _world(60, meet_rate=1.0, assortative_mating=strength)
        for i, a in enumerate(w.agents):
            a.sex = "F" if i % 2 else "M"
            a.education = float(i * 1.5 % 90)
            a.age_days = 30 * 365
        family.match(w)
        gaps = [abs(a.education - w.by_id[a.partner_id].education)
                for a in w.agents if a.sex == "F" and a.partner_id is not None]
        return sum(gaps) / len(gaps)
    assert mean_gap(8.0) < mean_gap(0.0)


def test_unpaired_outside_fertile_window_do_not_pair():
    w = _world(10, meet_rate=1.0)
    for i, a in enumerate(w.agents):
        a.sex = "F" if i % 2 else "M"
        a.age_days = 55 * 365
    family.match(w)
    assert all(a.partner_id is None for a in w.agents)


# ---- birth gating -------------------------------------------------------------------------

def _try_births(w):
    w.cfg.birth_rate = 1e9                    # certainty, so only the gates decide
    before = len(w.pending)
    family.births(w)
    return len(w.pending) - before


def test_poor_couple_has_no_child_unless_wealth_is_switched_off():
    w = _world()
    _couple(w, cash=10.0)
    assert _try_births(w) == 0
    w.cfg.birth_reserve_gate = False
    assert _try_births(w) == 1


def test_birth_gates_health_age_spacing_and_crowding():
    w = _world()
    mom, dad = _couple(w)
    mom.health = 40.0
    assert _try_births(w) == 0                                # unhealthy
    mom.health = 90.0
    mom.age_days = 45 * 365
    assert _try_births(w) == 0                                # outside the fertile window
    mom.age_days = 30 * 365
    mom.last_birth_day = w.day - 100
    assert _try_births(w) == 0                                # too soon after the last one
    mom.last_birth_day = -10**9
    w.cfg.max_population = 2
    assert _try_births(w) == 0                                # no room
    w.cfg.max_population = 300
    assert _try_births(w) == 1


def test_child_genes_come_from_the_parents_and_stay_in_range():
    w = _world(mutation_sigma=0.0)
    mom, dad = _couple(w)
    ch = _force_birth(w, mom, dad)
    for k, v in ch.genes.items():
        assert v in (mom.genes[k], dad.genes[k])
    assert ch.generation == 1 and ch.parents == (mom.id, dad.id)
    w2 = _world(mutation_sigma=5.0)                           # huge noise: must still be clipped
    m2, d2 = _couple(w2)
    ch2 = _force_birth(w2, m2, d2)
    for k, v in ch2.genes.items():
        lo, hi = getattr(w2.cfg, family.GENE_RANGES[k])
        assert lo <= v <= hi


def test_birth_is_deterministic_for_the_same_seed():
    def genes():
        w = _world(mutation_sigma=0.2)
        mom, dad = _couple(w)
        return _force_birth(w, mom, dad).genes
    assert genes() == genes()


# ---- raising a child --------------------------------------------------------------------------

def test_parents_split_the_cost_and_a_survivor_pays_all():
    w = _world(child_invest_frac=0.0)
    mom, dad = _couple(w, cash=1000.0)
    ch = _force_birth(w, mom, dad)
    family.pay_child_costs(w)
    assert math.isclose(mom.cash, 1000.0 - 5.0) and math.isclose(dad.cash, 1000.0 - 5.0)
    assert math.isclose(ch.spent_real, 10.0)
    dad.alive = False
    family.pay_child_costs(w)
    assert math.isclose(mom.cash, 1000.0 - 5.0 - 10.0) and math.isclose(dad.cash, 995.0)


def test_orphan_costs_nothing_and_richer_parents_invest_more():
    w = _world()
    mom, dad = _couple(w, cash=1000.0)
    ch = _force_birth(w, mom, dad)
    mom.alive = dad.alive = False
    family.pay_child_costs(w)
    assert ch.spent_real == 0.0

    def spend(cash):
        w2 = _world()
        m, d = _couple(w2, cash=cash)
        c = _force_birth(w2, m, d)
        family.pay_child_costs(w2)
        return c.spent_real
    assert spend(500_000.0) > spend(100.0)


def test_child_matures_at_18_with_genes_schooling_and_a_new_id():
    w = _world(mutation_sigma=0.0)
    mom, dad = _couple(w)
    ch = _force_birth(w, mom, dad)
    ch.spent_real = 60_000.0
    family.mature(w)
    assert ch in w.pending                                    # too young
    w.day = ch.birth_day + int(18 * 365)
    n_before = len(w.agents)
    family.mature(w)
    kid = w.by_id[ch.id]
    assert len(w.agents) == n_before + 1 and ch not in w.pending
    assert kid.id not in (mom.id, dad.id) and kid.age >= 18 and kid.generation == 1
    assert kid.genes == ch.genes and kid.parent_ids == (mom.id, dad.id)
    assert kid.education > w.cfg.start_education[0]
    assert ch.id in w.policies

    def edu(spent):
        w2 = _world()
        m, d = _couple(w2)
        c = _force_birth(w2, m, d)
        c.spent_real = spent
        w2.day = c.birth_day + int(18 * 365)
        family.mature(w2)
        return w2.by_id[c.id].education
    assert edu(0.0) < edu(20_000.0) < edu(200_000.0) <= w.cfg.start_education[1]


# ---- death and inheritance ---------------------------------------------------------------------

def _die(w, *agents):
    """Kill agents on the same day, then settle (as World.tick does after the agent loop)."""
    for a in agents:
        w._die(a, "age")
    w.settle_deaths()


def test_estate_half_to_children_half_to_spouse_and_trust_for_minors():
    w = _world(3)
    mom, dad = _couple(w, cash=0.0)
    adult = w.agents[2]
    adult.parent_ids = (mom.id, dad.id)
    mom.children, dad.children = [adult.id], [adult.id]
    adult.cash = 0.0
    minor = _force_birth(w, mom, dad)
    dad.cash = 1000.0
    _die(w, dad)
    # heirs: one adult + one minor; half of 1000 to children = 250 each; half to the widow
    assert math.isclose(adult.cash, 250.0) and math.isclose(minor.trust, 250.0)
    assert math.isclose(mom.cash, 500.0) and mom.partner_id is None
    w.day = minor.birth_day + int(18 * 365)
    family.mature(w)
    assert math.isclose(w.by_id[minor.id].cash, w.cfg.start_cash + 250.0)


def test_estate_without_spouse_goes_to_children_and_debt_is_forgiven():
    w = _world(2)
    mom, dad = _couple(w, cash=0.0)
    ch = _force_birth(w, mom, dad)
    dad.alive = False
    dad.partner_id = None
    mom.partner_id = None
    mom.cash = 800.0
    _die(w, mom)
    assert math.isclose(ch.trust, 800.0)
    ch.trust = 0.0
    w2 = _world(2)
    m2, d2 = _couple(w2, cash=0.0)
    m2.cash = -5000.0
    _die(w2, m2)
    assert d2.cash == 0.0                                      # no inheritance from a debt


def test_estate_with_no_heirs_goes_to_the_spouse_or_is_lost():
    w = _world(2)
    mom, dad = _couple(w, cash=0.0)
    dad.cash = 400.0
    _die(w, dad)
    assert math.isclose(mom.cash, 400.0)
    ev = [e for e in w.events if e["event"] == "estate"][-1]
    assert ev["lost"] == 0.0
    w2 = _world(1)
    solo = w2.agents[0]
    solo.cash = 300.0
    _die(w2, solo)
    ev2 = [e for e in w2.events if e["event"] == "estate"][-1]
    assert math.isclose(ev2["lost"], 300.0)


def test_widow_can_pair_again_and_household_discount_applies_to_couples():
    w = _world(4, meet_rate=1.0)
    mom, dad = _couple(w)
    _die(w, dad)
    assert mom.partner_id is None and family._pairable(mom, w.cfg)
    from lifesim import economy
    alone, paired = Agent(id=90, sex="M", age_days=9000), Agent(id=91, sex="F", age_days=9000, partner_id=90)
    assert economy.household_factor(alone, w.cfg) == 1.0
    assert economy.household_factor(paired, w.cfg) == w.cfg.household_cost_factor
    assert economy.household_factor(paired, Config(reproduction=False)) == 1.0


# ---- whole-world behaviour -----------------------------------------------------------------------

def _run_world(burn=False, seed=3):
    class P:
        def __init__(self, rng):
            self.rng = rng

        def act(self, obs):
            if burn:
                self.rng.random(500)
            return Action.EAT if obs[2] < 50 else Action.REST
    w = World(Config(seed=seed, n_agents=40, years=45, reproduction=True, shocks=False,
                     birth_rate=1.0, meet_rate=0.3, log_every=365 * 50,
                     birth_reserve_gate=False),          # this policy never earns, so skip the wealth gate
              lambda rng, genes: P(rng))
    w.run()
    return w


def test_world_grows_generations_with_unique_ids_and_consistent_links():
    w = _run_world()
    ids = [a.id for a in w.agents]
    assert len(ids) == len(set(ids)) and max(a.generation for a in w.agents) >= 1
    for a in w.agents:
        if a.parent_ids:
            m, f = (w.by_id[i] for i in a.parent_ids)
            assert a.id in m.children and a.id in f.children and a.generation == max(m.generation, f.generation) + 1
        if a.alive and a.partner_id is not None:
            b = w.by_id[a.partner_id]
            assert b.alive and b.partner_id == a.id
    queued = {c.id for c in w.pending}
    assert not queued & set(ids)


def test_policy_side_randomness_does_not_change_family_events():
    a, b = _run_world(burn=False), _run_world(burn=True)
    assert a.events == b.events and a.deaths == b.deaths
    assert any(e["event"] == "birth" for e in a.events)


def test_envrng_normal_is_keyed_and_roughly_standard():
    e = R.EnvRng(5)
    e.begin_day(3)
    assert e.normal(7, R.MUTATION, 2) == e.normal(7, R.MUTATION, 2)
    xs = []
    for i in range(3000):
        e.begin_day(i)
        xs.append(e.normal(1, R.MUTATION, 0))
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    assert abs(mean) < 0.08 and 0.9 < var < 1.1


# ======================= review round: the six findings ==========================

# ---- 1. estates settle after the whole day, independent of agent order ---------------------

def _blended(order):
    """A and B are spouses with one earlier child each (C, D); both die the same day."""
    w = _world(4)
    A, B, C, D = w.agents
    A.sex, B.sex = "M", "F"
    A.partner_id, B.partner_id = B.id, A.id
    C.parent_ids, D.parent_ids = (90, A.id), (91, B.id)
    A.children, B.children = [C.id], [D.id]
    A.cash, B.cash, C.cash, D.cash = 1000.0, 100.0, 0.0, 0.0
    w.agents[:] = [w.by_id[i] for i in order]
    return w, A, B, C, D


def test_same_day_spouse_deaths_do_not_chain_estates_and_ignore_list_order():
    results = set()
    for order in ([0, 1, 2, 3], [1, 0, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        w, A, B, C, D = _blended(order)
        _die(w, A, B)
        # each estate goes only to that person's own child: the dead spouse inherits nothing
        assert math.isclose(C.cash, 1000.0) and math.isclose(D.cash, 100.0)
        results.add((round(C.cash, 9), round(D.cash, 9)))
    assert len(results) == 1


def test_estate_is_settled_after_the_day_not_during_it():
    w, A, B, C, D = _blended([0, 1, 2, 3])
    w._die(A, "age")                              # marked dead, but nothing paid out yet
    assert C.cash == 0.0 and B.cash == 100.0
    w.settle_deaths()
    assert math.isclose(C.cash, 500.0) and math.isclose(B.cash, 100.0 + 500.0)   # B is alive: half to each


def test_heir_who_dies_the_same_day_is_skipped_and_trust_collects_from_both_parents():
    w = _world(3)
    mom, dad = _couple(w, cash=0.0)
    kid = w.agents[2]
    kid.parent_ids = (mom.id, dad.id)
    mom.children, dad.children = [kid.id], [kid.id]
    minor = _force_birth(w, mom, dad)
    mom.cash, dad.cash, kid.cash = 600.0, 400.0, 0.0
    _die(w, mom, dad, kid)                       # parents and the adult child all die today
    assert kid.cash == 0.0                       # the dead heir gets nothing
    assert math.isclose(minor.trust, 1000.0)     # the minor collects both estates (no partner survives)


def test_a_surviving_widow_still_gets_her_half_when_the_children_survive_too():
    w, A, B, C, D = _blended([0, 1, 2, 3])
    _die(w, A)
    assert math.isclose(C.cash, 500.0) and math.isclose(B.cash, 600.0) and B.partner_id is None


# ---- 2. kinship: ancestry to kinship_depth generations ----------------------------------------------

def _family_tree(n=12):
    w = _world(n)
    ag = w.agents
    g1, g2 = ag[0], ag[1]                                       # grandparents
    p1, p2 = ag[2], ag[3]                                       # their two children
    p1.parent_ids = p2.parent_ids = (g1.id, g2.id)
    c1, c2 = ag[4], ag[5]                                       # first cousins
    c1.parent_ids, c2.parent_ids = (p1.id, 70), (p2.id, 71)
    s1, s2 = ag[6], ag[7]                                       # children of the cousins: second cousins
    s1.parent_ids, s2.parent_ids = (c1.id, 72), (c2.id, 73)
    return w, dict(g1=g1, p1=p1, p2=p2, c1=c1, c2=c2, s1=s1, s2=s2)


def test_kinship_blocks_cousins_aunts_and_grandparents_but_not_second_cousins():
    w, t = _family_tree()
    rel = lambda x, y: family.related(w, t[x], t[y])             # noqa: E731
    assert rel("p1", "p2")                                       # siblings
    assert rel("c1", "c2")                                       # first cousins
    assert rel("p1", "c2")                                       # aunt/uncle and niece/nephew
    assert rel("g1", "c1")                                       # grandparent and grandchild
    assert rel("c1", "s1")                                       # parent and child
    assert not rel("s1", "s2")                                   # second cousins may pair
    assert not family.related(w, w.agents[8], w.agents[9])       # unrelated founders


def test_kinship_depth_is_configurable_and_match_respects_it():
    w, t = _family_tree()
    w.cfg.kinship_depth = 1
    assert not family.related(w, t["c1"], t["c2"])               # cousins pass at depth 1
    w.cfg.kinship_depth = 2
    # lone eligible cousins: sexes set, everyone else ineligible
    for a in w.agents:
        a.age_days = 70 * 365
    t["c1"].sex, t["c2"].sex = "F", "M"
    t["c1"].age_days = t["c2"].age_days = 25 * 365
    w.cfg.meet_rate = 1.0
    family.match(w)
    assert t["c1"].partner_id is None
    w.cfg.kinship_depth = 1
    family.match(w)
    assert t["c1"].partner_id == t["c2"].id


# ---- 3. the reserve gate is honestly named; wealth still acts indirectly ------------------------------

def test_reserve_gate_off_removes_only_the_direct_gate_and_debt_still_hurts_health():
    assert "birth_reserve_gate" in Config.__dataclass_fields__
    assert "fertility_needs_wealth" not in Config.__dataclass_fields__
    w = _world(2, birth_reserve_gate=False, birth_rate=1e9)
    mom, dad = w.agents
    mom.sex, dad.sex = "F", "M"
    mom.partner_id, dad.partner_id = dad.id, mom.id
    mom.cash = dad.cash = -1500.0
    for _ in range(220):
        w.tick()
    assert min(mom.health, dad.health) < w.cfg.min_parent_health   # poverty still blocks later births


# ---- 4. a hard population cap, with an order-independent winner ----------------------------------------

def _two_mothers(order):
    w = _world(6, max_population=5, birth_rate=1e9)
    ag = w.agents
    for f, m in ((ag[0], ag[1]), (ag[2], ag[3])):
        f.sex, m.sex = "F", "M"
        f.partner_id, m.partner_id = m.id, f.id
        f.cash = m.cash = 1e6
    ag[4].alive = ag[5].alive = False                    # 4 alive, cap 5: exactly one free slot
    w.agents[:] = [w.by_id[i] for i in order]
    return w


def test_births_never_exceed_free_slots_and_the_same_mother_wins_whatever_the_list_order():
    winners = set()
    for order in ([0, 1, 2, 3, 4, 5], [2, 3, 0, 1, 5, 4], [5, 4, 3, 2, 1, 0]):
        w = _two_mothers(order)
        family.births(w)
        assert len(w.pending) == 1
        assert sum(a.alive for a in w.agents) + len(w.pending) <= w.cfg.max_population
        winners.add(w.pending[0].mother_id)
    assert len(winners) == 1


def test_a_full_world_has_no_births_at_all():
    w = _two_mothers([0, 1, 2, 3, 4, 5])
    w.cfg.max_population = 4
    family.births(w)
    assert not w.pending


# ---- 5. output schema when reproduction is off ----------------------------------------------------------

_MAIN_OUTCOME_KEYS = ["id", "sex", "alive", "study_frac", "cash_buffer", "patience", "wealth_weight",
                      "years_lived", "age", "cash", "education", "career", "health", "job_level",
                      "taxes_paid", "n_illness", "n_job_losses", "chronic", "wealth_per_year"]


def test_outcomes_schema_is_unchanged_when_reproduction_is_off_and_extended_when_on():
    off = World(Config(seed=1, n_agents=3, years=1), lambda r, g: _Rest()).outcomes()[0]
    assert list(off) == _MAIN_OUTCOME_KEYS
    on = _world(3).outcomes()[0]
    extra = [k for k in on if k not in _MAIN_OUTCOME_KEYS]
    assert extra == ["generation", "n_children", "mother_id", "father_id", "inherited",
                     "inherited_real", "cash_real", "price_idx_end"]


# ---- 6. starter money and the inheritance baseline ---------------------------------------------------------

def test_new_adults_get_the_same_starter_money_as_founders_and_inheritance_counts_as_wealth():
    w = _world(2)
    mom, dad = _couple(w)
    ch = _force_birth(w, mom, dad)
    ch.trust = 250.0
    w.day = ch.birth_day + int(18 * 365)
    family.mature(w)
    kid = w.by_id[ch.id]
    assert kid.cash == w.cfg.start_cash + 250.0
    assert kid.start_cash == w.cfg.start_cash == w.agents[0].start_cash      # same baseline as founders
    assert kid.inherited == 250.0
    assert kid.cash - kid.start_cash == 250.0                                  # inheritance shows up as wealth gained


def test_adult_heirs_and_widows_accumulate_inherited():
    w, A, B, C, D = _blended([0, 1, 2, 3])
    _die(w, A)
    assert math.isclose(C.inherited, 500.0) and math.isclose(B.inherited, 500.0) and D.inherited == 0.0


# ================= round 2: matching order, real money, exposure metric, RL guard =================

# ---- 1. matching is independent of list order and ids -----------------------------------------------

def _pair_set(w):
    return frozenset(frozenset((a.id, a.partner_id)) for a in w.agents if a.partner_id is not None)


def _scarce(order, n_women=2, n_men=1, seed=1):
    w = _world(n_women + n_men, seed=seed, meet_rate=1.0)
    for i, a in enumerate(w.agents):
        a.sex = "F" if i < n_women else "M"
        a.age_days = 25 * 365
    w.agents[:] = [w.by_id[i] for i in order]
    return w


def test_scarce_partner_goes_to_the_same_woman_whatever_the_list_order():
    winners = set()
    for order in ([0, 1, 2], [1, 0, 2], [2, 1, 0], [2, 0, 1]):
        w = _scarce(order)
        family.match(w)
        winners.add(w.by_id[2].partner_id)
    assert len(winners) == 1


def test_one_woman_two_equal_men_picks_the_same_man_whatever_the_candidate_order():
    picks = set()
    for order in ([0, 1, 2], [0, 2, 1], [2, 1, 0], [1, 2, 0]):
        w = _world(3, meet_rate=1.0)
        for i, a in enumerate(w.agents):
            a.sex = "F" if i == 0 else "M"
            a.age_days = 25 * 365
            a.education = 30.0
        w.agents[:] = [w.by_id[i] for i in order]
        family.match(w)
        picks.add(w.by_id[0].partner_id)
    assert len(picks) == 1


def test_whole_matching_round_gives_the_same_pairs_when_the_agent_list_is_reversed():
    def pairs(reverse):
        w = _world(40, seed=7, meet_rate=0.6)
        for i, a in enumerate(w.agents):
            a.sex = "F" if i % 3 else "M"                 # more women than men: scarcity
            a.age_days = int((22 + i % 12) * 365)
            a.education = float(i * 7 % 60)
        if reverse:
            w.agents[:] = list(reversed(w.agents))
        family.match(w)
        return _pair_set(w)
    assert pairs(False) == pairs(True) and len(pairs(False)) > 5


def test_no_id_advantage_in_competition_for_a_scarce_partner():
    wins_low, trials = 0, 120
    for seed in range(trials):
        w = _scarce([0, 1, 2], seed=seed)
        family.match(w)
        wins_low += w.by_id[2].partner_id == 0            # woman with the lowest id
    assert 0.3 < wins_low / trials < 0.7                   # roughly a coin flip, not always id 0


# ---- 2. real money: starter amount, buffer gene and windfalls keep their meaning --------------------

def test_real_money_is_automatic_with_reproduction_and_can_be_forced_either_way():
    assert Config().uses_real_money is False
    assert Config(reproduction=True).uses_real_money is True
    assert Config(reproduction=True, real_money=False).uses_real_money is False
    assert Config(real_money=True).uses_real_money is True


def test_cash_buffer_gene_is_read_in_day_zero_money_in_real_money_mode():
    def act(real, price_idx):
        w = _world(2, real_money=real)
        w.prices.price_idx = price_idx
        from lifesim.policy import RulePolicy
        pol = RulePolicy(study_frac=1.0, cash_buffer=300.0, rng=w.policy_rng)
        pol.bind(w)
        obs = Agent(id=0, sex="M", age_days=9000, cash=600.0, health=90.0).observe()
        return pol.act(obs)
    assert act(real=True, price_idx=1.0) == Action.STUDY          # 600 > buffer 300
    assert act(real=True, price_idx=3.0) == Action.WORK           # buffer is now 900 nominal
    assert act(real=False, price_idx=3.0) == Action.STUDY         # nominal mode ignores prices (old behaviour)


def test_new_adults_get_starter_money_that_buys_the_same_at_any_date():
    def starter(real, price_idx):
        w = _world(2, real_money=real)
        mom, dad = _couple(w)
        ch = _force_birth(w, mom, dad)
        w.prices.price_idx = price_idx
        w.day = ch.birth_day + int(18 * 365)
        family.mature(w)
        return w.by_id[ch.id].cash
    assert math.isclose(starter(True, 7.0), 7.0 * Config().start_cash)
    assert starter(False, 7.0) == Config().start_cash             # nominal mode: unchanged behaviour


def test_windfalls_scale_with_prices_in_real_money_mode_only():
    from lifesim import shocks
    def mean_windfall(real, price_idx):
        w = _world(1, shocks=True, windfall_rate=1e9, illness_rate=0.0, job_loss_rate=0.0, real_money=real)
        w.prices.price_idx = price_idx
        a = w.agents[0]
        total, n = 0.0, 400
        for d in range(n):
            w.day = d
            w.env.begin_day(d)
            before = a.cash
            shocks.resolve(w, a)
            total += a.cash - before
        return total / n
    assert mean_windfall(True, 5.0) > 3.5 * mean_windfall(False, 5.0)


# ---- 3. the wealth rate is defined for new adults ----------------------------------------------------------

def _heir_with_trust(trust=250.0):
    w = _world(2)
    mom, dad = _couple(w)
    ch = _force_birth(w, mom, dad)
    ch.trust = trust
    w.day = ch.birth_day + int(18 * 365)
    family.mature(w)
    return w, w.by_id[ch.id]


def test_wealth_per_year_is_nan_without_adult_exposure_not_an_astronomical_number():
    w, kid = _heir_with_trust()
    row = next(r for r in w.outcomes() if r["id"] == kid.id)
    assert row["years_lived"] == 0.0 and math.isnan(row["wealth_per_year"])
    assert row["inherited_real"] == 250.0 and row["cash_real"] == w.cfg.start_cash + 250.0
    kid.age_days += 200                                           # a few months: still below min_rate_years
    assert math.isnan(next(r for r in w.outcomes() if r["id"] == kid.id)["wealth_per_year"])


def test_wealth_per_year_excludes_inheritance_once_exposure_is_enough():
    w, kid = _heir_with_trust()
    kid.age_days += 2 * 365                                       # two adult years, no earnings at all
    row = next(r for r in w.outcomes() if r["id"] == kid.id)
    assert math.isclose(row["wealth_per_year"], 0.0, abs_tol=1e-9)    # the 250 was a transfer, not accumulation
    kid.cash += 2000.0                                            # now they actually save something
    row = next(r for r in w.outcomes() if r["id"] == kid.id)
    assert math.isclose(row["wealth_per_year"], 1000.0)


def test_wealth_per_year_is_in_real_money_for_the_dead_too():
    w, kid = _heir_with_trust()
    kid.age_days += 3 * 365
    kid.cash = w.cfg.start_cash * 4 + 250.0 + 4000.0               # 4000 nominal saved while prices were x4
    w.prices.price_idx = 4.0
    w._die(kid, "age")
    w.prices.price_idx = 10.0                                      # later inflation must not move a dead agent's figures
    row = next(r for r in w.outcomes() if r["id"] == kid.id)
    assert row["price_idx_end"] == 4.0 and math.isclose(row["cash_real"], w.cfg.start_cash + 250.0 / 4 + 1000.0)


def test_off_mode_wealth_per_year_formula_is_untouched():
    w = World(Config(seed=1, n_agents=2, years=1), lambda r, g: _Rest())
    a = w.agents[0]
    a.cash, a.start_cash = 700.0, 200.0
    a.age_days = a.start_age_days + 365
    assert math.isclose(next(r for r in w.outcomes() if r["id"] == a.id)["wealth_per_year"], 500.0)


# ---- 4. the RL pipeline refuses reproduction ----------------------------------------------------------------------

def test_es_trainer_and_evaluator_refuse_reproduction(tmp_path):
    import numpy as np
    import pytest

    from lifesim.rl.es import evaluate, train
    from lifesim.rl.mlp import MLPPolicy
    from lifesim.rl.reward import Reward
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="reproduction"):
        train(out, {"n_agents": 4, "years": 1, "reproduction": True}, Reward(), hidden=(8,),
              generations=1, pairs=1, workers=1, clone_worlds=0, say=lambda s: None)
    assert not out.exists()                                       # refused up front: no output, no wasted work
    with pytest.raises(ValueError, match="reproduction"):
        evaluate(MLPPolicy().get_flat(), [1], {"n_agents": 4, "years": 1, "reproduction": True},
                 Reward(), (32, 32))
    assert np.isfinite(evaluate(MLPPolicy().get_flat(), [1], {"n_agents": 4, "years": 1},
                                Reward(), (32, 32))["fitness"])        # the normal path still works


# ================= round 3: trust accounting, ablate guard, minors in outputs, real reporting =================

# ---- 1. a minor's trust is nominal cash; the adult baseline is its value at entry --------------------------

def test_inflation_before_adulthood_is_not_charged_to_the_heirs_accumulation():
    w = _world(2)
    mom, dad = _couple(w, cash=0.0)
    ch = _force_birth(w, mom, dad)
    mom.alive = False
    mom.partner_id = dad.partner_id = None
    dad.cash = 2000.0
    w.prices.price_idx = 1.0
    _die(w, dad)                                         # the child is the only heir
    assert ch.trust == 2000.0
    w.prices.price_idx = 2.0                             # prices double before the child turns 18
    w.day = ch.birth_day + int(18 * 365)
    family.mature(w)
    kid = w.by_id[ch.id]
    assert kid.cash == 2 * w.cfg.start_cash + 2000.0     # the trust is paid out in nominal cash, no money created
    assert kid.inherited_real == 1000.0                  # worth 1000 day-0 units at entry, not the 2000 at receipt
    kid.age_days += 2 * 365                              # two adult years in which nothing happens
    row = next(r for r in w.outcomes() if r["id"] == kid.id)
    assert math.isclose(row["wealth_per_year"], 0.0, abs_tol=1e-9)    # was -500 a year before the fix


def test_lineage_reports_a_minors_trust_at_its_current_real_value():
    w = _world(2)
    mom, dad = _couple(w, cash=0.0)
    ch = _force_birth(w, mom, dad)
    ch.trust = 600.0
    w.prices.price_idx = 3.0
    row = next(r for r in w.lineage() if r["id"] == ch.id)
    assert row["status"] == "minor" and row["inherited"] == 600.0 and row["inherited_real"] == 200.0


# ---- 2. ablate refuses reproduction -----------------------------------------------------------------------

def test_ablate_refuses_reproduction_in_the_function_and_fails_fast_in_the_cli(monkeypatch):
    import sys

    import pytest

    from lifesim import ablate
    with pytest.raises(ValueError, match="reproduction"):
        ablate.summarize({"reproduction": True}, seeds=1, agents=2, years=1, policy="rule")

    def must_not_run(*a, **k):
        raise AssertionError("a slow run started before the guard fired")
    monkeypatch.setattr(ablate, "summarize", must_not_run)
    for argv in (["ablate", "--set", "reproduction=True"],                       # in the override
                 ["ablate", "--base", "reproduction=True", "--set", "shocks=False"]):  # only in the base
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit, match="reproduction"):
            ablate.main()


def test_all_fixed_population_tools_share_one_guard():
    import pytest

    from lifesim.guards import require_fixed_population
    require_fixed_population({"reproduction": False}, "x", "y")
    require_fixed_population({}, "x", "y")
    with pytest.raises(ValueError, match="reproduction=True is not supported by thing"):
        require_fixed_population({"reproduction": True}, "thing", "do this instead")


# ---- 3. children who have not come of age are part of the population and the outputs ------------------------

def _young_world():
    w = World(Config(seed=2, n_agents=30, years=12, reproduction=True, shocks=False, meet_rate=0.5,
                     birth_rate=1.0, birth_reserve_gate=False, log_every=365),
              lambda r, g: _Rest())
    w.run()
    return w


def test_lineage_includes_queued_children_and_every_listed_child_is_in_the_file():
    w = _young_world()
    assert w.pending                                              # the case the bug hid
    lin = w.lineage()
    ids = [r["id"] for r in lin]
    assert len(ids) == len(set(ids))
    assert {c for a in w.agents for c in a.children} <= set(ids)       # no dangling child ids
    assert {r["mother_id"] for r in lin if r["mother_id"] is not None} <= set(ids)
    minors = [r for r in lin if r["status"] == "minor"]
    assert len(minors) == len(w.pending)
    assert all(r["alive"] and r["entered_day"] is None and r["n_children"] == 0 for r in minors)
    assert {r["status"] for r in lin} == {"adult", "minor"}


def test_population_reports_adults_children_and_total_and_matches_the_cap():
    w = _young_world()
    pop = w.population()
    assert pop["alive_adults"] == sum(a.alive for a in w.agents)
    assert pop["pending_children"] == len(w.pending) > 0
    assert pop["population_total"] == pop["alive_adults"] + pop["pending_children"] == family._occupants(w)
    log = w.population_log
    assert len(log) == 12 and [r["day"] for r in log] == [365 * (i + 1) for i in range(12)]
    assert all(r["population_total"] == r["alive_adults"] + r["pending_children"] for r in log)
    off = World(Config(seed=1, n_agents=3, years=2, log_every=365), lambda r, g: _Rest())
    off.run()
    assert off.population_log == []                               # nothing new when reproduction is off


def test_cli_reports_children_separately_and_writes_population_csv(monkeypatch, tmp_path, capsys):
    import sys

    import pandas as pd

    from lifesim import run
    monkeypatch.setattr(sys, "argv", ["run", "--agents", "30", "--years", "12", "--set", "reproduction=True",
                                      "birth_rate=1.0", "birth_reserve_gate=False", "meet_rate=0.5",
                                      "--out", str(tmp_path)])
    run.main()
    text = capsys.readouterr().out
    assert "alive adults" in text and "children not yet adults" in text and "people" in text
    lin = pd.read_csv(tmp_path / "lineage.csv")
    pop = pd.read_csv(tmp_path / "population.csv")
    assert "status" in lin.columns and {"alive_adults", "pending_children", "population_total"} <= set(pop.columns)
    assert set(lin.mother_id.dropna().astype(int)) <= set(lin.id)
    off_dir = tmp_path / "off"
    monkeypatch.setattr(sys, "argv", ["run", "--agents", "10", "--years", "1", "--out", str(off_dir)])
    run.main()
    assert "alive adults" not in capsys.readouterr().out and not (off_dir / "population.csv").exists()


# ---- 4. real-money reporting follows the money convention, not reproduction ---------------------------------------

def test_real_money_without_reproduction_reports_in_real_money():
    w = World(Config(seed=1, n_agents=2, years=1, real_money=True, shocks=False), lambda r, g: _Rest())
    a = w.agents[0]
    a.age_days = a.start_age_days + 3 * 365
    a.cash, a.start_cash = 600.0, 200.0
    w.prices.price_idx = 2.0
    row = next(r for r in w.outcomes() if r["id"] == a.id)
    assert row["cash_real"] == 300.0 and row["price_idx_end"] == 2.0 and row["inherited_real"] == 0.0
    assert math.isclose(row["wealth_per_year"], (300.0 - 200.0) / 3)
    assert "generation" not in row and "inherited" not in row          # family columns still need reproduction


def test_reproduction_with_nominal_money_reports_nominal_figures_with_the_exposure_guard():
    w = _world(2, real_money=False)
    row = w.outcomes()[0]
    assert "cash_real" not in row and "inherited_real" not in row and "generation" in row
    mom, dad = _couple(w)
    ch = _force_birth(w, mom, dad)
    ch.trust = 250.0
    w.day = ch.birth_day + int(18 * 365)
    family.mature(w)
    kid = w.by_id[ch.id]
    new = next(r for r in w.outcomes() if r["id"] == kid.id)
    assert math.isnan(new["wealth_per_year"])                         # no absurd rate for a fresh adult
    kid.age_days += 2 * 365
    later = next(r for r in w.outcomes() if r["id"] == kid.id)
    assert math.isclose(later["wealth_per_year"], 0.0, abs_tol=1e-9)    # inheritance excluded here too


# ================= round 4: investment basis, partnership timing, RL real-money, assortative softmax =================

# ---- 1. investment depends on household wealth, not on how many parents hold it ----------------------------

def _invest_part(cash_mom, cash_dad, dad_alive=True):
    w = _world(2, child_invest_frac=0.02)
    mom, dad = _couple(w, cash=0.0)
    mom.cash, dad.cash = cash_mom, cash_dad
    ch = _force_birth(w, mom, dad)
    dad.alive = dad_alive
    family.pay_child_costs(w)
    return ch.spent_real - w.cfg.child_cost                   # the wealth-driven part only


def test_same_household_wealth_gives_the_same_investment_whoever_holds_it():
    two = _invest_part(100_000.0, 100_000.0)
    one = _invest_part(200_000.0, 0.0, dad_alive=False)       # a parent has died: the survivor holds it all
    assert math.isclose(two, one) and two > 0


def test_a_second_equally_wealthy_parent_doubles_the_wealth_basis():
    alone = _invest_part(100_000.0, 0.0, dad_alive=False)
    assert math.isclose(_invest_part(100_000.0, 100_000.0), 2 * alone)


def test_a_parents_debt_does_not_reduce_the_other_parents_contribution():
    assert math.isclose(_invest_part(100_000.0, -50_000.0), _invest_part(100_000.0, 0.0))
    assert _invest_part(-50_000.0, -50_000.0) == 0.0


# ---- 2. a couple cannot have a child the day it forms ------------------------------------------------------------

def _new_couple_world(**cfg):
    w = _world(2, birth_rate=1e9, meet_rate=1.0, birth_reserve_gate=False, max_population=100, **cfg)
    f, m = w.agents
    f.sex, m.sex = "F", "M"
    f.age_days = m.age_days = 25 * 365
    f.last_birth_day = m.last_birth_day = -10**9
    return w, f, m


def test_births_run_before_matching_so_a_couple_formed_today_waits_until_tomorrow():
    w, f, m = _new_couple_world(min_partnership_days=0)       # the order alone must protect us
    w.day = 0                                                 # a matching day
    family.step(w)
    assert f.partner_id == m.id and not w.pending             # paired today, no child today
    w.day = 1
    family.step(w)
    assert len(w.pending) == 1                                # eligible from the next day


def test_a_new_couple_waits_the_gestation_period_and_pair_day_is_recorded():
    w, f, m = _new_couple_world()
    w.day = 0
    family.step(w)
    assert f.pair_day == m.pair_day == 0
    for day in (1, 100, w.cfg.min_partnership_days - 1):
        w.day = day
        family.births(w)
        assert not w.pending, day
    w.day = w.cfg.min_partnership_days
    family.births(w)
    assert len(w.pending) == 1


def test_a_widow_who_re_pairs_does_not_give_the_new_partner_a_same_day_child():
    w = _world(3, birth_rate=1e9, meet_rate=1.0, birth_reserve_gate=False, max_population=100)
    mom, old, new = w.agents
    mom.sex, old.sex, new.sex = "F", "M", "M"
    for a in w.agents:
        a.age_days = 28 * 365
    mom.partner_id, old.partner_id = old.id, mom.id
    mom.pair_day = old.pair_day = -2000
    mom.last_birth_day = -10**9
    w.day = 30                                                # a matching day
    w._die(old, "age")
    w.settle_deaths()
    family.step(w)
    assert mom.partner_id == new.id and not w.pending         # re-paired, but no child that day
    w.day = 30 + w.cfg.min_partnership_days
    family.births(w)
    assert len(w.pending) == 1 and w.pending[0].father_id == new.id


def test_hand_built_couples_without_a_pair_day_are_not_blocked():
    w = _world(2, birth_rate=1e9, birth_reserve_gate=False)
    mom, dad = _couple(w)
    assert mom.pair_day is None
    family.births(w)
    assert len(w.pending) == 1


# ---- 3. the RL tools refuse real_money=True; ablate reports money uniformly -------------------------------------------

def test_es_trainer_and_evaluator_refuse_real_money(tmp_path):
    import pytest

    from lifesim.rl.es import check_supported, evaluate, train
    from lifesim.rl.mlp import MLPPolicy
    check_supported({"real_money": False})
    check_supported({})
    with pytest.raises(ValueError, match="real_money"):
        check_supported({"real_money": True})
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="real_money"):
        train(out, {"n_agents": 4, "years": 1, "real_money": True}, Reward(), hidden=(8,),
              generations=1, pairs=1, workers=1, clone_worlds=0, say=lambda s: None)
    assert not out.exists()
    with pytest.raises(ValueError, match="real_money"):
        evaluate(MLPPolicy().get_flat(), [1], {"n_agents": 4, "years": 1, "real_money": True}, Reward(), (32, 32))


def test_evaluator_cli_refuses_real_money(monkeypatch):
    import sys

    import pytest

    from lifesim.rl import evaluate as ev
    monkeypatch.setattr(sys, "argv", ["evaluate", "--theta", "x.npz", "--set", "real_money=True"])
    with pytest.raises(SystemExit, match="real_money"):
        ev.main()


def test_ablate_reports_nominal_and_real_money_the_same_way_in_every_arm():
    from lifesim.ablate import summarize
    kw = dict(seeds=1, agents=6, years=3, policy="rule")
    nominal_arm = summarize({}, **kw)
    real_arm = summarize({"real_money": True}, **kw)
    for res in (nominal_arm, real_arm):
        assert {"median_final_cash", "median_wealth_per_year", "median_final_cash_real",
                "median_wealth_per_year_real"} <= set(res)
        assert res["median_final_cash_real"] < res["median_final_cash"]          # prices rose over 3 years
    # the legacy nominal figure is the legacy formula in the nominal arm ...
    w = World(Config(seed=0, n_agents=6, years=3), lambda rng, g: make_policy("rule", rng, g))
    w.run()
    legacy = float(np.median([r["wealth_per_year"] for r in w.outcomes()]))
    assert math.isclose(nominal_arm["median_wealth_per_year"], legacy)
    # ... and in the real-money arm it is still the nominal figure, not the real-mode column
    w2 = World(Config(seed=0, n_agents=6, years=3, real_money=True), lambda rng, g: make_policy("rule", rng, g))
    w2.run()
    nominal_again = float(np.median([(a.cash - a.start_cash) / ((a.age_days - a.start_age_days) / 365) for a in w2.agents]))
    assert math.isclose(real_arm["median_wealth_per_year"], nominal_again)


# ---- 4. a strong assortative preference no longer underflows to "highest id wins" ---------------------------------------

def _pick(strength, edu, seed):
    w = _world(4, seed=seed, assortative_mating=strength, meet_rate=1.0)
    f, *men = w.agents
    f.sex, f.education, f.age_days = "F", 10.0, 30 * 365
    for m, e in zip(men, edu):
        m.sex, m.education, m.age_days = "M", e, 30 * 365
    family.match(w)
    return f.partner_id


def test_extreme_assortative_strength_still_picks_the_closest_candidate():
    for strength in (5.0, 1e3, 1e5, 1e9):
        picks = {_pick(strength, (50.0, 90.0, 130.0), seed) for seed in range(20)}
        assert picks == {1}, (strength, picks)                    # the man at education 50, not the highest id


def test_extreme_assortative_strength_with_equal_gaps_is_not_decided_by_id():
    picks = {_pick(1e9, (50.0, 50.0, 50.0), seed) for seed in range(40)}
    assert len(picks) >= 2                                        # a fair roulette, not always the last id


def test_non_finite_preference_degrades_to_a_uniform_pick_instead_of_crashing():
    picks = {_pick(1e308, (50.0, 90.0, 130.0), seed) for seed in range(40)}
    assert len(picks) >= 2 and picks <= {1, 2, 3}
