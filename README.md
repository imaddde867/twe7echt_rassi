# lifesim

A small agent-based life simulation. Agents (sex, age, health, energy, satiety,
mood, cash, education, career) pick 3 actions per day (work, study, eat, rest,
socialize) and live until health or old age kills them. Built for fun first;
research questions later.

```
pip install -r requirements-dev.txt     # runtime deps + pytest + ruff
python -m lifesim.run --policy rule   --years 50 --out runs/rule
python -m lifesim.run --policy random --years 50 --out runs/random
ruff check . && python -m pytest        # what CI runs (.github/workflows/ci.yml)
```

Each run writes `snapshots.csv` (monthly agent stats), `deaths.csv`, `events.csv` (shocks), `outcomes.csv` (genes + result per agent) and
`summary.png` into the output folder.

## Changing the world
Every number lives in `config.py`. Override from the CLI without editing code:
`python -m lifesim.run --set retire_age=70 pension_frac=0.3`.
Measure what a feature does by comparing configs over several seeds:
`python -m lifesim.ablate --base life_stages=False --set life_stages=True`.
Every ablation below switches ONE layer on or off in an otherwise default world.

## Randomness
Initial conditions (ages, sex, genes) come from one seeded stream. Policies get
their own stream (`world.policy_rng`). All exogenous events (illness, layoffs,
windfalls, mortality, and their severities and durations) are drawn by `rng.py`
as a pure function of (seed, day, agent, event kind), so two runs with the same
seed see the same luck even after their trajectories diverge, even if one policy
burns random numbers, and even if a conditional draw fires in one run and not
the other. This pairs the raw draws; it does not make outcomes identical, because
whether a draw triggers an event still depends on state (health, employment).
Regression tests cover this (`tests/test_review_fixes.py`).

## Layout
| file | job |
|---|---|
| `config.py` | every tunable number |
| `agent.py` | agent state + `observe()` vector |
| `actions.py` | what each action does to the stats (the sim's assumptions) |
| `policy.py` | `act(obs) -> Action`. Random and rule-based now, RL later |
| `world.py` | daily loop, aging, mortality, logging |
| `shocks.py` | random life events: illness, job loss, windfall |
| `ablate.py` | compare two configs over several seeds |
| `family.py` | couples, births, inheritance (generations, off by default) |
| `rng.py` | keyed environmental randomness |

## Known limits of v1
- Strategy genes (`study_frac`, `cash_buffer`) are random at birth and fixed for
  life. Survival barely depends on them (corr ~0.01), only wealth does, so
  there is no selection pressure until births + inheritance exist.
- The economy numbers (tiers, tax, welfare, inflation) were tuned by eye and
  then re-measured after a job-tier bug fix (see layer 3). Treat them as
  assumptions, not calibrations.
- `restricted_mean_years` estimates E[min(T, horizon)], the mean years lived
  inside the simulated window. It is not life expectancy. Median age at death is
  biased by the same fixed window.
- Differences of a few tenths of a year between configs were not tested for
  significance (180 agents per arm); treat them as small and uncertain.
- No births yet, so no generations. Population only shrinks.
- Sex mortality multipliers are neutral (1.0) on purpose.

## Generations (`reproduction`, off by default)
Step A of generations: couples, births, inheritance (`lifesim/family.py`).
Turn on with `--set reproduction=True` and run long enough to see several
generations (a generation is about 30 years): `--years 150`. With it off, the four
data files (`outcomes.csv`, `events.csv`, `deaths.csv`, `snapshots.csv`) are
byte-identical to `main` for the same seed (checked for the rule and utility
policies), and the output schema is unchanged.
- **Couples:** every 30 days unpaired adults aged 20 to 40 can meet an opposite-sex
  partner within 8 years of their age. **Kinship:** two people never pair if one is
  an ancestor of the other, or they share an ancestor within `kinship_depth`
  generations (default 2: parent-child, siblings, half-siblings, aunt/uncle-niece,
  grandparent-grandchild and first cousins are blocked; second cousins may pair).
  Founders have no recorded ancestors, so they count as unrelated by assumption.
  `assortative_mating` (default 0 = random) prefers similar education. Partners pay
  a discounted living cost (`household_cost_factor`), cash is not pooled. No
  divorce; a widow can re-pair. **Matching does not depend on list order or id:**
  women who attempt are processed in a keyed random order, candidate men are
  enumerated by id, and the roulette draw is keyed, so nobody gets first pick of
  scarce partners because of an early birth or a low id (tested, including a
  coin-flip check on who wins a scarce partner).
- **Births:** an eligible couple (both healthy, mother 20 to 40, spacing, enough
  cash) has a child at `birth_rate` per year, scaled down as the population nears
  `max_population`, **and** births never exceed the free slots (a hard cap; when
  mothers compete for the last slots a keyed draw decides, not list order).
  `birth_reserve_gate=False` removes only the direct cash-reserve requirement. It is
  **not** a wealth-neutral control: debt still damages health (which gates births)
  and parental spending still buys schooling. Separating selection from drift needs
  a neutral marker gene that is inherited and mutated like the others but never read
  by a policy; that is part of step B.
- **Children are queued, not simulated.** Parents pay `child_cost` per day plus an
  extra share of their wealth (capped) until 18. At 18 the child appears as an
  adult: each gene comes from a random parent plus Gaussian noise
  (`mutation_sigma`, clipped to the gene's range), schooling rises with the
  parents' total spending (saturating), and any inheritance held in trust is paid
  out. Every new adult also gets the same starter money founders got, in day-0
  money (see **Real money** below), a deliberate small injection so generations
  are comparable.
- **Real money (`real_money`, automatic with reproduction):** prices inflate (2% a
  year by default, so a price index of about 7 at year 100 and 19.5 at year 150),
  so a fixed nominal amount would lose its meaning over generations, and because
  `cash_buffer` is a heritable gene that would look like evolution. In real-money
  mode the `cash_buffer` gene is read in day-0 money (scaled by the price index when
  the rule policy compares it with cash), the starter money and the mean windfall
  scale with the price index, and outcomes report real figures. Set
  `real_money=False` to get the old nominal behaviour (that is also what every run
  without reproduction does, so earlier results are unchanged). Money amounts
  that were already in day-0 terms (child cost, credit limit, healthcare) were
  scaled before.
- **Estate:** at death, half of positive cash goes to the children (adults and
  minors in trust), half to the surviving partner; with no partner all of it goes
  to the children; with no children it goes to the partner, else it is lost. Debt
  is forgiven. **Simultaneous deaths:** estates are settled after every agent has
  lived the day, in id order. Only people alive at the end of the day inherit: a
  partner or child who dies the same day gets nothing and passes nothing on. So the
  result does not depend on the order of the agent list. A dead agent's `cash` stays
  at its value at death (not zeroed), so do not sum cash over dead agents.
- **Outputs (only with reproduction on):** `lineage.csv` (parents, generation,
  inherited, genes), `pair`/`birth`/`adult`/`estate` rows in `events.csv`, and
  `generation`, `n_children`, `mother_id`, `father_id`, `inherited`,
  `inherited_real`, `cash_real`, `price_idx_end` in `outcomes.csv`.
  **`wealth_per_year` changes meaning in this mode:** it is real accumulation
  excluding transfers, `(cash_real - start_cash - inherited_real) / years`, and NaN
  until an agent has `min_rate_years` (1) of adult exposure, so a freshly matured
  heir no longer produces an absurd rate. Inheritance stays visible in
  `inherited_real` and `cash_real`. Dead agents are valued at the price index of
  their death day.
- **Population (rule agents, 100 founders, 150 years, seed 1, current code,
  real-money mode):** birth rate 0.3 settles near 70 to 85 alive; 0.5 (the default)
  grows to about 146 by year 100 and about 198 by year 150; 1.0 peaks near 232 at
  year 100 and then falls to about 162 as the founder cohort ages out (a cohort
  echo). Generations 5 to 6 are reached. The defaults were set by eye; every number
  is an assumption.
- **Not built yet:** step B (per-generation gene means, parent-child wealth
  correlation, inequality, a neutral marker gene as the drift control, and the
  ablations `assortative_mating` 0 vs high). A first single-seed run showed mean
  `study_frac` falling across generations (0.53, 0.40, 0.38); that is a hint, not a
  finding.
- **Not supported with the RL pipeline:** the ES trainer and evaluator refuse
  `reproduction=True`. Their reward is defined for a fixed population (it divides
  `years_lived` by the whole horizon, which would penalise descendants for not
  existing earlier, and dead agents keep cash that was already passed on). Run
  generations with `lifesim.run`. The RL network is one shared brain with no genes,
  so nothing is inherited by it.

## Learning a policy (RL)
`lifesim/rl/` trains one shared network (numpy MLP, no torch) with evolution
strategies. Every candidate network is the brain of a whole world of agents and
is scored by `rl/reward.py`: years lived, final wealth and health. Steps:
1. **Behaviour cloning** (`rl/clone.py`): copy the rule policy into the network,
   so search starts from a policy that already survives.
2. **ES** (`rl/es.py`): antithetic sampling, rank-shaped fitness, Adam, weight
   decay. All candidates see the same world seeds in a generation. Worlds are
   independent, so it scales linearly over cores (`--workers`). Checkpoints every
   generation (`latest.npz`, including the RNG state, so an interrupted and
   resumed run matches an uninterrupted one), resumable with `--resume`;
   `log.csv` has the curve. `best.npz` holds exactly the weights that earned its
   recorded fitness and is never overwritten by a worse generation, also across
   resumes.
3. **Held-out evaluation** (`rl/evaluate.py`): network vs rule vs random on seeds
   never used in training.

```
python -m lifesim.rl.es --out runs/rl --generations 300 --pairs 32 --agents 20 --years 20
python -m lifesim.rl.evaluate --theta runs/rl/best.npz --seeds 20 --agents 40 --years 30
python -m lifesim.run --policy mlp --theta runs/rl/best.npz --years 50   # watch it live a life
```
Cost (measured on 4 cores of a sandbox): one world of 20 agents x 20 years takes
about 8 s, 40 agents x 40 years about 31 s. A generation with 32 pairs x 2 seeds
is about 130 worlds, so roughly 18 CPU-minutes at the small size: on 64 cores,
about 20 s per generation. The reward is the assumption that decides the result
(`--reward w_wealth=1.0 ...`). The learned brain is shared, so agents differ only
by state and luck, not by genes. On CSC: `slurm/train_es.sbatch` (fill in the
placeholders; partition and module names are not verified for Roihu).

## Layers (single-life realism)
Numbers below are from the current code (3 seeds x 60 agents x 50 years, rule
policy), one layer switched off vs the default world. The default ("on") arm is
identical in all three: alive 47.8%, restricted mean years 43.5, median final
cash 138k, median wealth/year 3.6k.

| layer off -> on | alive | restricted mean years | median wealth/year |
|---|---|---|---|
| `life_stages` | 50.0% -> 47.8% | 43.85 -> 43.51 | 9.2k -> 3.6k |
| `shocks` | 50.6% -> 47.8% | 44.29 -> 43.51 | 5.5k -> 3.6k |
| `economy` | 50.6% -> 47.8% | 44.02 -> 43.51 | 12.3k -> 3.6k |

1. **Life stages** (`life_stages`): mandatory retirement at 65, pension from
   lifetime earnings, slower energy recovery when old. Its clear effect is on
   wealth; survival barely moves because death is mostly the age-based mortality
   curve.
2. **Random shocks** (`shocks`): illness (likelier when old or unhealthy; paying
   for treatment cuts the damage to 30%), chronic conditions, job loss with
   benefits, windfalls. Sick, laid-off or retired agents cannot work. Events go
   to `events.csv`. Lifespan cost about 0.8 years, wealth about a third.
3. **Jobs and money** (`economy`): discrete job tiers gated by education and
   career (an employed agent is promoted the moment they qualify), progressive
   tax, prices and pay that grow over time, lifestyle creep, interest on savings
   and debt, a credit limit past which you cannot buy food, and a welfare floor.
   **Correction:** an earlier version only re-evaluated the tier after a layoff,
   so agents kept old pay for years. Fixing it changed the picture: rule agents
   now end rich (median 100k+ at 30 years), not stuck near the credit limit.
   Does saving protect lifespan? Not visibly. With a wide `cash_buffer` range
   (50 to 30000) about 88% of illnesses get treated in every quartile and the
   correlation of buffer with years lived is 0.013. Agents are rich enough that
   money is rarely the binding constraint, so the money -> treatment -> health
   -> lifespan link exists in the code (unit-tested) but is not exercised.
8. **Utility policy** (experimental, `--policy utility`): one-step lookahead.
   The agent tries each action on a scratch copy of itself (reusing
   `actions.apply`) and picks the best resulting state under a utility with
   needs, health, mood, concave cash, expected future pay from education and
   career, an emergency-fund target and a wall at the credit limit. Genes:
   `patience`, `wealth_weight`. **It loses to the rule policy** (2 seeds x 40
   agents x 30y): alive 41% vs 88%, restricted mean years 16.5 vs 28.2, median
   final cash -144 vs 104k. In a single-seed diagnosis only 11% of planner
   illnesses get treated vs 65% for rule agents, all planner deaths are health
   collapse, median around year 3. The planner hovers near zero cash and never
   builds a reserve. Weights are hand-tuned guesses, not optimised. Adults start
   with varied schooling (`start_education`).
4. Body (fitness, diet). 5. Time and place. 6. Social. 7. Traits/background.
   Not built.

## Roadmap
1. ~~Per-agent heterogeneity~~ done. `outcomes.csv` has one row per agent.
2. Births and inheritance (what passes to the child?).
3. Swap `RulePolicy` for a shared RL policy; judge it against the baselines.
4. 2D visualization.
