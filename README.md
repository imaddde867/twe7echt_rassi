# lifesim

A small agent-based life simulation. Agents (sex, age, health, energy, satiety, mood, cash, education, career)
pick three actions a day (work, study, eat, rest, socialize) and live until health or old age kills them. One tick
is one day. On top of the single-life model sit shocks, an economy, retirement, couples, births and inheritance
(generations), and a way to learn a policy with evolution strategies. Built for fun first; research questions later.

```
pip install -r requirements-dev.txt        # runtime deps + pytest + ruff
python -m lifesim.run --policy rule --years 50 --out runs/rule
ruff check . && python -m pytest           # what CI runs (.github/workflows/ci.yml)
scripts/check_off_mode.sh                  # features that are off must leave the outputs byte-identical to main
```

## Where things stand

**Built and tested** (see `tests/`; every behaviour change ships with regression tests that were checked to fail
when the bug is put back):
- Single life: life stages (retirement, pension), random shocks (illness, job loss, windfalls), an economy (job
  tiers, tax, inflation, credit limit, welfare).
- Policies: `random`, `rule` (the baseline), `utility` (one-step planner, experimental, loses to `rule`), and a
  shared neural network trained by evolution strategies (`lifesim/rl`).
- Generations (off by default): couples, births, inheritance, real money, and `lifesim.generations`, which runs
  births-on experiments over many seeds and separates selection from drift with a neutral marker gene.

**Never run at real scale yet** (these are the open actions, and they need a cluster):
- The ES trainer has only run at toy size (8 agents, a few generations). Whether the learned network beats the
  rule policy at 20+ years is unknown: issue #2, `slurm/train_es.sbatch`.
- The generations experiment has not been run with enough seeds to say anything about selection: run
  `slurm/generations.sbatch` and read `summary.csv` (see **Generations** below). The one smoke-run hint
  (`study_frac` falling across generations) is only a hint.
- Both Slurm scripts have placeholders; partition, module and account names are not verified for CSC Roihu.

**Known weak spots:** the utility planner loses to the rule policy (below); every economy and family default is
an assumption set by eye, not a calibration; layers 4 to 7 (body, place, social, traits) are not built.

## Layout

| path | job |
|---|---|
| `lifesim/config.py` | every tunable number (override from the CLI with `--set key=value`) |
| `lifesim/agent.py` | agent state and the observation vector |
| `lifesim/actions.py` | what each action does to the stats (the model's assumptions) |
| `lifesim/world.py` | daily loop, aging, mortality, logging, outputs |
| `lifesim/economy.py` | job tiers, tax, prices, lifestyle creep, interest, credit limit, household discount |
| `lifesim/shocks.py` | illness, chronic conditions, job loss, windfalls |
| `lifesim/family.py` | couples, births, children, inheritance (off by default) |
| `lifesim/policy.py` | `act(obs) -> Action`: random, rule, utility |
| `lifesim/rng.py` | keyed environmental randomness |
| `lifesim/run.py` | one run, writes the output files |
| `lifesim/ablate.py` | compare two configs over several seeds (fixed population only) |
| `lifesim/generations.py` | generation-aware experiments over seeds, drift null, parent-child wealth, Gini |
| `lifesim/guards.py` | refusals for tools that assume a fixed population or nominal money |
| `lifesim/parallel.py` | `default_workers()` (respects a Slurm allocation) |
| `lifesim/plot.py` | the summary figure |
| `lifesim/rl/` | numpy MLP policy, behaviour cloning, evolution strategies, evaluator, reward |
| `slurm/` | `train_es.sbatch`, `generations.sbatch` (placeholders for CSC) |
| `scripts/check_off_mode.sh` | byte-identity check of reproduction-off outputs against a git ref |
| `tests/` | `test_sim.py`, `test_review_fixes.py`, `test_pension.py`, `test_family.py`, `test_generations.py`, `test_rl.py` |

## Running things

**One run.** `python -m lifesim.run --policy rule --years 50 --out runs/rule` writes `snapshots.csv` (agent stats
every `log_every` days, 30 by default), `deaths.csv`, `events.csv`, `outcomes.csv` (genes and result per agent)
and `summary.png`. Change any number without editing code: `--set retire_age=70 pension_frac=0.3`. With births on
(`--set reproduction=True`) it also writes `lineage.csv` and `population.csv`.

**Compare configs.** `python -m lifesim.ablate --base life_stages=False --set life_stages=True` runs both over
several seeds and prints a table. It reports money in nominal and real (day-0) terms in every arm. It refuses
`reproduction=True`: its statistics assume a fixed population.

**Generations.** `python -m lifesim.generations --seeds 8 --years 150 --out runs/gen` (see **Generations**).

**Learn a policy.** `python -m lifesim.rl.es --out runs/rl ...`, then `python -m lifesim.rl.evaluate --theta
runs/rl/best.npz ...` (see **Learning a policy**).

## Randomness and invariants

Initial conditions (ages, sex, genes) come from one seeded stream. Policies get their own stream
(`world.policy_rng`). Every exogenous event (illness, layoffs, windfalls, mortality, matching, births, crossover,
mutation, and their severities and durations) is a pure function of (seed, day, agent id, event kind) in
`rng.py`. So two runs with the same seed see the same luck even after their trajectories diverge, even if a
policy burns random numbers, and even if a conditional draw fires in one run and not the other. This pairs the raw
draws; it does not make outcomes identical, because whether a draw triggers an event still depends on state.

**Invariant: a feature that is off changes nothing.** With `reproduction=False` the four data files
(`outcomes.csv`, `events.csv`, `deaths.csv`, `snapshots.csv`) are byte-identical to `main` for the same seed.
`scripts/check_off_mode.sh [REF]` checks it (rule and utility policies, plus a 50-year run, because founders are at
most 40 and only a long run reaches retirement). Run it before merging anything that touches the simulation.

## How the model works (single life)

Numbers below are from the current code: 3 seeds x 60 agents x 50 years, rule policy, one layer switched off vs
the default world (`python -m lifesim.ablate --base X=False --set X=True`). The "on" arm is the same in all three:
alive 47.8%, restricted mean years 43.51, median final cash 147k nominal (63k in day-0 money).

| layer off -> on | alive | restricted mean years | median wealth/year, nominal | median wealth/year, real |
|---|---|---|---|---|
| `life_stages` | 50.6% -> 47.8% | 43.87 -> 43.51 | 9.2k -> 3.6k | 3.8k -> 1.6k |
| `shocks` | 50.6% -> 47.8% | 44.29 -> 43.51 | 5.5k -> 3.6k | 2.8k -> 1.6k |
| `economy` | 50.6% -> 47.8% | 44.02 -> 43.51 | 12.3k -> 3.6k | 12.3k -> 1.6k |

`restricted_mean_years` estimates E[min(T, horizon)], the mean years lived inside the simulated window; it is not
life expectancy. Differences of a few tenths of a year were not tested for significance (180 agents per arm).
Without an economy there is no inflation, so nominal and real coincide in the last row.

1. **Life stages** (`life_stages`): mandatory retirement at 65, pension from lifetime earnings, slower energy
   recovery when old. The clear effect is on wealth; survival barely moves because death is mostly the age-based
   mortality curve.
2. **Random shocks** (`shocks`): illness (likelier when old or unhealthy; paying for treatment cuts the damage to
   30%), chronic conditions, job loss with benefits, windfalls. Sick, laid-off or retired agents cannot work.
3. **Jobs and money** (`economy`): job tiers gated by education and career (an employed agent is promoted the
   moment they qualify), progressive tax, prices and pay that grow, lifestyle creep, interest on savings and
   debt, a credit limit past which you cannot buy food, and a welfare floor. Rule agents end rich (median 100k+ at
   30 years). Saving does not visibly protect lifespan: with a wide `cash_buffer` range (50 to 30000) about 88% of
   illnesses get treated in every quartile, because money is rarely the binding constraint. The money, treatment,
   health, lifespan link exists in the code (unit-tested) but is not exercised.
8. **Utility policy** (experimental, `--policy utility`): one-step lookahead. The agent tries each action on a
   scratch copy of itself and picks the best resulting state under a utility with needs, health, mood, concave
   cash, expected future pay, an emergency-fund target and a wall at the credit limit (genes `patience`,
   `wealth_weight`). **It loses to the rule policy** (2 seeds x 40 agents x 30 years): alive 45% vs 88%,
   restricted mean years 16.6 vs 28.2, median final cash -145 vs 101k. A single-seed diagnosis found only 11% of
   planner illnesses treated vs 65% for rule agents; the planner hovers near zero cash. Its weights are hand-tuned
   guesses. Options: retire it as a documented negative result, or let the learner replace it.
4. to 7. Body (fitness, diet), time and place, social, traits and background: not built.

## Generations (`reproduction`, off by default)

Turn on with `--set reproduction=True` and run long enough to see several generations (about 30 years each):
`--years 150`.

**Couples.** Every 30 days unpaired adults aged 20 to 40 can meet an opposite-sex partner within 8 years of their
age. Two people never pair if one is an ancestor of the other or they share an ancestor within `kinship_depth`
generations (default 2: parent-child, siblings, half-siblings, aunt/uncle-niece, grandparent-grandchild and first
cousins are blocked; second cousins may pair; founders count as unrelated). `assortative_mating` (default 0 =
random) prefers similar education (a stable softmax, so a strong value cannot underflow into "highest id wins").
Partners pay a discounted living cost (`household_cost_factor`); cash is not pooled. No divorce; a widow can
re-pair. Matching has no systematic priority by list position or id (women who attempt are processed in a keyed
random order, candidates by id, the pick is keyed), but it is a function of (seed, day, agent ids), so relabelling
agents can change who pairs with whom.

**Births.** An eligible couple (both healthy, mother 20 to 40, spacing, enough cash, together at least
`min_partnership_days` = 270) has a child at `birth_rate` per year, slowed as the population nears
`max_population`, and births never exceed the free slots (a hard cap; competition for the last slots is decided by
a keyed draw). Pregnancy is not simulated: births run before matching and need the 270-day partnership, so a couple
(or a re-paired widow) cannot produce a child the day they form, and a child is not born if the father has died by
then. `birth_reserve_gate=False` removes only the direct cash requirement; it is **not** a wealth-neutral control
(debt still damages health, which gates births, and parental spending still buys schooling), which is why the
neutral marker gene exists.

**Children are queued, not simulated.** Parents pay `child_cost` per day plus a capped share of their household
wealth (the sum of the living parents' positive cash; each child gets it independently of siblings) until 18. At 18
the child appears as an adult: each gene comes from a random parent plus Gaussian noise (`mutation_sigma`, clipped
to the gene's range), schooling rises with total parental spending (saturating), any inheritance held in trust is
paid out, and the child gets the same starter money founders got, in day-0 money.

**Estate.** At death, half of positive cash goes to the children (adults and minors in trust), half to the
surviving partner; with no partner all of it goes to the children; with no children it goes to the partner, else it
is lost. Debt is forgiven. Estates are settled after every agent has lived the day, in id order, and only people
alive at the end of the day inherit, so the result does not depend on the order of the agent list. A minor's trust
is nominal cash (no money is created); inflation erodes it until adulthood and the adult's baseline is its real
value at entry. A dead agent's `cash` stays at its value at death, so do not sum cash over dead agents.

**Real money (`real_money`, automatic with reproduction).** Prices inflate (2% a year by default: index about 7 at
year 100, 19.5 at year 150), so a fixed nominal amount would lose its meaning over generations, and because
`cash_buffer` is a heritable gene that would look like evolution. In real-money mode the `cash_buffer` gene is read
in day-0 money, starter money and the mean windfall scale with the price index, the pension is based on real career
earnings and indexed to prices through retirement, and outcomes report real figures. Set `real_money=False` for the
old nominal behaviour (what every run without reproduction does). Keep `wage_growth` equal to `inflation` for a
generational run (otherwise real wages grow), or set both to 0.

**Outputs (reproduction on).** `pair`/`birth`/`adult`/`estate` events; `marker`, `generation`, `n_children`,
`mother_id`, `father_id`, `inherited` in `outcomes.csv`; `lineage.csv` (adults and children not yet adults, with a
`status` column) and `population.csv` (alive adults, pending children, total). With real money on, `outcomes.csv`
also has `inherited_real`, `cash_real`, `price_idx_end`, and `wealth_per_year` becomes real accumulation excluding
transfers, `(cash_real - start_cash - inherited_real) / years`, NaN until 1 year of adult exposure. Children who
have not come of age are people: they hold population slots, so they are in the lineage and the run summary
reports adults, children and total separately.

**Population (rule agents, 100 founders, 150 years, seed 1):** birth rate 0.3 holds near 70 to 75 alive; 0.5 (the
default) grows to about 133 by year 100 and 168 by year 150; 1.0 peaks near 224 at year 100 then falls to about 173
as the founder cohort ages out (a cohort echo). Generation 6 is reached.

### `lifesim.generations`: selection or drift?

```
python -m lifesim.generations --seeds 8 --years 150 --out runs/gen
python -m lifesim.generations --arm no_gate birth_reserve_gate=False --arm assort assortative_mating=3
```

Runs births-on worlds over several seeds per experiment arm, in parallel (default arms: `baseline`,
`no_reserve_gate`, `assortative`; add yours with `--arm NAME KEY=VALUE ...`; an arm that sets `reproduction=False`
is refused up front). Writes `summary.csv` (arm, metric, value, standard error, runs) and the per-seed tables
`genes_by_generation.csv`, `gini.csv`, `parent_child.csv`, `population.csv`. For CSC use `slurm/generations.sbatch`.

- **Drift null.** A neutral `marker` gene exists only when reproduction is on (own random stream; its crossover
  and mutation draws are keyed by a fixed per-gene index, so no other draw and no off-mode output changes). It is
  inherited and mutated exactly like the strategy genes but read by no policy, so its movement is what chance
  alone does. Under the rule policy `patience` and `wealth_weight` are neutral too. A gene the policy reads
  (`study_frac`, `cash_buffer`) is compared with the neutral genes **of the same runs, paired by run**, with a
  paired t-test over runs: `t[gene]` is reported next to `t_crit_95[gene]` (Student's t for the number of runs,
  for example 2.37 at 8 runs, not 1.96); the gene moved more than drift explains only if |t| exceeds it. Change is
  measured from the founders to the **last complete generation** (the highest with at least 10 adults and no
  children still waiting to come of age, otherwise it would be only its earliest-maturing members), in units of
  the gene's allowed range. The summary says which generation was used; with a short horizon there may be none.
- **Other measures.** Rank correlation between a child's real net worth at 40 and their parents' (mean of the
  two; the snapshot must be within half the snapshot interval plus 0.1 years of age 40), with the share of pairs
  where both parents' wealth is known; Gini of real net worth among living adults (debts as zero) and the share
  in debt (last 50 years); last generation reached; final population including children.
- **Limits, stated plainly.** With a handful of seeds the standard errors are large and nothing is corrected for
  multiple comparisons. Wealth at 40 exists only for people who lived to 40 and were observed then (survivorship;
  founders already past about 40 at the first snapshot, roughly 1 to 2% with the default start ages, have none).
  The neutral genes are a valid null only because the model has no linkage between genes. "Generation" is a label
  (founders 0, a child is one more than the older parent), not a birth cohort. The rule policy reads only two of
  the four strategy genes, so the selection question is about those two.

**Fixed-population tools refuse what they cannot handle** (`guards.py`). The ES trainer, the evaluator and
`lifesim.ablate` refuse `reproduction=True`: `years_lived` over the whole horizon means different things per
generation, the sample is the number of adults ever created, and dead agents keep cash already passed on in an
estate, so summing cash double counts. The ES trainer and evaluator also refuse `real_money=True`: the policy
observes nominal cash with no price index and the reward uses a fixed nominal scale. (Default 2% inflation already
makes nominal cash drift over a 20-year training horizon; that is the existing RL setup.)

## Learning a policy (RL)

`lifesim/rl/` trains one shared network (numpy MLP, no torch) with evolution strategies. Every candidate network is
the brain of a whole world of agents and is scored by `rl/reward.py` (years lived, final wealth, health; the reward
is the assumption that decides the result, change it with `--reward w_wealth=1.0 ...`).
1. **Behaviour cloning** (`rl/clone.py`): copy the rule policy into the network so search starts from agents that
   already survive.
2. **ES** (`rl/es.py`): antithetic sampling, rank-shaped fitness, Adam, weight decay. All candidates see the same
   world seeds in a generation. Worlds are independent, so it scales over cores (`--workers`). Checkpoints every
   generation (including the RNG state, so an interrupted and resumed run matches an uninterrupted one),
   resumable with `--resume`; `log.csv` has the curve. `best.npz` holds exactly the weights that earned its
   recorded fitness and is never overwritten by a worse generation, also across resumes.
3. **Held-out evaluation** (`rl/evaluate.py`): network vs rule vs random on seeds never used in training.

```
python -m lifesim.rl.es --out runs/rl --generations 300 --pairs 32 --agents 20 --years 20
python -m lifesim.rl.evaluate --theta runs/rl/best.npz --seeds 20 --agents 40 --years 30
python -m lifesim.run --policy mlp --theta runs/rl/best.npz --years 50   # watch it live a life
```

Cost (measured on 4 cores of a sandbox): one world of 20 agents x 20 years takes about 8 s, 40 agents x 40 years
about 31 s. A generation with 32 pairs x 2 seeds is about 130 worlds: roughly 18 CPU-minutes at the small size, so
about 20 s per generation on 64 cores. The learned brain is shared, so agents differ only by state and luck, not by
genes (nothing is inherited by it). For CSC use `slurm/train_es.sbatch`.

## Known limits and assumptions

- Every default in `config.py` (economy tiers, tax, welfare, inflation, birth and meeting rates, child cost, estate
  split, mutation size, kinship depth) is an assumption set by eye. Results are conditional on them.
- Sex mortality multipliers are neutral (1.0) on purpose; sex only matters for who can bear children.
- No divorce, no pooled cash, no explicit childhood, no pregnancy state.
- Strategy genes barely affect survival (correlation about 0.01); they act through wealth and, with births on,
  through who can afford children.
- The ES trainer is unproven at scale and the utility planner is a negative result (see above).

## Roadmap

Next, in rough order of value:
1. Run `slurm/generations.sbatch` on CSC and read `summary.csv` (is `study_frac` or `cash_buffer` moving beyond
   what the neutral genes do?).
2. Run the ES pilot (issue #2) and compare the learned network with the rule policy on held-out seeds.
3. Decide the utility planner's fate (retire it, or fix it).
4. Optional hardening: range validation for config knobs (`meet_interval`, `min_partnership_days`,
   `estate_to_children_with_spouse`, population limits).
5. Layers 4 to 7 (body, place, social, traits), a dynamic-population reward so RL can run with births, 2D
   visualisation.
