# lifesim

A small agent-based life simulation. Agents (sex, age, health, energy, satiety,
mood, cash, education, career) pick 3 actions per day (work, study, eat, rest,
socialize) and live until health or old age kills them. Built for fun first;
research questions later.

```
pip install -r requirements.txt
python -m lifesim.run --policy rule   --years 50 --out runs/rule
python -m lifesim.run --policy random --years 50 --out runs/random
pytest
```

Each run writes `snapshots.csv` (monthly agent stats), `deaths.csv`, `events.csv` (shocks), `outcomes.csv` (genes + result per agent) and
`summary.png` into the output folder.

## Changing the world
Every number lives in `config.py`. Override from the CLI without editing code:
`python -m lifesim.run --set retire_age=70 pension_frac=0.3`.
Measure what a feature does by comparing configs over several seeds:
`python -m lifesim.ablate --base life_stages=False --set life_stages=True`.

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

## Known limits of v1
- Strategy genes (`study_frac`, `cash_buffer`) are random at birth and fixed for
  life. Survival barely depends on them (corr ~0.03), only wealth does, so
  there is no selection pressure until births + inheritance exist.
- Rule agents do not plan: they stop working once cash passes a small buffer, so
  they under-save and most end near the credit limit. Needs layer 8.
- The economy numbers (tiers, tax, welfare, inflation) were tuned by eye until
  the wealth distribution stopped being bimodal. Treat them as assumptions.
- Median age at death is biased by the fixed window; compare `mean_years_lived`.
- No births yet, so no generations. Population only shrinks.
- Sex mortality multipliers are neutral (1.0) on purpose.

## Layers (single-life realism)
1. **Life stages** (done, `life_stages`): mandatory retirement at 65, pension
   from lifetime earnings, slower energy recovery when old. Ablation (3 seeds x
   60 agents x 50y): median wealth/year 16.8k -> 12.6k, survival unchanged.
2. **Random shocks** (done, `shocks`): illness (likelier when old or unhealthy;
   paying for treatment cuts the damage to 30%), chronic conditions, job loss
   with benefits, windfalls. Sick, laid-off or retired agents cannot work.
   Events go to `events.csv`. Ablation (3 seeds x 60 agents x 50y): median
   wealth/year 12.6k -> 10.9k, mean years lived 44.4 -> 44.1 (within noise).
   The intended link money -> treatment -> health -> lifespan is NOT visible
   yet: 85% of illnesses get treated and `cash_buffer` has no effect on
   lifespan (corr -0.03), because rule agents are rich within a few years. In a
   poor world (`base_wage=12 start_cash=0`) everyone dies within ~2 years. The
   economy is bimodal (rich or dead); layer 3 has to create the middle.
3. **Jobs and money** (done, `economy`): discrete job tiers gated by education
   and career, progressive tax, prices and pay that grow over time, lifestyle
   creep (spending rises with income), interest on savings and debt, a credit
   limit past which you cannot buy food, and a welfare floor that keeps people
   alive at the limit. Ablation (3 seeds x 60 agents x 50y): mean years lived
   44.1 -> 42.0, median final cash 484k -> about -2k. The bimodal world is now a
   spread: in a 30y run, cash p10/50/90 = -2k / 1.3k / 79k, 86% alive.
   Does saving protect lifespan? Barely. With a wide `cash_buffer` range
   (50 to 30000) the share of illnesses that get treated rises 73% -> 79% from
   the lowest to highest quartile, but the lifespan effect is within noise
   (corr 0.05). Most agents still end near the credit limit: rule agents have no
   plan for retirement (pension is 50% of nominal average earnings). That is a
   decision-making gap (layer 8), not something more economy rules will fix.
   Without `welfare_daily` the poor simply starve (44 of 49 deaths in a test).
4. Body (fitness, diet). 5. Time and place. 6. Social. 7. Traits/background.
8. Utility-based decisions, then RL.

## Roadmap
1. ~~Per-agent heterogeneity~~ done. `outcomes.csv` has one row per agent.
2. Births and inheritance (what passes to the child?).
3. Swap `RulePolicy` for a shared RL policy; judge it against the baselines.
4. 2D visualization.
