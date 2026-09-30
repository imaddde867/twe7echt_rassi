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

Each run writes `snapshots.csv` (monthly agent stats), `deaths.csv`, `outcomes.csv` (genes + result per agent) and
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

## Known limits of v1
- Strategy genes (`study_frac`, `cash_buffer`) are random at birth and fixed for
  life. Survival barely depends on them (corr ~0.03), only wealth does, so
  there is no selection pressure until births + inheritance exist.
- Cash has no sink, so rule agents pile up millions. Needs prices/assets.
- No births yet, so no generations. Population only shrinks.
- Sex mortality multipliers are neutral (1.0) on purpose.

## Layers (single-life realism)
1. **Life stages** (done, `life_stages`): mandatory retirement at 65, pension
   from lifetime earnings, slower energy recovery when old. Ablation (3 seeds x
   60 agents x 50y): median wealth/year 16.8k -> 12.6k, survival unchanged.
2. Random shocks (illness, job loss, windfall). 3. Jobs, prices, debt, savings.
4. Body (fitness, diet). 5. Time and place. 6. Social. 7. Traits/background.
8. Utility-based decisions, then RL.

## Roadmap
1. ~~Per-agent heterogeneity~~ done. `outcomes.csv` has one row per agent.
2. Births and inheritance (what passes to the child?).
3. Swap `RulePolicy` for a shared RL policy; judge it against the baselines.
4. 2D visualization.
