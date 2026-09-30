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

Each run writes `snapshots.csv` (monthly agent stats), `deaths.csv` and
`summary.png` into the output folder.

## Layout
| file | job |
|---|---|
| `config.py` | every tunable number |
| `agent.py` | agent state + `observe()` vector |
| `actions.py` | what each action does to the stats (the sim's assumptions) |
| `policy.py` | `act(obs) -> Action`. Random and rule-based now, RL later |
| `world.py` | daily loop, aging, mortality, logging |

## Known limits of v1
- One shared policy, so every agent behaves the same. Give each agent its own
  parameters (e.g. `study_frac`) to get inequality and something to study.
- Cash has no sink, so rule agents pile up millions. Needs prices/assets.
- No births yet, so no generations. Population only shrinks.
- Sex mortality multipliers are neutral (1.0) on purpose.

## Roadmap
1. Per-agent heterogeneity (strategy genes) + compare strategies.
2. Births and inheritance (what passes to the child?).
3. Swap `RulePolicy` for a shared RL policy; judge it against the baselines.
4. 2D visualization.
