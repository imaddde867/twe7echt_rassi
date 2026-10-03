# ES replication with training seed 1 (jobs 2004980 and 2005005)

Provenance. Same setup as the pilot (`docs/results/es_pilot_2004461.md`), changing only the training seed:
`sbatch -A project_2020845 slurm/train_es.sbatch 20 --seed 1` (job 2004980) and `... 1 --seed 1` (job 2005005, the
clone-only control: one generation, so its `best.npz` is the behaviour-cloned network before any ES update). Roihu
`small` partition, 32 workers, `python-data` 3.12, repository at commit dbb979e (the first runs with the new
`--seed` argument). Reward is the default (`w_survival=1.0, w_wealth=0.5, w_health=0.25, cash_scale=20000`), as the
`reward=` line in both outputs shows. Held-out evaluation: the same 20 seeds (9000000 to 9000019), 40 agents, 30
years; the `rule` and `random` rows equal the pilot's.

- **2004980**: COMPLETED, 15 min 6 s elapsed (about 8.0 CPU-hours if all 32 cores were allocated; the allocation
  was not printed). Evaluated network: its `best.npz`, selected at generation 11, so the weights after 11 updates
  (see "Which network" in the pilot doc). `latest.npz` was not evaluated.
- **2005005**: COMPLETED, 3 min 17 s elapsed (about 1.8 CPU-hours on the same assumption). Its generation-0 fitness
  (1.445) equals generation 0 of 2004980 (1.445): the clone for a given seed is reproducible.

Evidence (`docs/results/es_replication_2004980/`): `eval_2004980_mlp.csv` and `eval_2005005_mlp.csv` (the `mlp` rows
of each job's `eval.csv`, **hand-copied from the terminal**: the last digits of a float may differ from the Roihu
originals; every mean and standard deviation in the printed table is reproduced from them, 8 columns each),
`slurm-2004980.out`, `slurm-2005005.out` (as printed, module banner removed), `paired_test.py`. The `rule` rows
and the seed-0 rows come from the pilot's folder.

## Held-out evaluation (mean over 20 seeds, standard deviation across seeds in parentheses)

| network | reward | survival | wealth | health | alive | median cash | illness treated |
|---|---|---|---|---|---|---|---|
| rule | 1.515 (0.058) | 0.970 | 0.323 | 0.223 | 0.896 | 111 283 | 0.633 |
| seed 0, clone only (2004551) | 1.613 (0.039) | 0.969 | 0.423 | 0.221 | 0.885 | 112 969 | 0.845 |
| seed 0, ES best, 11 updates (2004461) | 1.689 (0.038) | 0.970 | 0.493 | 0.226 | 0.904 | 269 866 | 0.891 |
| seed 1, clone only (2005005) | 1.410 (0.050) | 0.965 | 0.239 | 0.206 | 0.829 | 21 010 | 0.485 |
| seed 1, ES best, 11 updates (2004980) | 1.683 (0.041) | 0.969 | 0.488 | 0.225 | 0.901 | 213 815 | 0.974 |

Survival, wealth and health are the weighted reward terms (`lifesim.rl.evaluate`, PR #10). Cash is nominal.

## Paired by held-out seed (`paired_test.py`)

| comparison | mean difference | se | t (df 19) | first higher on |
|---|---|---|---|---|
| seed-1 ES minus seed-1 clone | +0.2731 | 0.0074 | 37.1 | 20 of 20 |
| seed-1 clone minus rule | -0.1055 | 0.0118 | -8.9 | 0 of 20 |
| seed-1 ES minus rule | +0.1677 | 0.0098 | 17.1 | 20 of 20 |
| seed-1 ES minus seed-0 ES | -0.0060 | 0.0023 | -2.6 | 3 of 20 |

Critical value 2.093 for one test; 2.759 with a Bonferroni correction for these four (scipy `t.ppf`). The first three
pass it by a wide margin. The last, -2.6, passes the single-test value but not the corrected one: the two ES
networks are close (0.006 apart), and the data here do not settle whether that small gap is real.

## What this shows

1. **The ES result replicates over two training seeds.** From a different clone, 20 generations of ES reached a best
   checkpoint at 1.683, against 1.689 for seed 0, and 0.168 above `rule` on all 20 paired worlds. The reward terms
   are almost the same (survival 0.969 / 0.970, wealth 0.488 / 0.493, health 0.225 / 0.226).
2. **"The cloned network alone beats `rule`" does not hold in general.** The pilot's seed-0 clone did (1.613 vs
   1.515); the seed-1 clone is worse than `rule` on all 20 worlds (1.410). How good the clone is depends on the
   seed. It is the wealth term that differs: the seed-1 clone has a mean median-cash of 21k, with a median cash of
   only 110 to 150 in 8 of the 20 worlds, and treats fewer illnesses than `rule` (0.485 vs 0.633).
3. **ES does more where the start is worse:** +0.273 from the seed-1 clone, against +0.076 from the seed-0 clone.
   Both runs end near the same reward.
4. **Different networks, same reward.** The two ES networks earn about the same reward with different behaviour:
   median cash 270k vs 214k, illness treated 0.891 vs 0.974. The cash term saturates (ceiling 0.5), so cash beyond
   about 60k per agent does not show up in the reward.
5. Both training curves dip at generation 7 (fitness 1.538 and 1.523, years 17.9). `gen_seeds` depends only on the
   generation, so both runs score generation 7 on the same two worlds: the dip belongs to those worlds, not to
   training being unstable.

## What this does not support

- Two training seeds, one reward weighting, 20 of 300 generations. The reward is an assumption (`w_wealth`,
  `cash_scale`); a different weighting could change what ES learns, and that has not been tried.
- Why the clone depends so much on the seed, or why ES ends near 1.68 to 1.69 from both starts (is that a ceiling of
  this reward, or of the 20 generations?). The per-agent cash distribution was not looked at.
- Whether the 0.006 gap between the two ES networks is real (see the last row above).
- Evaluated networks are 11 updates into a 20-generation run; `latest.npz` of this run was not scored (it matched
  `best.npz` for seed 0).
- Same model, defaults and reward as training; nothing here says the policy does well in a different world or for
  real people. The numbers were transcribed by hand from the terminal.
