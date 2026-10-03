# Generations experiment, first run (Roihu job 2002416)

Provenance. Command: `sbatch -A <project> slurm/generations.sbatch` (defaults: 16 seeds per arm, seeds 0 to 15, 100
founders, 150 years, `rule` policy, 3 arms). 48 cores on the `small` partition, wall 2 min 51 s, peak memory 2.9 GB
for the whole job. Repository at commit 64f716e: the simulation and analysis code was the same as on `main`
(that commit differs only by a Python version check in `lifesim/__init__.py`, the Slurm scripts and docs).
Python 3.12.13, numpy 2.5.3, pandas 3.0.6, x86_64.

The numbers below were transcribed by hand from the `summary.csv` the job wrote, rounded to the digits shown. The
original is `~/lifesim/runs/generations_2002416/` on Roihu (it is git-ignored). Each cell is the mean over 16 runs with
the standard error across runs in parentheses. Arms: `baseline` (defaults), `no_reserve_gate`
(`birth_reserve_gate=False`), `assortative` (`assortative_mating=3.0`).

## Gene shifts (last complete generation used: 4 in all arms)

A shift is the change of the population-mean gene value from the founders to the last complete generation. "minus
neutral" subtracts the mean shift of the three neutral genes (`marker`, `patience`, `wealth_weight`, which the `rule`
policy never reads); `t` is that difference over its standard error. Critical value per test: 2.131 (df = 15, 95%).

| arm | gene | shift | minus neutral | t |
|---|---|---|---|---|
| baseline | cash_buffer | +0.0194 (0.0164) | +0.0095 (0.0214) | 0.44 |
| baseline | study_frac | -0.1285 (0.0144) | -0.1384 (0.0169) | -8.17 |
| no_reserve_gate | cash_buffer | +0.0172 (0.0144) | +0.0181 (0.0166) | 1.09 |
| no_reserve_gate | study_frac | -0.0263 (0.0116) | -0.0255 (0.0108) | -2.35 |
| assortative | cash_buffer | -0.0207 (0.0132) | -0.0339 (0.0145) | -2.33 |
| assortative | study_frac | -0.1383 (0.0172) | -0.1514 (0.0213) | -7.11 |

Neutral genes (the drift null):

| arm | marker | patience | wealth_weight |
|---|---|---|---|
| baseline | +0.0016 (0.0169) | +0.0070 (0.0219) | +0.0212 (0.0198) |
| no_reserve_gate | +0.0073 (0.0132) | +0.0106 (0.0131) | -0.0205 (0.0178) |
| assortative | +0.0149 (0.0122) | +0.0171 (0.0136) | +0.0075 (0.0147) |

## Population, wealth and inequality

| arm | parent-child spearman (wealth at 40) | pairs per run | pairs with both parents' wealth known | gini of real net worth (last 50 y) | adults in debt (last 50 y) | last generation reached | final population |
|---|---|---|---|---|---|---|---|
| baseline | 0.359 (0.018) | 269.8 | 0.800 (0.012) | 0.612 (0.006) | 0.0072 (0.0006) | 5.94 (0.06) | 211.9 (2.5) |
| no_reserve_gate | 0.437 (0.020) | 298.7 | 0.795 (0.010) | 0.641 (0.007) | 0.0147 (0.0014) | 5.94 (0.06) | 206.1 (4.0) |
| assortative | 0.346 (0.020) | 273.1 | 0.818 (0.010) | 0.610 (0.008) | 0.0076 (0.0006) | 5.88 (0.09) | 212.6 (3.4) |

## What this supports

1. `study_frac` falls across generations in the baseline and assortative arms, by about 0.14 (minus the neutral
   genes), beyond what the neutral genes do (|t| 7 to 8). It survives a Bonferroni correction
   for the six tests run (critical value about 3.04).
2. `cash_buffer` does not move beyond drift in the baseline (t 0.44).
3. The neutral genes moved by 0.002 to 0.021 with standard errors 0.012 to 0.022: the drift null behaves like noise
   around zero.
4. Turning the birth reserve gate off shrinks the `study_frac` fall from -0.138 to -0.025, and doubles the share of
   adults in debt (0.7% to 1.5%). So the gate is involved in the fall.

## What it does not support

- Two results are borderline and should not be read as effects: `study_frac` without the gate (t -2.35) and
  `cash_buffer` under assortative mating (t -2.33). Both fail the corrected 3.04 cutoff, and about 0.3 of six tests
  would pass 2.131 by chance.
- Why the gate matters is a hypothesis (for example, studying delays reaching the cash reserve a birth requires, so
  students have fewer children). It was not tested directly.
- Arm differences were not tested with a paired or unpaired difference test. The contrast in point 4 is by eye
  (about 5.5 standard errors, computed from the two arms' separate standard errors).
- One run of 16 seeds, one configuration. Every default in `config.py` (including the gate, wages and the study payoff)
  is an assumption set by eye. This describes what the model rewards, not what is true of people.
- Neither the 150-year horizon nor the 100-founder start was varied; the last complete generation is 4 of about 6.

## Next checks

A second seed block (`sbatch -A <project> slurm/generations.sbatch 16 16`, seeds 16 to 31) to see whether the
baseline `study_frac` result repeats; a direct measurement of when students reach the birth reserve; a paired
test of baseline against `no_reserve_gate` over the same seeds.
