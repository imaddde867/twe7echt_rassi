# Working in lifesim

Read `README.md` first ("Where things stand" and "Roadmap"). This file is the working agreements.

## Commands
- `pip install -r requirements-dev.txt`
- `ruff check . && python -m pytest` (CI runs exactly this; ruff rules are pinned in `pyproject.toml`)
- `scripts/check_off_mode.sh [REF]` before merging anything that touches the simulation: outputs with
  `reproduction=False` must stay byte-identical to `main`
- `python -m lifesim.run`, `lifesim.ablate`, `lifesim.generations`, `lifesim.rl.es` (see the README)

## Invariants (each has tests)
- A feature that is off changes nothing: no extra columns, draws, or output files. New columns, files and random
  draws go behind the feature's switch, and new random draws are keyed (`rng.py`), never taken from a shared stream.
- Environmental randomness is a pure function of (seed, day, agent id, kind). Policies never touch it.
- Money that spans a long run is in day-0 money (real-money mode). Do not compare nominal cash across decades.
- Tools that assume a fixed population or nominal money refuse otherwise (`guards.py`). Do not remove a guard to make
  a command work; fix the tool or add a generation-aware one.
- Nothing depends on the order of `world.agents`. If you write a loop that decides who gets a scarce thing, order by
  a keyed draw, and add a test that reverses the agent list.

## How changes get made here
- Reproduce a reported problem before fixing it (a script that shows the wrong number). Some reports were wrong; say so.
- Every fix gets a regression test, and the test is checked by putting the bug back in a scratch copy: the test must
  fail. Equivalent mutants (no observable difference) are reported as such, not hidden.
- Claims in the README, docstrings and PR text must match the code. Re-measure a number after the code it depends on
  changed, and say when something was not re-measured.
- Defaults in `config.py` are assumptions set by eye. Say so; do not present them as findings. A result from one
  seed is a hint.
- Prefer a numbered list of what was and was not verified over adjectives.

## Pitfalls that actually happened
- A scripted README or file edit that fails halfway must stop the commit. Check that the diff is non-empty and
  contains the change before committing (`git show --stat`). Do not rely on `set -e` in a chained shell command.
- `pkill -f <pattern>` can kill the shell that contains the pattern. Kill by exact PID.
- A byte-identity check on a short run cannot see retirement or pension code (founders are at most 40); keep the
  50-year run in `scripts/check_off_mode.sh`.
- Do not report an experiment's numbers if the analysis code changed after it started; rerun or drop them.

## Open work
See the README roadmap and the open GitHub issues (the Roihu ES pilot, the generations experiment, config validation).
The Slurm scripts contain placeholders for partition, module and account that have not been verified for CSC Roihu.
