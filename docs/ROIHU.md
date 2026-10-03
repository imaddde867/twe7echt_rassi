# Running lifesim on CSC Roihu

A runbook from the first real session (2026-10-03). Everything marked **observed** was seen in command output on
Roihu; everything marked **not verified** is an assumption. CSC's own docs (https://docs.csc.fi) were not readable
when this was written, so check them for limits and billing.

## What was observed

- Two login/compute architectures share one Slurm: the GPU login node `roihu-gpu-login2` is `aarch64` (GH200); the
  CPU compute nodes (`rc....`) are `x86_64`. A virtualenv built on one does not run on the other.
- `sinfo` partitions seen: CPU `test`, `interactive`, `longrun`, `small` (default), `medium`, `large`, `hugemem`,
  `hugemem_longrun`; GPU `gputest`, `gpuinteractive`, `gpumedium`, `gpularge`; `vizinteractive`. CPU nodes show 384
  CPUs and about 760 GB; GPU nodes 288 CPUs, 868 GB and 4 GH200.
- `sinteractive -i` from the GPU login node picks `gpuinteractive` with 1 GPU (and 1/4 of the node's memory).
  lifesim uses no GPU, so do not use it for lifesim work. Billing rate for GPU vs CPU: **not verified**.
- The system Python is 3.9.25: too old (lifesim needs 3.10 or newer). `module load python` is a dummy that changes
  nothing. `module load python-data` gave Python 3.12.13 on both architectures; it is a container-backed module.
- Working stack on both architectures: numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2, pytest 9.1.1, ruff 0.16.10.
  155 tests pass (57 s on 16 aarch64 cores, 42 s on 8 x86_64 cores) and `scripts/check_off_mode.sh` reports all 12
  files byte-identical to `main` on both.
- `test` accepted `-c 8 --mem=16G -t 00:15:00`; `small` accepted 48 cores for the generations job (it ran).

## Measured cost (lifesim.generations, 100 founders, 150 years, `rule` policy)

| job | cores | wall | peak memory (whole job) |
|---|---|---|---|
| 6 worlds (2 seeds x 3 arms), job 2002410 | 8 | 2 min 24 s | 765 MB |
| 48 worlds (16 seeds x 3 arms), job 2002416 | 48 | 2 min 51 s | 2.9 GB |

So one world is about 2.4 min on one x86 core, and about 125 MB. Worlds are independent, so useful cores are at most
the number of worlds. 48 cores for 171 s is 2.3 core-hours. ES cost, measured on Roihu (jobs 2004461, 2004551, 32
cores): behaviour cloning 127 to 133 s, about 33 s per generation (about 18 CPU-minutes, as the sandbox estimate
said), and the 20-generation pilot took 13 min 39 s, 26 208 CPU-seconds (about 7.3 CPU-hours) including
evaluation. Not measured: a run beyond 20 generations (an extrapolation is about 90 CPU-hours for 300).

## One-time setup

Do this on a **compute node**, not the login node (login nodes are for light work). From the GPU login node:

```bash
srun -A project_XXXXXXX -p test -c 8 --mem=16G -t 00:15:00 --export=NONE --pty /bin/bash -l
```

Inside it (replace nothing in these lines except the project number above):

```bash
hostname; uname -m                         # a node named rc..., and x86_64
export HOME=$(getent passwd $(id -un) | cut -d: -f6)
module load python-data
python3 --version                          # must be 3.10 or newer
python3 -m venv ~/venvs/lifesim-$(uname -m)
source ~/venvs/lifesim-$(uname -m)/bin/activate
git clone https://github.com/imaddde867/twe7echt_rassi.git ~/lifesim   # skip if already cloned
cd ~/lifesim && pip install -r requirements-dev.txt
ruff check . && python -m pytest
scripts/check_off_mode.sh
exit
```

`--export=NONE` and the absolute `/bin/bash` both matter (see the table below). The venv name carries the
architecture because the GPU side needs its own (`lifesim-aarch64`).

## Running jobs

Submit from the login node. The scripts pass an account on the command line (sbatch rejects placeholder values):

```bash
cd ~/lifesim && git pull
sbatch -A project_XXXXXXX -p test --cpus-per-task=8 --time=00:15:00 slurm/generations.sbatch 2   # smoke test
sbatch -A project_XXXXXXX slurm/generations.sbatch                  # 3 arms x 16 seeds, 48 cores, 1 h limit
sbatch -A project_XXXXXXX slurm/generations.sbatch 16 16            # a second block: seeds 16..31
sbatch -A project_XXXXXXX slurm/train_es.sbatch 20                  # ES pilot, 20 generations (see below)
```

Positional arguments, not environment variables, because the scripts run with a clean environment. `generations.sbatch`
takes seeds per arm, then the first seed. `train_es.sbatch` takes the number of generations, then (to resume) the run
directory, then optionally `--seed N` (training seed, trainer only; the held-out evaluation seeds stay fixed) and
`--reward KEY=VALUE ...` (reward settings, passed to both the trainer and the evaluation); for example
`sbatch -A project_XXXXXXX slurm/train_es.sbatch 20 --seed 1`. The generations script was run on Roihu (jobs 2002410, 2002416), and so was the ES script (the 20-generation
pilot 2004461 and the one-generation control 2004551 ran to completion with the defaults, no errors; the resume
argument has not been used on Roihu). The defaults (32 cores, 2 h) suit the pilot; for the remaining 280 generations
of a 300-generation run (about 2.6 h on 32 cores, about 83 CPU-hours, an estimate) pass `--time=05:00:00`, or
`--cpus-per-task=64` with a shorter limit. Results of the pilot: `docs/results/es_pilot_2004461.md`.

Look at a job without a job number:

```bash
squeue --me                       # R running, PD pending; empty means finished OR failed, check the next line
sacct -S today --format=JobID,JobName,Elapsed,MaxRSS,AllocCPUS,State,ExitCode
tail -n 20 slurm-*.out            # the newest file is the last one listed
```

Success is `COMPLETED 0:0` and a last log line `done: runs/<name>_<jobid>/summary.csv ...`. Output lands in
`~/lifesim/runs/`, which counts against the home quota (limit **not verified**); `scp` results back or move them to
the project's scratch or work area for anything large.

## Errors that actually happened

| what you see | cause | fix |
|---|---|---|
| `FATAL: ... image's architecture (arm64) could not run on the host's (amd64)` on a compute node | the job inherited the aarch64 login node's module paths | start with `--export=NONE` (interactive) and keep `#SBATCH --export=NONE` (scripts) |
| `execve(): bash: No such file or directory` right after `--export=NONE` | PATH is empty in a clean environment | use the absolute `/bin/bash -l` |
| `mkdir: cannot create directory '/.local'` | `--export=NONE` drops HOME | `export HOME=$(getent passwd $(id -un) \| cut -d: -f6)` |
| `module: command not found`, exit 127, in a batch job | `/etc/profile.d/zz-csc-env.sh` returns early in non-interactive shells | `export CSC_ENV_INIT_NON_INTERACTIVE=yes` and `source` that file, with `set +u` around it (the file reads unset variables) |
| `execve(): python: No such file or directory`, exit 2, from `srun python` | `srun` execs `python` directly; the module sets it up in the shell | run `python` without `srun` (inferred cause; the fix worked) |
| `sbatch: error: Invalid numeric value "<CORES>"` | placeholders in `#SBATCH` lines are rejected even if overridden on the command line | real defaults in the script, account via `-A` |
| `TypeError: unsupported operand type(s) for \|` while importing lifesim | Python 3.9 | load `python-data`; `lifesim/__init__.py` now raises a readable error |
| shell says `jobid: No such file or directory`, or `HOME=...` after pasting | `<...>` or an explanatory snippet pasted as a command | only paste blocks meant as commands; fill in `<...>` first |

Exit codes: 127 is "command not found", 2 is a usage error or a missing file, 0:0 is success.

## Not verified

- Billing: units per core-hour, per GPU-hour, and whether Slurm bills elapsed or requested time (check
  `csc-workspaces` or My CSC). The first runs used the quota of a project that belongs to other work; the full ES run
  (about 90 CPU-hours) should be agreed with the project owner, or run under your own project.
- Partition limits (`scontrol show partition small`), the home-directory quota, whether `longrun` is better for
  multi-hour jobs.
- That `train_es.sbatch` resumes (`--resume`) correctly on Roihu, and that a run longer than the 20-generation pilot
  completes (see above).
