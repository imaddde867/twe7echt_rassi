#!/bin/bash -l
# Train the shared life policy with evolution strategies on one CPU node.
#
# Resource values below are my sizing, not verified Roihu limits. Account is passed on the command line.
#
# Submit:   sbatch -A project_XXXXXXX slurm/train_es.sbatch
# Pilot:    sbatch -A project_XXXXXXX slurm/train_es.sbatch 20                (first argument = generations, default 300)
# Resume:   sbatch -A project_XXXXXXX slurm/train_es.sbatch 300 runs/rl_JOBID (second argument = run directory to resume)
# Seed:     sbatch -A project_XXXXXXX slurm/train_es.sbatch 20 --seed 1      (training seed; held-out seeds stay fixed)
# Reward:   sbatch -A project_XXXXXXX slurm/train_es.sbatch 20 --reward w_wealth=0.25 cash_scale=5000 (trainer and evaluation)
#
# Clean environment on purpose: sbatch normally copies the login shell's environment, and on Roihu that is the
# aarch64 module tree when you submit from the GPU login node, so python-data fails on an x86 node. With
# --export=NONE the node sets up its own modules, but that drops HOME, USER and the module setup: restored below.
# Because of that, env vars on the sbatch line (SEEDS=24) do not reach the job: options are positional arguments.
#
#SBATCH --job-name=lifesim-es
# no --account line on purpose: pass it at submit time, sbatch -A project_XXXXXXX (sbatch refuses placeholder values)
#SBATCH --partition=small                    # CPU partition seen in sinfo on Roihu; check its limits with scontrol
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                   # training uses exactly this many worker processes
#SBATCH --mem=32G                            # each worker is small; 1 GB per core is plenty
#SBATCH --time=02:00:00                      # sized for a pilot; the full 300 generations is about 90 CPU-hours (estimate)
#SBATCH --output=slurm-%j.out
#SBATCH --export=NONE

set -euo pipefail

# --- software: use whatever the docs say gives you Python >= 3.10 -------------
# The system python is 3.9 and lifesim needs >= 3.10. python-data gave 3.12 on both aarch64 (GPU) and x86_64 (CPU)
# nodes in interactive tests; no sbatch run yet. Build the venv once with the same module (one venv per architecture):
#   module load python-data && python3 --version     # must be >= 3.10
#   python3 -m venv ~/venvs/lifesim-$(uname -m) && source ~/venvs/lifesim-$(uname -m)/bin/activate
#   pip install -r requirements-dev.txt
# CSC's /etc/profile.d/zz-csc-env.sh (which defines `module`) returns early in a non-interactive shell unless
# CSC_ENV_INIT_NON_INTERACTIVE=yes; it also reads $HOME and $USER and uses unset variables, so no `set -u` around it.
export HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
export USER="$(id -un)"
export CSC_ENV_INIT_NON_INTERACTIVE=yes
set +u
source /etc/profile.d/zz-csc-env.sh
module load python-data
source "${LIFESIM_VENV:-$HOME/venvs/lifesim-$(uname -m)}/bin/activate"
set -u
# No `srun` before python: one task on one node needs none, and srun execve()s `python` directly, which failed
# here ("execve(): python: No such file or directory"), probably because the module sets `python` up in the shell.

# numpy must not spawn its own threads inside each worker: we parallelise over processes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Write results somewhere with space/quota, not $HOME (see the docs' storage page).
OUT="runs/rl_${SLURM_JOB_ID}"
# --- args ---
# GENERATIONS [RESUME_DIR] [--seed N] [--reward KEY=VALUE ...]; --seed goes to the trainer only (the held-out
# evaluation seeds stay fixed so networks stay comparable), --reward goes to both the trainer and the evaluation.
GENERATIONS="${1:-300}"; [ $# -gt 0 ] && shift
RESUME=""; SEED_ARGS=(); REWARD_ARGS=()
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then RESUME="$1"; shift; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --seed) [ $# -ge 2 ] || { echo "train_es.sbatch: --seed needs a value" >&2; exit 2; }
                SEED_ARGS=(--seed "$2"); shift 2 ;;
        --reward) REWARD_ARGS=(--reward); shift
                  while [ $# -gt 0 ] && [[ "$1" != --* ]]; do REWARD_ARGS+=("$1"); shift; done ;;
        *) echo "train_es.sbatch: unknown argument '$1' (usage: GENERATIONS [RESUME_DIR] [--seed N] [--reward K=V ...])" >&2
           exit 2 ;;
    esac
done
# --- end args ---
RESUME_FLAG=""; [ -n "${RESUME}" ] && { RESUME_FLAG="--resume"; OUT="${RESUME}"; }

cd "${SLURM_SUBMIT_DIR}"
python -m lifesim.rl.es \
    --out "${OUT}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --generations "${GENERATIONS}" --pairs 32 --seeds-per-eval 2 \
    --agents 20 --years 20 \
    ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} ${REWARD_ARGS[@]+"${REWARD_ARGS[@]}"} \
    ${RESUME_FLAG}

# Held-out comparison against the rule and random policies (writes a CSV too)
python -m lifesim.rl.evaluate \
    --theta "${OUT}/best.npz" --seeds 20 --agents 40 --years 30 \
    --workers "${SLURM_CPUS_PER_TASK}" ${REWARD_ARGS[@]+"${REWARD_ARGS[@]}"} \
    --out "${OUT}/eval.csv" | tee "${OUT}/eval.txt"
