#!/bin/bash -l
# Train the shared life policy with evolution strategies on one CPU node.
#
# Resource values below are my sizing, not verified Roihu limits. Account is passed on the command line.
#
# Submit:   sbatch -A project_XXXXXXX slurm/train_es.sbatch
# Resume:   sbatch -A project_XXXXXXX slurm/train_es.sbatch runs/rl_<jobid>   (first argument = run directory to resume)
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

# numpy must not spawn its own threads inside each worker: we parallelise over processes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Write results somewhere with space/quota, not $HOME (see the docs' storage page).
OUT="runs/rl_${SLURM_JOB_ID}"
RESUME_FLAG=""; [ -n "${1:-}" ] && { RESUME_FLAG="--resume"; OUT="$1"; }

cd "${SLURM_SUBMIT_DIR}"
srun python -m lifesim.rl.es \
    --out "${OUT}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --generations 300 --pairs 32 --seeds-per-eval 2 \
    --agents 20 --years 20 \
    ${RESUME_FLAG}

# Held-out comparison against the rule and random policies (writes a CSV too)
srun python -m lifesim.rl.evaluate \
    --theta "${OUT}/best.npz" --seeds 20 --agents 40 --years 30 \
    --workers "${SLURM_CPUS_PER_TASK}" --out "${OUT}/eval.csv" | tee "${OUT}/eval.txt"
