#!/bin/bash -l
# Generation-aware experiment: couples, births and inheritance over many seeds, with a drift null.
# Runs lifesim.generations on one CPU node; every (arm, seed) is an independent world, so it scales over cores.
#
# Resource values below are my sizing, not verified Roihu limits. Account is passed on the command line.
#
# Submit:   sbatch -A project_XXXXXXX slurm/generations.sbatch          (16 seeds per arm)
# Bigger:   sbatch -A project_XXXXXXX slurm/generations.sbatch 24       (first argument = seeds per arm)
# New seeds: sbatch -A project_XXXXXXX slurm/generations.sbatch 16 16    (second argument = first seed, here 16..31)
# Smoke:    sbatch -A project_XXXXXXX -p test --cpus-per-task=8 --time=00:15:00 slurm/generations.sbatch 2
#
# Clean environment on purpose: sbatch normally copies the login shell's environment, and on Roihu that is the
# aarch64 module tree when you submit from the GPU login node, so python-data fails on an x86 node. With
# --export=NONE the node sets up its own modules, but that drops HOME, USER and the module setup: restored below.
# Because of that, env vars on the sbatch line (SEEDS=24) do not reach the job: options are positional arguments.
#
# Rough cost, measured on a 3-core sandbox: one world of 100 founders over 150 years takes several minutes
# (the run is dominated by Python per-agent work), so arms x seeds worlds / cores x that time. 3 arms x 24 seeds
# on 64 cores is on the order of minutes, not hours. Check the first job's wall time before scaling further.
#
#SBATCH --job-name=lifesim-generations
# no --account line on purpose: pass it at submit time, sbatch -A project_XXXXXXX (sbatch refuses placeholder values)
#SBATCH --partition=small                    # CPU partition seen in sinfo on Roihu; check its limits with scontrol
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48                   # one worker process per core; the default job is 3 arms x 16 seeds = 48 worlds
#SBATCH --mem=48G                            # about 100 MB measured per world; 1 GB per core is generous
#SBATCH --time=01:00:00                      # expected a few minutes (assumption from other hardware, not measured on Roihu-CPU)
#SBATCH --output=slurm-%j.out
#SBATCH --export=NONE

set -euo pipefail

# The venv's python points at the module's interpreter, so load the SAME module that built it, before activating.
# python-data gave Python 3.12 on both aarch64 (GPU) and x86_64 (CPU) nodes in interactive tests; no sbatch run yet.
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1    # parallelise over processes, not BLAS threads

OUT="runs/generations_${SLURM_JOB_ID}"
cd "${SLURM_SUBMIT_DIR}"

python -m lifesim.generations \
    --seeds "${1:-16}" --seed0 "${2:-0}" \
    --agents 100 --years 150 --policy rule \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT}"

echo "done: ${OUT}/summary.csv has one row per arm and metric (value, standard error, runs)"
