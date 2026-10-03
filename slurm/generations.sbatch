#!/bin/bash
# Generation-aware experiment: couples, births and inheritance over many seeds, with a drift null.
# Runs lifesim.generations on one CPU node; every (arm, seed) is an independent world, so it scales over cores.
#
# Resource values below are my sizing, not verified Roihu limits. Account is passed on the command line.
#
# Submit:   sbatch -A project_XXXXXXX slurm/generations.sbatch
# Bigger:   sbatch -A project_XXXXXXX --export=ALL,SEEDS=24 slurm/generations.sbatch
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

set -euo pipefail

# The venv's python points at the module's interpreter, so load the SAME module that built it, before activating.
# python-data gave Python >= 3.10 on a Roihu-GPU node (aarch64); not checked on Roihu-CPU nodes.
module load python-data
source "${LIFESIM_VENV:-$HOME/venvs/lifesim-$(uname -m)}/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1    # parallelise over processes, not BLAS threads

OUT="${LIFESIM_OUT:-runs/generations_${SLURM_JOB_ID}}"
cd "${SLURM_SUBMIT_DIR}"

srun python -m lifesim.generations \
    --seeds "${SEEDS:-16}" \
    --agents 100 --years 150 --policy rule \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT}"

echo "done: ${OUT}/summary.csv has one row per arm and metric (value, standard error, runs)"
