#!/bin/bash
# Generation-aware experiment: couples, births and inheritance over many seeds, with a drift null.
# Runs lifesim.generations on one CPU node; every (arm, seed) is an independent world, so it scales over cores.
#
# FILL IN the <...> values from the Roihu docs (https://docs.csc.fi/computing/running/creating-job-scripts-roihu/
# and .../batch-job-partitions/). Partition, module and account names are NOT verified for Roihu.
#
# Submit:   sbatch slurm/generations.sbatch
# Bigger:   sbatch --export=ALL,SEEDS=24 slurm/generations.sbatch
#
# Rough cost, measured on a 3-core sandbox: one world of 100 founders over 150 years takes several minutes
# (the run is dominated by Python per-agent work), so arms x seeds worlds / cores x that time. 3 arms x 24 seeds
# on 64 cores is on the order of minutes, not hours. Check the first job's wall time before scaling further.
#
#SBATCH --job-name=lifesim-generations
#SBATCH --account=<project_XXXXXXX>          # your CSC project
#SBATCH --partition=<CPU-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<CORES>              # one worker process per core
#SBATCH --mem=<MEM, e.g. 32G>                # a run keeps its snapshots in memory: about 1 GB per core is plenty
#SBATCH --time=<HH:MM:SS>
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
