#!/bin/bash
# Train the shared life policy with evolution strategies on one CPU node.
#
# FILL IN the <...> values from the Roihu docs (https://docs.csc.fi/computing/running/creating-job-scripts-roihu/
# and .../batch-job-partitions/). I could not read those pages when writing this,
# so nothing below about partitions or modules is verified for Roihu.
#
# Submit:   sbatch slurm/train_es.sbatch
# Resume:   sbatch --export=ALL,RESUME=1 slurm/train_es.sbatch
#
#SBATCH --job-name=lifesim-es
#SBATCH --account=<project_XXXXXXX>          # your CSC project
#SBATCH --partition=<CPU-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<CORES>              # training uses exactly this many worker processes
#SBATCH --mem=<MEM, e.g. 32G>                # each worker is small; 1 GB per core is plenty
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=slurm-%j.out

set -euo pipefail

# --- software: use whatever the docs say gives you Python >= 3.10 -------------
# module load <python-module>
# Create the venv once (on a login node or in an interactive job), then reuse it:
#   python3 -m venv ~/venvs/lifesim && source ~/venvs/lifesim/bin/activate
#   pip install -r requirements.txt
source "${LIFESIM_VENV:-$HOME/venvs/lifesim}/bin/activate"

# numpy must not spawn its own threads inside each worker: we parallelise over processes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Write results somewhere with space/quota, not $HOME (see the docs' storage page).
OUT="${LIFESIM_OUT:-runs/rl_${SLURM_JOB_ID}}"
RESUME_FLAG=""; [ "${RESUME:-0}" = "1" ] && { RESUME_FLAG="--resume"; OUT="${LIFESIM_OUT:?set LIFESIM_OUT to the run you resume}"; }

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
