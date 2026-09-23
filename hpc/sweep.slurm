#!/bin/bash
#SBATCH --job-name=negsamp
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-15%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/negsamp_%A_%a.out
#SBATCH --error=logs/negsamp_%A_%a.err
#
# partition/qos/nodes/ntasks and the array throttle follow the first paper's ten
# SLURM scripts, which is the only evidence available here of what this cluster
# accepts.  No --mail-user: add your own if you want the notifications.  No GPU
# is requested because nothing here needs one.
#
# The publication-grade version of the prefilter-severity x feature-fidelity
# plane: every network crossed with every prefilter severity, 20 seeds per cell
# instead of the 3 the laptop run used.  Embarrassingly parallel, so one task
# per (network, severity) pair.
#
#   sbatch hpc/sweep.slurm
#
# Each task writes its own TSV; hpc/collect.py merges them.  Nothing here needs
# a GPU and nothing needs more than a few GB, so this should schedule quickly
# even on a busy partition.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# MAXTRAIN caps training pairs.  200000 leaves STRING at its natural size and
# caps only the dense networks; a smaller value equalises all four, which is what
# makes the cross-network comparison free of a training-size confound.  RESULTS
# keeps the two runs apart.
MAXTRAIN=${MAXTRAIN:-200000}
RESULTS=${RESULTS:-results}
mkdir -p logs "$RESULTS"
echo "MAXTRAIN=$MAXTRAIN  RESULTS=$RESULTS"

# Match BLAS threads to the cores actually requested.  These were pinned to 1 on
# the first run to stop tasks fighting over cores, but --cpus-per-task=4 already
# reserves them, so pinning to 1 just left three idle -- and on the dense
# networks the MLP fit is the whole cost.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null && conda activate base
python - <<'CHK' || { echo "environment is missing a dependency"; exit 1; }
import numpy, scipy, sklearn
print('numpy %s  scipy %s  sklearn %s'
      % (numpy.__version__, scipy.__version__, sklearn.__version__))
CHK

NETS=(string_phys_700 string_phys_900 hippie_063 biogrid_human)
FRACS=(0.064 0.2 0.5 1.0)

i=${SLURM_ARRAY_TASK_ID}
net=${NETS[$((i / 4))]}
frac=${FRACS[$((i % 4))]}
edges="data/${net}.tsv"

if [ ! -s "$edges" ]; then
    echo "no edge list at $edges -- run hpc/fetch_networks.sh first"
    exit 0
fi

echo "task $i : network=$net  prefilter_frac=$frac"
python ppi_consequence.py "$edges" --sweep --only-frac "$frac" --seeds 20 \
       --max-train "$MAXTRAIN" --out "$RESULTS/ppi_${net}_f${frac}"

# The rule characterisations do not depend on the severity grid or on MAXTRAIN --
# they train no model at all -- so they run once per network rather than once per
# task, and their output is identical between the two runs.
if [ "$frac" = "0.064" ]; then
    python cl3_exclusion.py "$edges" --out "$RESULTS/cl3_${net}"
    python cl3_exclusion.py "$edges" --prefilter-frac 0.064 \
           --out "$RESULTS/cl3full_${net}"
    python ddb_sampling.py "$edges" --out "$RESULTS/ddb_${net}"
fi

echo "task $i done"
