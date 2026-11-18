#!/bin/bash
#SBATCH --job-name=realppi
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mantis-045,mantis-046,mantis-047,mantis-048,mantis-049,mantis-075
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>
#SBATCH -o <HPC_HOME>/realppi/logs/run_%j.log
# Real-model PPI test (DESIGN.md, Amendment 1, A9): G1 for both models, then the
# gate, then the study pairs.  Any failing step ends the job (set -e), so study
# pairs are scored only after the gate has written out/gate/PASS.
set -euo pipefail
R=$HOME/realppi
export TORCH_HOME=$R/downloads/torch_home HF_HUB_OFFLINE=1
export TUNA_SRC=$R/downloads/tunar_code/TUnA-R/src TUNA_WEIGHTS=$R/downloads/tunar_weights
export TUNA_CODE_COMMIT=$(cat $R/out/setup/tuna_code_commit.txt)
export DSCRIPT_SAV=$R/downloads/dscript_models/human_v1.sav
export OMP_NUM_THREADS=8 REALPPI_WORK=$R/work
TP=$R/envs/tuna/bin/python
DP=$R/envs/dscript/bin/python
stage() { echo "[$(date '+%F %T')] $*"; }
cd $R/work
stage "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

stage g1-tuna
$TP -I $R/scripts/score_tuna.py g1 $R/inputs/g1_tuna_pairs.tsv $R/out/g1_tuna
stage g1-dscript
$DP -I $R/scripts/score_dscript.py g1 $R/downloads/dscript_data/human_test.tsv \
    $R/downloads/dscript_data/human.fasta $R/out/g1_dscript
stage g1-dscript-cli
$DP -I $R/scripts/score_dscript.py cli $R/inputs/g1_dscript_cli_subset.tsv \
    $R/downloads/dscript_data/human.fasta $R/out/g1_dscript
stage gate
$DP -I $R/scripts/gate_check.py $R
test -f $R/out/gate/PASS

stage main-tuna
$TP -I $R/scripts/score_tuna.py main $R/inputs $R/out/main_tuna
stage main-dscript
$DP -I $R/scripts/score_dscript.py main $R/inputs $R/out/main_dscript
(cd $R/out && sha256sum main_tuna/* main_dscript/* g1_tuna/* g1_dscript/*.tsv gate/gate.json) > $R/out/OUTPUT_SHA256
touch $R/out/DONE
stage done
