#!/bin/bash
#SBATCH --job-name=realppi-tt
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mantis-045,mantis-046,mantis-047,mantis-048,mantis-049,mantis-075
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>
#SBATCH -o <HPC_HOME>/realppi/logs/run_%j.log
# Topsy-Turvy extension (DESIGN.md, Amendment 4, A21-A23): G1 on the released
# mouse test set, the official-CLI check, the gate, then the study samples.  Any
# failing step ends the job (set -e); the study pairs are scored only after the
# gate has written out/gate_tt/PASS.
set -euo pipefail
R=$HOME/realppi
export TORCH_HOME=$R/downloads/torch_home HF_HUB_OFFLINE=1
export TT_PT=$R/downloads/tt_weights/dscript_human_v2.pt
export OMP_NUM_THREADS=8 REALPPI_WORK=$R/work
DP=$R/envs/dscript/bin/python
stage() { echo "[$(date '+%F %T')] $*"; }
cd $R/work
stage "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

stage g1-tt
$DP -I $R/scripts/score_tt.py g1 $R/downloads/dscript_mouse/mouse_test.tsv \
    $R/downloads/dscript_mouse/mouse.fasta $R/out/g1_tt
stage g1-tt-cli
$DP -I $R/scripts/score_tt.py cli $R/inputs/g1_tt_cli_subset.tsv \
    $R/downloads/dscript_mouse/mouse.fasta $R/out/g1_tt
stage gate-tt
$DP -I $R/scripts/gate_check_tt.py $R
test -f $R/out/gate_tt/PASS

stage main-tt
$DP -I $R/scripts/score_tt.py main $R/inputs $R/out/main_tt
(cd $R/out && sha256sum main_tt/* g1_tt/*.tsv gate_tt/gate.json) > $R/out/OUTPUT_SHA256_TT
touch $R/out/DONE_TT
stage done
