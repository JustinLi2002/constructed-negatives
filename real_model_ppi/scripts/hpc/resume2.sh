#!/bin/bash
# Second resume: the CLI check now uses the official two-step route (dscript embed,
# then predict_serial --embeddings).  Reruns the whole synthetic smoke test in a new
# directory, then submits run_all.sbatch.
set -euo pipefail
R=$HOME/realppi
S=$R/out/setup
stamp() { echo "[$(date '+%F %T')] $*"; }
fail() { stamp "RESUME2 FAILED: $*"; touch $S/RESUME2_FAILED; exit 1; }
trap 'fail "command failed near line $LINENO"' ERR
cd $R/work
export TORCH_HOME=$R/downloads/torch_home HF_HUB_OFFLINE=1
export TUNA_SRC=$R/downloads/tunar_code/TUnA-R/src TUNA_WEIGHTS=$R/downloads/tunar_weights
export TUNA_CODE_COMMIT=$(cat $S/tuna_code_commit.txt)
export DSCRIPT_SAV=$R/downloads/dscript_models/human_v1.sav REALPPI_WORK=$R/work
TP=$R/envs/tuna/bin/python
DP=$R/envs/dscript/bin/python

stamp "resume2 on $(hostname)"
stamp smoke-test-synthetic-sequences
SM=$R/work/smoke3
mkdir $SM
export REALPPI_ALLOW_CPU=1
$TP -I $R/scripts/score_tuna.py g1 $R/inputs_smoke/g1_tuna_pairs.tsv $SM/g1_tuna > $SM/log.txt 2>&1
$DP -I $R/scripts/score_dscript.py g1 $R/inputs_smoke/human_test.tsv $R/inputs_smoke/sequences.fasta $SM/g1_dscript >> $SM/log.txt 2>&1
$DP -I $R/scripts/score_dscript.py cli $R/inputs_smoke/cli_subset.tsv $R/inputs_smoke/sequences.fasta $SM/g1_dscript >> $SM/log.txt 2>&1
$TP -I $R/scripts/score_tuna.py main $R/inputs_smoke $SM/main_tuna >> $SM/log.txt 2>&1
$DP -I $R/scripts/score_dscript.py main $R/inputs_smoke $SM/main_dscript >> $SM/log.txt 2>&1
unset REALPPI_ALLOW_CPU
$DP -I $R/scripts/smoke_check.py $SM | tee $S/smoke_check_3.txt

stamp submit
JID=$(sbatch --parsable $R/scripts/run_all.sbatch)
echo $JID > $S/job_id.txt
stamp "submitted job $JID"
touch $S/RESUME2_OK
stamp done
