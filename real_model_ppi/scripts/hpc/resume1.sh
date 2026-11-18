#!/bin/bash
# Resume after setup.sh stopped at the CLI step of the synthetic smoke test
# (dscript 0.3.1 on PyPI takes an integer --device).  Downloads and environments
# are already in place and verified.  Reruns the D-SCRIPT load test, the whole
# smoke test in a new directory, then submits run_all.sbatch.
set -euo pipefail
R=$HOME/realppi
S=$R/out/setup
stamp() { echo "[$(date '+%F %T')] $*"; }
fail() { stamp "RESUME1 FAILED: $*"; touch $S/RESUME1_FAILED; exit 1; }
trap 'fail "command failed near line $LINENO"' ERR
cd $R/work
export TORCH_HOME=$R/downloads/torch_home HF_HUB_OFFLINE=1
export TUNA_SRC=$R/downloads/tunar_code/TUnA-R/src TUNA_WEIGHTS=$R/downloads/tunar_weights
export TUNA_CODE_COMMIT=$(cat $S/tuna_code_commit.txt)
export DSCRIPT_SAV=$R/downloads/dscript_models/human_v1.sav DSCRIPT_LM_SAV=$R/downloads/dscript_models/lm_v1.sav
TP=$R/envs/tuna/bin/python
DP=$R/envs/dscript/bin/python

stamp "resume1 on $(hostname)"
stamp loadtest-dscript
$DP -I $R/scripts/score_dscript.py loadtest > $S/loadtest_dscript_2.txt 2>&1

stamp smoke-test-synthetic-sequences
SM=$R/work/smoke2
mkdir $SM
export REALPPI_ALLOW_CPU=1
$TP -I $R/scripts/score_tuna.py g1 $R/inputs_smoke/g1_tuna_pairs.tsv $SM/g1_tuna > $SM/log.txt 2>&1
$DP -I $R/scripts/score_dscript.py g1 $R/inputs_smoke/human_test.tsv $R/inputs_smoke/sequences.fasta $SM/g1_dscript >> $SM/log.txt 2>&1
$DP -I $R/scripts/score_dscript.py cli $R/inputs_smoke/cli_subset.tsv $R/inputs_smoke/sequences.fasta $SM/g1_dscript >> $SM/log.txt 2>&1
$TP -I $R/scripts/score_tuna.py main $R/inputs_smoke $SM/main_tuna >> $SM/log.txt 2>&1
$DP -I $R/scripts/score_dscript.py main $R/inputs_smoke $SM/main_dscript >> $SM/log.txt 2>&1
unset REALPPI_ALLOW_CPU
$DP -I $R/scripts/smoke_check.py $SM | tee $S/smoke_check_2.txt

stamp submit
JID=$(sbatch --parsable $R/scripts/run_all.sbatch)
echo $JID > $S/job_id.txt
stamp "submitted job $JID"
touch $S/RESUME1_OK
stamp done
