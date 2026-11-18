#!/bin/bash
# Topsy-Turvy extension (DESIGN.md, Amendment 4): download the weights and the
# mouse G1 data into new directories, check their sha256 against the local
# downloads, run the load test and the synthetic-sequence smoke test, then submit
# run_tt.sbatch.
set -euo pipefail
R=$HOME/realppi
S=$R/out/setup
stamp() { echo "[$(date '+%F %T')] $*"; }
fail() { stamp "TT SETUP FAILED: $*"; touch $S/TT_SETUP_FAILED; exit 1; }
trap 'fail "command failed near line $LINENO"' ERR
cd $R/work
export TORCH_HOME=$R/downloads/torch_home HF_HUB_OFFLINE=1 REALPPI_WORK=$R/work
export TT_PT=$R/downloads/tt_weights/dscript_human_v2.pt
DP=$R/envs/dscript/bin/python
C=23cbb0fbbb454d09ee41ff749c1454d8b3c8a4b8

stamp "tt setup on $(hostname)"
mkdir $R/downloads/tt_weights $R/downloads/dscript_mouse
curl -sSfL --retry 3 -o $TT_PT https://cb.csail.mit.edu/dscript/data/models/dscript_human_v2.pt
curl -sSfL --retry 3 -o $R/downloads/dscript_mouse/mouse_test.tsv \
  https://raw.githubusercontent.com/samsledje/D-SCRIPT/$C/data/pairs/mouse_test.tsv
curl -sSfL --retry 3 -o $R/downloads/dscript_mouse/mouse.fasta \
  https://raw.githubusercontent.com/samsledje/D-SCRIPT/$C/data/seqs/mouse.fasta
(cd $R/downloads && sha256sum -c $R/inputs/tt_downloads_sha256.txt) | tee $S/tt_downloads_check.txt

stamp loadtest-tt
$DP -I $R/scripts/score_tt.py loadtest > $S/loadtest_tt.txt 2>&1

stamp smoke-test-tt-synthetic-sequences
SM=$R/work/smoke_tt
mkdir $SM
export REALPPI_ALLOW_CPU=1
$DP -I $R/scripts/score_tt.py g1 $R/inputs_smoke/human_test.tsv $R/inputs_smoke/sequences.fasta $SM/g1_tt > $SM/log.txt 2>&1
$DP -I $R/scripts/score_tt.py cli $R/inputs_smoke/cli_subset.tsv $R/inputs_smoke/sequences.fasta $SM/g1_tt >> $SM/log.txt 2>&1
$DP -I $R/scripts/score_tt.py main $R/inputs_smoke $SM/main_tt >> $SM/log.txt 2>&1
unset REALPPI_ALLOW_CPU
$DP -I $R/scripts/smoke_check_tt.py $SM | tee $S/smoke_check_tt.txt

stamp submit
JID=$(sbatch --parsable $R/scripts/run_tt.sbatch)
echo $JID > $S/job_id_tt.txt
stamp "submitted job $JID"
touch $S/TT_SETUP_OK
stamp done
