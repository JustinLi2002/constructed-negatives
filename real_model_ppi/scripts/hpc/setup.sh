#!/bin/bash
# Real-model PPI test: one-time setup on the HRP login node, run under setsid.
# Downloads (DESIGN.md section 12 and Amendment 1), two environments, load tests
# with no study or G1 pair, then submits run_all.sbatch.  Log: logs/setup.log.
set -euo pipefail
R=$HOME/realppi
S=$R/out/setup
mkdir -p $S $R/work $R/tools $R/envs $R/downloads
stamp() { echo "[$(date '+%F %T')] $*"; }
fail() { stamp "SETUP FAILED: $*"; touch $S/SETUP_FAILED; exit 1; }
trap 'fail "command failed near line $LINENO"' ERR
cd $R/work
export UV_CACHE_DIR=$R/tools/uv_cache UV_PYTHON_INSTALL_DIR=$R/tools/python UV_NO_PROGRESS=1

stamp "start on $(hostname)"
df -h $R | tail -1
(quota -s 2>&1 || true) | tail -4

stamp uv
BENV=$HOME/env/bin/python
[ -x $BENV ] || BENV=$(command -v python3)
$BENV -m venv $R/tools/uvenv
$R/tools/uvenv/bin/python -m pip install -q --disable-pip-version-check uv==0.13.9
UV=$R/tools/uvenv/bin/uv
$UV --version

stamp tunar-code
TUNA_COMMIT_PIN=104756851004a05ff4a27caf1c4c5f4a556cff7d
mkdir -p $R/downloads/tunar_code
git clone -q https://github.com/young-su-ko/TUnA-R.git $R/downloads/tunar_code/TUnA-R < /dev/null
git -C $R/downloads/tunar_code/TUnA-R -c advice.detachedHead=false checkout -q $TUNA_COMMIT_PIN < /dev/null
TUNA_COMMIT=$(git -C $R/downloads/tunar_code/TUnA-R rev-parse HEAD)
[ "$TUNA_COMMIT" = "$TUNA_COMMIT_PIN" ] || fail "TUnA-R commit $TUNA_COMMIT"
echo $TUNA_COMMIT > $S/tuna_code_commit.txt

stamp tuna-env
(cd $R/downloads/tunar_code/TUnA-R && UV_PROJECT_ENVIRONMENT=$R/envs/tuna $UV sync --frozen --no-dev --python 3.13)
$R/envs/tuna/bin/python -V

stamp dscript-env
$UV venv -q --python 3.12 $R/envs/dscript
$UV pip install -q --python $R/envs/dscript/bin/python dscript==0.3.1 torch==2.7.1
$UV pip freeze --python $R/envs/tuna/bin/python > $S/freeze_tuna.txt
$UV pip freeze --python $R/envs/dscript/bin/python > $S/freeze_dscript.txt

stamp tunar-weights
W=$R/downloads/tunar_weights
mkdir -p $W
REV=4e9ab56ffe2768daf632949543e5f440a4cd5a58
for f in config.json pytorch_model.bin README.md; do
  curl -sSfL --retry 3 -o $W/$f https://huggingface.co/yk0/tuna-r-tuna/resolve/$REV/$f
done
GOT=$(sha256sum $W/pytorch_model.bin | cut -d' ' -f1)
[ "$GOT" = 6756d40bb36c5f642014abc946200c56b64dd26feea5d0aebb6e515e5885e0bb ] || fail "TUnA-R checkpoint $GOT"

stamp esm2-150M
E=$R/downloads/torch_home/hub/checkpoints
mkdir -p $E
curl -sSfL --retry 3 -o $E/esm2_t30_150M_UR50D.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt
curl -sSfL --retry 3 -o $E/esm2_t30_150M_UR50D-contact-regression.pt \
  https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t30_150M_UR50D-contact-regression.pt
[ $(stat -c %s $E/esm2_t30_150M_UR50D.pt) -eq 592774773 ] || fail "ESM-2 size"

stamp dscript-models
D=$R/downloads/dscript_models
mkdir -p $D
for f in human_v1.sav lm_v1.sav dscript_lm_v1.pt; do
  curl -sSfL --retry 3 -o $D/$f https://cb.csail.mit.edu/dscript/data/models/$f
done
[ $(stat -c %s $D/human_v1.sav) -eq 2542758 ] || fail "human_v1.sav size"
[ $(stat -c %s $D/lm_v1.sav) -eq 238258695 ] || fail "lm_v1.sav size"
[ $(stat -c %s $D/dscript_lm_v1.pt) -eq 238235733 ] || fail "dscript_lm_v1.pt size"
PKG=$($R/envs/dscript/bin/python -I -c 'import dscript, os; print(os.path.dirname(dscript.__file__))')
cp $D/dscript_lm_v1.pt $PKG/dscript_lm_v1.pt
echo "package LM copied to $PKG"

stamp dscript-data
$R/envs/dscript/bin/python -I $R/scripts/fetch_inputs.py $R/downloads dscript_data
cat > $S/dscript_data.sha256 <<'EOF'
a51caf0b590decf96911b09d1e2cc6afc9a9d669d4e67d8bb3c2f1c94e16cd0b  human_train.tsv
19498f9761e9cb5645799f9e4e36f18b3abe7d2b015bf50b4a78d53283fe8f53  human_test.tsv
ff027c405225204c9c3469ee2aa6dee807253a00af96936f4776e3580319cb14  human.fasta
EOF
(cd $R/downloads/dscript_data && sha256sum -c $S/dscript_data.sha256)

stamp checksums
(cd $R/downloads && find . -path ./tunar_code -prune -o -type f -print0 | sort -z | xargs -0 sha256sum) > $S/downloads_sha256.txt
du -sh $R/downloads $R/envs $R/tools 2>/dev/null || true

stamp load-tests
export TORCH_HOME=$R/downloads/torch_home HF_HUB_OFFLINE=1
export TUNA_SRC=$R/downloads/tunar_code/TUnA-R/src TUNA_WEIGHTS=$W TUNA_CODE_COMMIT=$TUNA_COMMIT
export DSCRIPT_SAV=$D/human_v1.sav DSCRIPT_LM_SAV=$D/lm_v1.sav
$R/envs/tuna/bin/python -I $R/scripts/score_tuna.py loadtest > $S/loadtest_tuna.txt 2>&1
$R/envs/dscript/bin/python -I $R/scripts/score_dscript.py loadtest > $S/loadtest_dscript.txt 2>&1
$R/envs/dscript/bin/dscript --version > $S/dscript_cli.txt 2>&1 || true
$R/envs/dscript/bin/dscript predict_serial --help >> $S/dscript_cli.txt 2>&1
(sinfo -p general -o '%N %G %t' 2>&1 || true) | head -60 > $S/cluster.txt

stamp smoke-test-synthetic-sequences
# Same code paths as the job, on CPU, on made-up sequences only (inputs_smoke/).
SM=$R/work/smoke
mkdir -p $SM
TP=$R/envs/tuna/bin/python
DP=$R/envs/dscript/bin/python
export REALPPI_ALLOW_CPU=1
$TP -I $R/scripts/score_tuna.py g1 $R/inputs_smoke/g1_tuna_pairs.tsv $SM/g1_tuna > $SM/log.txt 2>&1
$DP -I $R/scripts/score_dscript.py g1 $R/inputs_smoke/human_test.tsv $R/inputs_smoke/sequences.fasta $SM/g1_dscript >> $SM/log.txt 2>&1
$DP -I $R/scripts/score_dscript.py cli $R/inputs_smoke/cli_subset.tsv $R/inputs_smoke/sequences.fasta $SM/g1_dscript >> $SM/log.txt 2>&1
$TP -I $R/scripts/score_tuna.py main $R/inputs_smoke $SM/main_tuna >> $SM/log.txt 2>&1
$DP -I $R/scripts/score_dscript.py main $R/inputs_smoke $SM/main_dscript >> $SM/log.txt 2>&1
unset REALPPI_ALLOW_CPU
$DP -I $R/scripts/smoke_check.py $SM | tee $S/smoke_check.txt

stamp submit
JID=$(sbatch --parsable $R/scripts/run_all.sbatch)
echo $JID > $S/job_id.txt
stamp "submitted job $JID"
touch $S/SETUP_OK
stamp done
