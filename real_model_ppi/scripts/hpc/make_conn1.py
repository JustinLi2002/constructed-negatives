"""Build the payload and the connection-1 script for HRP (run locally).

    python make_conn1.py OUT_DIR

Writes OUT_DIR/payload.tgz and OUT_DIR/conn1.sh.  conn1.sh is piped to
`ssh <user>@<hpc-login-host> 'bash -s'`; it unpacks the payload into a new
$HOME/realppi (aborting if that exists), verifies hashes, starts
setup.sh under setsid, waits up to about 6.5 minutes and prints the logs.
"""
import base64
import hashlib
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RM = os.path.dirname(os.path.dirname(HERE))
SCRIPTS = {
    "scripts/setup.sh": os.path.join(HERE, "setup.sh"),
    "scripts/run_all.sbatch": os.path.join(HERE, "run_all.sbatch"),
    "scripts/score_tuna.py": os.path.join(HERE, "score_tuna.py"),
    "scripts/score_dscript.py": os.path.join(HERE, "score_dscript.py"),
    "scripts/gate_check.py": os.path.join(HERE, "gate_check.py"),
    "scripts/smoke_check.py": os.path.join(HERE, "smoke_check.py"),
    "scripts/fetch_inputs.py": os.path.join(RM, "scripts", "fetch_inputs.py"),
}
INPUTS = ["pairs_positives.tsv", "pairs_O.tsv", "pairs_U.tsv", "sequences.fasta",
          "g1_tuna_pairs.tsv", "g1_dscript_cli_subset.tsv", "target_summary.json"]
SMOKE = ["sequences.fasta", "g1_tuna_pairs.tsv", "human_test.tsv", "cli_subset.tsv",
         "pairs_positives.tsv", "pairs_O.tsv", "pairs_U.tsv", "README.txt"]


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    files = dict(SCRIPTS)
    files.update({f"inputs/{f}": os.path.join(RM, "results", "target", f) for f in INPUTS})
    files.update({f"inputs_smoke/{f}": os.path.join(RM, "smoke_inputs", f) for f in SMOKE})
    blobs = {}
    for arc, src in files.items():
        data = open(src, "rb").read()
        if b"\r" in data:
            sys.exit(f"CR character in {src}")
        blobs[arc] = data
    sums = "".join(f"{hashlib.sha256(blobs[f'inputs/{f}']).hexdigest()}  {f}\n" for f in INPUTS)
    blobs["inputs/SHA256SUMS"] = sums.encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(blobs):
            ti = tarfile.TarInfo(arc)
            ti.size = len(blobs[arc])
            ti.mode = 0o755 if arc.endswith(".sh") else 0o644
            ti.mtime = 1790000000
            tar.addfile(ti, io.BytesIO(blobs[arc]))
    payload = buf.getvalue()
    open(os.path.join(out_dir, "payload.tgz"), "wb").write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    b64 = base64.encodebytes(payload).decode()
    script = f"""set -u
R=$HOME/realppi
echo "== conn1 start $(date '+%F %T') on $(hostname)"
if [ -e "$R" ]; then echo "ABORT: $R already exists"; ls -la "$R" | head -20; exit 1; fi
mkdir -p "$R/logs"
base64 -d > "$R/payload.tgz" <<'B64'
{b64}B64
GOT=$(sha256sum "$R/payload.tgz" | cut -d' ' -f1)
echo "payload sha256 $GOT"
if [ "$GOT" != "{digest}" ]; then echo "ABORT: payload hash mismatch"; exit 1; fi
cd "$R" || exit 1
tar xzf payload.tgz || {{ echo "ABORT: unpack failed"; exit 1; }}
(cd inputs && sha256sum -c --quiet SHA256SUMS) || {{ echo "ABORT: input hashes"; exit 1; }}
echo "inputs verified"
setsid bash "$R/scripts/setup.sh" > "$R/logs/setup.log" 2>&1 < /dev/null &
echo "setup started $(date '+%F %T')"
i=0
while [ $i -lt 78 ]; do
  if [ -f "$R/out/setup/SETUP_OK" ] || [ -f "$R/out/setup/SETUP_FAILED" ]; then break; fi
  sleep 5
  i=$((i+1))
done
echo "== waited $((i*5)) s"
echo "== setup.log"
tail -n 80 "$R/logs/setup.log"
for f in "$R"/out/setup/*.txt; do
  if [ -f "$f" ]; then echo "== $f"; head -n 70 "$f"; fi
done
echo "== squeue"
squeue -u "$USER" -o '%i %j %T %M %l %R' 2>&1 | head -20
echo "== conn1 end $(date '+%F %T')"
"""
    with open(os.path.join(out_dir, "conn1.sh"), "w", newline="\n") as fh:
        fh.write(script)
    print(f"payload {len(payload):,} bytes, sha256 {digest}; conn1.sh {len(script):,} bytes; files {len(blobs)}")


if __name__ == "__main__":
    main()
