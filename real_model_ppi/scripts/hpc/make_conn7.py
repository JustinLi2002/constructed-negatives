"""Build the connection script for the Topsy-Turvy extension (run locally).

    python make_conn7.py OUT_DIR

Uploads the new scripts and inputs (DESIGN.md, Amendment 4), refuses to run if any
of them or any Topsy-Turvy output already exists on HRP, starts setup_tt.sh under
setsid (downloads with hash checks, load test, smoke test, sbatch run_tt.sbatch)
and waits at most about 2.5 minutes before printing the logs.
"""
import base64
import hashlib
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FILES = {  # path on HRP (relative to $R) -> local path (relative to real_model_ppi)
    "scripts/score_tt.py": "scripts/hpc/score_tt.py",
    "scripts/gate_check_tt.py": "scripts/hpc/gate_check_tt.py",
    "scripts/smoke_check_tt.py": "scripts/hpc/smoke_check_tt.py",
    "scripts/setup_tt.sh": "scripts/hpc/setup_tt.sh",
    "scripts/run_tt.sbatch": "scripts/hpc/run_tt.sbatch",
    "inputs/g1_tt_cli_subset.tsv": "results/tt_inputs/g1_tt_cli_subset.tsv",
    "inputs/tt_downloads_sha256.txt": "results/tt_inputs/downloads_sha256.txt",
}
MUST_NOT_EXIST = list(FILES) + ["out/g1_tt", "out/gate_tt", "out/main_tt", "out/DONE_TT",
                                "downloads/tt_weights", "downloads/dscript_mouse", "work/smoke_tt",
                                "out/setup/TT_SETUP_OK", "out/setup/TT_SETUP_FAILED"]


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc, src in sorted(FILES.items()):
            data = open(os.path.join(ROOT, src), "rb").read()
            if b"\r" in data:
                sys.exit(f"CR in {src}")
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mode, ti.mtime = len(data), 0o755 if src.endswith(".sh") else 0o644, 1790000000
            tar.addfile(ti, io.BytesIO(data))
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    b64 = base64.encodebytes(payload).decode()
    checks = "\n".join(f'[ -e "$R/{p}" ] && {{ echo "ABORT: $R/{p} exists"; exit 1; }}' for p in MUST_NOT_EXIST)
    script = f"""set -u
R=$HOME/realppi
S=$R/out/setup
echo "== conn7 $(date '+%F %T') on $(hostname)"
[ -f "$R/out/DONE" ] || {{ echo "ABORT: out/DONE missing"; exit 1; }}
{checks}
if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -q '^realppi'; then echo "ABORT: a realppi job exists"; exit 1; fi
base64 -d > "$R/payload_conn7.tgz" <<'B64'
{b64}B64
GOT=$(sha256sum "$R/payload_conn7.tgz" | cut -d' ' -f1)
if [ "$GOT" != "{digest}" ]; then echo "ABORT: payload hash $GOT"; exit 1; fi
cd "$R" || exit 1
tar xzf payload_conn7.tgz || {{ echo "ABORT: unpack"; exit 1; }}
sha256sum {' '.join(sorted(FILES))}
setsid bash "$R/scripts/setup_tt.sh" > "$R/logs/setup_tt.log" 2>&1 < /dev/null &
echo "setup_tt started $(date '+%F %T')"
i=0
while [ $i -lt 30 ]; do
  if [ -f "$S/TT_SETUP_OK" ] || [ -f "$S/TT_SETUP_FAILED" ]; then break; fi
  sleep 5
  i=$((i+1))
done
echo "== waited $((i*5)) s"
cat "$R/logs/setup_tt.log"
echo "== loadtest_tt"; tail -n 40 "$S/loadtest_tt.txt" 2>&1
echo "== smoke check"; cat "$S/smoke_check_tt.txt" 2>&1
echo "== smoke log tail"; tail -n 25 "$R/work/smoke_tt/log.txt" 2>&1 | cut -c1-300
echo "== squeue"; squeue -u "$USER" -o '%i %j %T %M %l %R' 2>&1 | head
echo "== conn7 end $(date '+%F %T')"
"""
    with open(os.path.join(out_dir, "conn7.sh"), "w", newline="\n") as fh:
        fh.write(script)
    print(f"payload {len(payload)} bytes sha256 {digest}")


if __name__ == "__main__":
    main()
