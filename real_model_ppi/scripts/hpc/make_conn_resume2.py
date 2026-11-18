"""Build the resume connection script (run locally).

    python make_conn_resume1.py OUT_DIR

The script uploads score_dscript.py (two-step CLI check), run_all.sbatch (exports
REALPPI_WORK) and resume2.sh, keeps the old two files beside them under new names,
starts resume2.sh under setsid
and waits at most about 2.5 minutes before printing the logs.
"""
import base64
import hashlib
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"scripts/score_dscript.py": "score_dscript.py", "scripts/run_all.sbatch": "run_all.sbatch", "scripts/resume2.sh": "resume2.sh"}


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc, src in sorted(FILES.items()):
            data = open(os.path.join(HERE, src), "rb").read()
            if b"\r" in data:
                sys.exit(f"CR in {src}")
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mode, ti.mtime = len(data), 0o755 if src.endswith(".sh") else 0o644, 1790000000
            tar.addfile(ti, io.BytesIO(data))
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    b64 = base64.encodebytes(payload).decode()
    script = f"""set -u
R=$HOME/realppi
S=$R/out/setup
echo "== resume2 conn $(date '+%F %T') on $(hostname)"
if [ ! -f "$S/RESUME1_FAILED" ] || [ -e "$S/RESUME1_OK" ] || [ -e "$S/RESUME2_OK" ] || [ -e "$S/RESUME2_FAILED" ] || [ -e "$R/work/smoke3" ]; then
  echo "ABORT: unexpected state"; ls -la "$S" "$R/work"; exit 1
fi
if [ -n "$(squeue -u "$USER" -h -n realppi 2>/dev/null)" ]; then echo "ABORT: a realppi job exists"; exit 1; fi
base64 -d > "$R/payload_resume2.tgz" <<'B64'
{b64}B64
GOT=$(sha256sum "$R/payload_resume2.tgz" | cut -d' ' -f1)
if [ "$GOT" != "{digest}" ]; then echo "ABORT: payload hash $GOT"; exit 1; fi
cd "$R" || exit 1
if [ -e scripts/score_dscript_v2_cli_seqs_bug.py ] || [ -e scripts/run_all_v1_no_workdir.sbatch ]; then echo "ABORT: old names taken"; exit 1; fi
mv scripts/score_dscript.py scripts/score_dscript_v2_cli_seqs_bug.py && mv scripts/run_all.sbatch scripts/run_all_v1_no_workdir.sbatch || {{ echo "ABORT: mv"; exit 1; }}
tar xzf payload_resume2.tgz || {{ echo "ABORT: unpack"; exit 1; }}
sha256sum scripts/score_dscript.py scripts/run_all.sbatch scripts/resume2.sh
setsid bash "$R/scripts/resume2.sh" > "$R/logs/resume2.log" 2>&1 < /dev/null &
echo "resume2 started $(date '+%F %T')"
i=0
while [ $i -lt 30 ]; do
  if [ -f "$S/RESUME2_OK" ] || [ -f "$S/RESUME2_FAILED" ]; then break; fi
  sleep 5
  i=$((i+1))
done
echo "== waited $((i*5)) s"
cat "$R/logs/resume2.log"
echo "== smoke check"; cat "$S/smoke_check_3.txt" 2>&1
echo "== smoke log tail"; tail -n 25 "$R/work/smoke3/log.txt" 2>&1
echo "== squeue"; squeue -u "$USER" -o '%i %j %T %M %l %R' 2>&1 | head
echo "== end $(date '+%F %T')"
"""
    with open(os.path.join(out_dir, "conn_resume2.sh"), "w", newline="\n") as fh:
        fh.write(script)
    print(f"payload {len(payload)} bytes sha256 {digest}")


if __name__ == "__main__":
    main()
