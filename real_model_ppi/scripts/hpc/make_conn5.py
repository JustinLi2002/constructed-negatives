"""Build the connection script that submits the continuation job (run locally).

    python make_conn5.py OUT_DIR

Uploads gate_check_a3.py and run_main.sbatch (DESIGN.md Amendment 3) as new
files, checks that the first job stopped at the gate with nothing scored after it,
submits run_main.sbatch and prints the queue.  No polling: the connection is short.
"""
import base64
import hashlib
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"scripts/gate_check_a3.py": "gate_check_a3.py", "scripts/run_main.sbatch": "run_main.sbatch"}


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
            ti.size, ti.mode, ti.mtime = len(data), 0o644, 1790000000
            tar.addfile(ti, io.BytesIO(data))
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    b64 = base64.encodebytes(payload).decode()
    script = f"""set -u
R=$HOME/realppi
echo "== conn5 $(date '+%F %T') on $(hostname)"
cd "$R" || exit 1
if [ ! -f out/gate/gate.json ] || [ -e out/gate/PASS ] || [ -e out/gate_a3 ] || [ -e out/main_tuna ] \\
   || [ -e out/main_dscript ] || [ -e out/DONE ] || [ -e scripts/gate_check_a3.py ] || [ -e scripts/run_main.sbatch ]; then
  echo "ABORT: unexpected state"; ls -la out scripts; exit 1
fi
if ! tail -n 2 out/gate/gate.json | grep -q '"pass": false'; then echo "ABORT: gate.json does not record a failure"; exit 1; fi
if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -q '^realppi'; then echo "ABORT: a realppi job exists"; exit 1; fi
base64 -d > "$R/payload_conn5.tgz" <<'B64'
{b64}B64
GOT=$(sha256sum "$R/payload_conn5.tgz" | cut -d' ' -f1)
if [ "$GOT" != "{digest}" ]; then echo "ABORT: payload hash $GOT"; exit 1; fi
tar xzf payload_conn5.tgz || {{ echo "ABORT: unpack"; exit 1; }}
sha256sum scripts/gate_check_a3.py scripts/run_main.sbatch
JID=$(sbatch --parsable scripts/run_main.sbatch) || {{ echo "ABORT: sbatch"; exit 1; }}
echo "$JID" > out/setup/job_id_main.txt
echo "submitted job $JID"
sleep 5
echo "== squeue"; squeue -u "$USER" -o '%i %j %T %M %l %R' 2>&1 | head
echo "== conn5 end $(date '+%F %T')"
"""
    with open(os.path.join(out_dir, "conn5.sh"), "w", newline="\n") as fh:
        fh.write(script)
    print(f"payload {len(payload)} bytes sha256 {digest}")


if __name__ == "__main__":
    main()
