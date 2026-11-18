"""Unpack the base64 tarball printed by conn3.sh (run locally).

    python -I unpack_fetch.py CONN3_OUT_TXT DEST_DIR

DEST_DIR must not exist.  The base64 text is checked against the sha256 that
conn3.sh printed, the archive is extracted with the "data" filter (no absolute
paths, no links outside DEST_DIR), and, if the job finished, every file listed in
out/OUTPUT_SHA256 is checked against its recorded hash.
"""
import base64
import hashlib
import io
import os
import re
import sys
import tarfile


def main():
    src, dest = sys.argv[1:3]
    text = open(src, encoding="utf-8", newline="").read()
    # not anchored at the line start: a file printed just before may lack a final newline
    m = re.search(r"fetch bytes (\d+) sha256 ([0-9a-f]{64})$", text, re.M)
    if not m:
        sys.exit("no fetch line: the job had not finished or failed before the gate")
    begin = text.index("==TGZ-BEGIN==\n") + len("==TGZ-BEGIN==\n")
    end = text.index("==TGZ-END==", begin)
    b64 = text[begin:end].encode("ascii")
    if len(b64) != int(m.group(1)) or hashlib.sha256(b64).hexdigest() != m.group(2):
        sys.exit(f"base64 text does not match: {len(b64)} bytes, sha256 {hashlib.sha256(b64).hexdigest()}")
    data = base64.b64decode(b64)
    os.makedirs(dest, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(f"extracted {len(data)} bytes into {dest}")
    manifest = os.path.join(dest, "out", "OUTPUT_SHA256")
    if not os.path.exists(manifest):
        print("no out/OUTPUT_SHA256 (job did not reach the end)")
        return
    bad = 0
    lines = [line.split(None, 1) for line in open(manifest, encoding="utf-8") if line.strip()]
    for digest, name in lines:
        name = name.strip().lstrip("*")
        got = hashlib.sha256(open(os.path.join(dest, "out", name), "rb").read()).hexdigest()
        if got != digest:
            bad += 1
            print(f"HASH MISMATCH {name}")
    print(f"OUTPUT_SHA256: {len(lines) - bad} of {len(lines)} files match")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
