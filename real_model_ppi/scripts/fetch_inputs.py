"""Download the label-free inputs of the real-model PPI test (DESIGN.md section 12).

Standard library only, so the same script runs locally and on HRP.
Every item goes into its own new directory under the given root; an existing
non-empty directory is never written into.  A JSON manifest records the URL,
size, sha256 and time of every file.

    python -I fetch_inputs.py ROOT [ITEM ...]

Items: string_seqs, dscript_data, bernett_tuna, tunar_gold, snooppi_list,
snooppi_positive.  No model is run and no prediction score is used here
except that the SNOOPPI member is saved as released (it is the G1 reference).
"""
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

UA = {"User-Agent": "python-urllib (research download)"}
DSCRIPT_COMMIT = "23cbb0fbbb454d09ee41ff749c1454d8b3c8a4b8"
TUNA_COMMIT = "b5bda8fee261a4f27821738db995cf5883dcd133"
TUNAR_COMMIT = "104756851004a05ff4a27caf1c4c5f4a556cff7d"
SNOOPPI_ZIP = "https://zenodo.org/api/records/22116351/files/snooppi_zenodo.zip/content"
SNOOPPI_ZIP_SIZE = 3568525998

GH = "https://raw.githubusercontent.com"
ITEMS = {
    "string_seqs": [
        ("https://stringdb-downloads.org/download/protein.sequences.v12.0/"
         "9606.protein.sequences.v12.0.fa.gz", "9606.protein.sequences.v12.0.fa.gz"),
    ],
    "dscript_data": [
        (f"{GH}/samsledje/D-SCRIPT/{DSCRIPT_COMMIT}/data/pairs/human_train.tsv", "human_train.tsv"),
        (f"{GH}/samsledje/D-SCRIPT/{DSCRIPT_COMMIT}/data/pairs/human_test.tsv", "human_test.tsv"),
        (f"{GH}/samsledje/D-SCRIPT/{DSCRIPT_COMMIT}/data/seqs/human.fasta", "human.fasta"),
    ],
    "bernett_tuna": [
        (f"{GH}/Wang-lab-UCSD/TUnA/{TUNA_COMMIT}/data/raw/bernett/{p}_{k}_rr.txt", f"{p}_{k}_rr.txt")
        for p in ("Intra0", "Intra1", "Intra2") for k in ("pos", "neg")
    ] + [
        (f"{GH}/Wang-lab-UCSD/TUnA/{TUNA_COMMIT}/data/raw/bernett/human_swissprot_oneliner.fasta",
         "human_swissprot_oneliner.fasta"),
        (f"{GH}/Wang-lab-UCSD/TUnA/{TUNA_COMMIT}/data/raw/bernett/README.md", "README.md"),
    ],
    "tunar_gold": [
        (f"{GH}/young-su-ko/TUnA-R/{TUNAR_COMMIT}/data/gold_standard/{p}_interaction_1500_or_less.tsv",
         f"{p}_interaction_1500_or_less.tsv") for p in ("Intra0", "Intra1", "Intra2")
    ] + [
        (f"{GH}/young-su-ko/TUnA-R/{TUNAR_COMMIT}/data/gold_standard/human_swissprot_oneliner.fasta",
         "human_swissprot_oneliner.fasta"),
    ],
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_dir(root, name):
    d = os.path.join(root, name)
    if os.path.isdir(d) and os.listdir(d):
        sys.exit(f"refusing to write into non-empty {d}")
    os.makedirs(d, exist_ok=True)
    return d


def get(url, dest, tries=3):
    tmp = dest + ".part"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
                expected = r.headers.get("Content-Length")
                n = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
                    n += len(b)
            if expected is not None and int(expected) != n:
                raise IOError(f"short read {n} of {expected}")
            os.replace(tmp, dest)
            return
        except Exception as exc:  # retry, then fail loudly
            print(f"  retry {k + 1} for {url}: {exc}", flush=True)
            time.sleep(5 * (k + 1))
    sys.exit(f"download failed: {url}")


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests."""

    def __init__(self, url, size):
        self.url, self.size, self.pos, self.requests = url, size, 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        hdr = dict(UA, Range=f"bytes={self.pos}-{end}")
        for k in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=hdr),
                                            timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"server ignored Range (status {r.status})")
                    data = r.read()
                break
            except IOError as exc:
                if "ignored Range" in str(exc) or k == 2:
                    raise
                time.sleep(5 * (k + 1))
        self.requests += 1
        n = len(data)
        if n != end - self.pos + 1:
            raise IOError("ranged read returned the wrong length")
        b[:n] = data
        self.pos += n
        return n


def open_snooppi():
    raw = RangeFile(SNOOPPI_ZIP, SNOOPPI_ZIP_SIZE)
    return raw, zipfile.ZipFile(io.BufferedReader(raw, buffer_size=4 << 20))


def main():
    root = sys.argv[1]
    wanted = sys.argv[2:] or list(ITEMS) + ["snooppi_list", "snooppi_positive"]
    os.makedirs(root, exist_ok=True)
    manifest_path = os.path.join(root, "MANIFEST.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for item in wanted:
        d = new_dir(root, item)
        rec = []
        if item in ITEMS:
            for url, fname in ITEMS[item]:
                dest = os.path.join(d, fname)
                print(f"{item}: {url}", flush=True)
                get(url, dest)
                rec.append({"url": url, "file": fname, "bytes": os.path.getsize(dest),
                            "sha256": sha256_of(dest)})
        elif item in ("snooppi_list", "snooppi_positive"):
            raw, zf = open_snooppi()
            if item == "snooppi_list":
                listing = os.path.join(d, "zip_members.tsv")
                with open(listing, "w", encoding="utf-8") as out:
                    out.write("name\tfile_size\tcompress_size\tcrc32\n")
                    for zi in zf.infolist():
                        out.write(f"{zi.filename}\t{zi.file_size}\t{zi.compress_size}\t{zi.CRC:08x}\n")
                rec.append({"url": SNOOPPI_ZIP, "file": "zip_members.tsv",
                            "bytes": os.path.getsize(listing), "sha256": sha256_of(listing),
                            "range_requests": raw.requests})
            else:
                members = [n for n in zf.namelist() if n.endswith("snooppi_positive__all_models.parquet")]
                if len(members) != 1:
                    sys.exit(f"expected one snooppi_positive member, found {members}")
                dest = os.path.join(d, "snooppi_positive__all_models.parquet")
                with zf.open(members[0]) as src, open(dest + ".part", "wb") as out:
                    while True:  # zipfile checks the CRC-32 when the member is read to its end
                        b = src.read(1 << 20)
                        if not b:
                            break
                        out.write(b)
                os.replace(dest + ".part", dest)
                zi = zf.getinfo(members[0])
                rec.append({"url": SNOOPPI_ZIP, "zip_member": members[0],
                            "file": os.path.basename(dest), "bytes": os.path.getsize(dest),
                            "crc32_checked": f"{zi.CRC:08x}", "sha256": sha256_of(dest),
                            "range_requests": raw.requests})
        else:
            sys.exit(f"unknown item {item}")
        manifest[item] = {"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "files": rec}
        with open(manifest_path, "w") as out:
            json.dump(manifest, out, indent=1)
        for r in rec:
            print(f"  {r['file']}  {r['bytes']:,} bytes  sha256 {r['sha256'][:16]}", flush=True)


if __name__ == "__main__":
    main()
