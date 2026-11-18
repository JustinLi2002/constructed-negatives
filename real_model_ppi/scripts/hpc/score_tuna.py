"""TUnA-R scoring for the real-model PPI test (DESIGN.md, Amendment 1, A3).

Same path as SNOOPPI's run_tuna_june.py: ESMEmbedder with one sequence per
forward pass, EmbeddingStore, Predictor.predict_from_embeddings; the score is
the returned probability.  The checkpoint sha256 is checked before loading.

    python -I score_tuna.py loadtest
    python -I score_tuna.py g1   G1_PAIRS_TSV OUT_DIR
    python -I score_tuna.py main INPUT_DIR OUT_DIR

Environment: TUNA_SRC (TUnA-R clone's src/), TUNA_WEIGHTS (snapshot dir),
TORCH_HOME (ESM-2 weights), TUNA_CODE_COMMIT (recorded only).
"""
import hashlib
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.environ["TUNA_SRC"])
import torch  # noqa: E402

JUNE_SHA = "6756d40bb36c5f642014abc946200c56b64dd26feea5d0aebb6e515e5885e0bb"
WDIR = os.environ["TUNA_WEIGHTS"]
LABELS = ("positives", "O", "U")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load(device):
    got = sha256(os.path.join(WDIR, "pytorch_model.bin"))
    if got != JUNE_SHA:
        sys.exit(f"checkpoint sha256 {got} is not the pinned June checkpoint")
    from tuna.inference.embeddings import EmbeddingStore, ESMEmbedder
    from tuna.inference.predictor import Predictor
    predictor = Predictor.from_pretrained(WDIR, device)
    embedder = ESMEmbedder(predictor.device)
    return predictor, embedder, EmbeddingStore(), got


def meta(device, sha, extra):
    m = {"model": "TUnA-R", "checkpoint_sha256": sha, "code_commit": os.environ.get("TUNA_CODE_COMMIT"),
         "torch": torch.__version__, "python": platform.python_version(), "device": str(device),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
         "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
         "embed_batch_size": 1, "host": platform.node()}
    m.update(extra)
    return m


def score(pairs, device):
    """pairs: list of (key, seq_a, seq_b).  Returns {key: (prob, uncertainty)}."""
    predictor, embedder, store, sha = load(device)
    uniq = list(dict.fromkeys(s for _, a, b in pairs for s in (a, b)))
    t0 = time.time()
    for i in range(0, len(uniq), 500):
        store.ensure(uniq[i:i + 500], embedder, batch_size=1)
        print(f"  embedded {min(i + 500, len(uniq))}/{len(uniq)}", flush=True)
    t_embed = time.time() - t0
    out = {}
    with torch.no_grad():
        for key, a, b in pairs:
            out[key] = predictor.predict_from_embeddings(store[a], store[b])
    info = {"pairs": len(pairs), "unique_sequences": len(uniq),
            "seconds_embedding": round(t_embed, 1), "seconds_scoring": round(time.time() - t0 - t_embed, 1)}
    return out, sha, info


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif line:
            seqs[name] += line
    return seqs


def main():
    mode = sys.argv[1]
    if mode == "loadtest":
        device = torch.device("cpu")
        toy = "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARR"
        out, sha, info = score([("toy", toy, toy[::-1])], device)
        print(json.dumps(meta(device, sha, {"loadtest_ok": True, **info}), indent=1))
        return
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif os.environ.get("REALPPI_ALLOW_CPU") == "1":     # smoke test on synthetic sequences only
        device = torch.device("cpu")
    else:
        sys.exit("no GPU visible; refusing to score on CPU")
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=False)
    if mode == "g1":
        rows = []
        with open(sys.argv[2]) as fh:
            assert fh.readline().split("\t")[:3] == ["row_index", "seq_a", "seq_b"]
            for line in fh:
                f = line.rstrip("\n").split("\t")
                rows.append((int(f[0]), f[1], f[2]))   # the reference column is not read here
        res, sha, info = score(rows, device)
        with open(os.path.join(out_dir, "g1_tuna_scores.tsv"), "w") as out:
            out.write("row_index\tprob\tuncertainty\n")
            for k, _, _ in rows:
                out.write(f"{k}\t{res[k][0]!r}\t{res[k][1]!r}\n")
        m = meta(device, sha, {"mode": "g1", **info})
    elif mode == "main":
        in_dir = sys.argv[2]
        seqs = read_fasta(os.path.join(in_dir, "sequences.fasta"))
        allpairs, per = [], {}
        for lab in LABELS:
            with open(os.path.join(in_dir, f"pairs_{lab}.tsv")) as fh:
                assert fh.readline().rstrip("\n") == "protein_a\tprotein_b"
                per[lab] = [tuple(line.rstrip("\n").split("\t")) for line in fh if line.strip()]
            allpairs += [((a, b), seqs[a], seqs[b]) for a, b in per[lab]]
        uniq_pairs = list({k: (k, sa, sb) for k, sa, sb in allpairs}.values())
        res, sha, info = score(uniq_pairs, device)
        for lab in LABELS:
            with open(os.path.join(out_dir, f"scores_tuna_{lab}.tsv"), "w") as out:
                out.write("protein_a\tprotein_b\tprob\tuncertainty\n")
                for a, b in per[lab]:
                    p, u = res[(a, b)]
                    out.write(f"{a}\t{b}\t{p!r}\t{u!r}\n")
        m = meta(device, sha, {"mode": "main", **info, "pairs_per_set": {k: len(v) for k, v in per.items()}})
    else:
        sys.exit(f"unknown mode {mode}")
    m["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "manifest.json"), "w") as out:
        json.dump(m, out, indent=1)
    print(json.dumps(m, indent=1), flush=True)


if __name__ == "__main__":
    main()
