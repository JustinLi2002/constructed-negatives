"""Topsy-Turvy scoring for the real-model PPI test (DESIGN.md, Amendment 4, A19-A21).

Same path as score_dscript.py (Amendment 1, A4), with the Topsy-Turvy weights:
the classifier is dscript_human_v2.pt, built with the package's own
dscript.pretrained.build_human_1 (the function get_pretrained("human_v2") uses).
Language model: get_pretrained("lm_v1"), set up exactly as in
dscript.language_model.lm_embed.  Each protein's language-model output is passed
once through the model's own projection (model.embedding) and cached; map_predict
is then called with model.embedding = None.  TF32 is off.

    python -I score_tt.py loadtest
    python -I score_tt.py g1   PAIRS_TSV FASTA OUT_DIR    (full mouse_test.tsv)
    python -I score_tt.py cli  SUBSET_TSV FASTA OUT_DIR   (official CLI on CPU)
    python -I score_tt.py main INPUT_DIR OUT_DIR

Environment: TT_PT (dscript_human_v2.pt); REALPPI_WORK for the CLI's h5 file.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch

TT_PT = os.environ["TT_PT"]
LABELS = ("positives", "O", "U")
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fasta(path):
    seqs, name, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def build_model():
    from dscript.pretrained import build_human_1
    return build_human_1(TT_PT)          # strict load_state_dict, eval mode


class Scorer:
    def __init__(self, device):
        from dscript.alphabets import Uniprot21
        from dscript.pretrained import get_pretrained
        self.device = device
        model = build_model()
        model.use_cuda = device.type == "cuda"
        self.model = model.to(device).eval()
        lm = get_pretrained("lm_v1")
        torch.nn.init.normal_(lm.proj.weight)          # as in lm_embed; transform() does not use proj
        lm.proj.bias = torch.nn.Parameter(torch.zeros(100))
        self.lm = lm.to(device).eval()
        self.proj = self.model.embedding
        self.model.embedding = None
        self.alphabet = Uniprot21()
        self.cache = {}

    @torch.no_grad()
    def embed(self, key, seq):
        if key not in self.cache:
            x = torch.from_numpy(self.alphabet.encode(seq.encode("utf-8"))).long().unsqueeze(0)
            z = self.lm.transform(x.to(self.device))
            self.cache[key] = self.proj(z).cpu()

    @torch.no_grad()
    def pair(self, a, b):
        try:
            _, p = self.model.map_predict(self.cache[a].to(self.device), self.cache[b].to(self.device))
            return float(p.item())
        except RuntimeError as exc:          # predict_serial skips such pairs too
            print(f"  skipped {a} x {b}: {exc}", flush=True)
            return float("nan")


def describe(model):
    keys = ("do_w", "do_sigmoid", "do_pool", "pool_size", "use_cuda")
    d = {k: getattr(model, k, None) for k in keys}
    for k in ("theta", "lambda_", "gamma"):
        if hasattr(model, k):
            d[k] = float(getattr(model, k).item())
    d["class"] = type(model).__module__ + "." + type(model).__name__
    return d


def meta(device, extra):
    import dscript
    m = {"model": "Topsy-Turvy (dscript human_v2)", "pt_sha256": sha256(TT_PT),
         "dscript": dscript.__version__, "torch": torch.__version__, "python": platform.python_version(),
         "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
         "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "host": platform.node()}
    m.update(extra)
    return m


def run_pairs(scorer, pairs, seqs):
    uniq = list(dict.fromkeys(x for a, b in pairs for x in (a, b)))
    t0 = time.time()
    for i, k in enumerate(uniq):
        scorer.embed(k, seqs[k])
        if (i + 1) % 1000 == 0:
            print(f"  embedded {i + 1}/{len(uniq)}", flush=True)
    t1 = time.time()
    res = {}
    for a, b in pairs:
        if (a, b) not in res:
            res[(a, b)] = scorer.pair(a, b)
    return res, {"pairs": len(pairs), "unique_proteins": len(uniq),
                 "seconds_embedding": round(t1 - t0, 1), "seconds_scoring": round(time.time() - t1, 1)}


def main():
    mode = sys.argv[1]
    if mode == "loadtest":
        device = torch.device("cpu")
        sc = Scorer(device)
        sd = torch.load(TT_PT, map_location="cpu", weights_only=True)
        info = {"model_built": describe(sc.model), "state_dict_tensors": len(sd),
                "state_dict_parameters": int(sum(v.numel() for v in sd.values()))}
        toy = "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARR"
        sc.embed("a", toy)
        sc.embed("b", toy[::-1])
        info["toy_forward_ok"] = sc.pair("a", "b") == sc.pair("a", "b")
        print(json.dumps(meta(device, info), indent=1, default=str))
        return
    if mode == "cli":
        subset, fasta, out_dir = sys.argv[2:5]
        os.makedirs(out_dir, exist_ok=True)
        pairs = [tuple(line.split()[:2]) for line in open(subset) if line.strip()]
        seqs = read_fasta(fasta)
        need = sorted({x for p in pairs for x in p})
        sub_fa = os.path.join(out_dir, "cli_subset.fasta")
        with open(sub_fa, "w") as out:
            for k in need:
                out.write(f">{k}\n{seqs[k]}\n")
        exe = os.path.join(os.path.dirname(sys.executable), "dscript")
        # Official two-step route on CPU (Amendment 2, A11).  predict_serial loads a
        # .sav/.pt --model as a whole pickled model, so the model built by
        # build_human_1 is saved whole and passed in (Amendment 4, A20).
        tmp = os.path.join(os.environ["REALPPI_WORK"], "cli_tt_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(tmp, exist_ok=False)          # h5 and model file stay out of the fetched outputs
        whole = os.path.join(tmp, "tt_human_v2_whole_model.sav")
        model = build_model()
        model.use_cuda = False
        torch.save(model, whole)
        h5 = os.path.join(tmp, "cli_subset.h5")
        cmd_embed = [exe, "embed", "--seqs", sub_fa, "-o", h5, "--device=-1"]
        cmd_pred = [exe, "predict_serial", "--pairs", subset, "--embeddings", h5, "--model", whole,
                    "--device=-1", "--load_proc", "1", "-o", os.path.join(out_dir, "cli_check")]
        t0 = time.time()
        subprocess.run(cmd_embed, check=True, cwd=tmp)
        subprocess.run(cmd_pred, check=True, cwd=tmp)
        with open(os.path.join(out_dir, "cli_manifest.json"), "w") as out:
            json.dump(meta(torch.device("cpu"), {"mode": "cli", "commands": [cmd_embed, cmd_pred],
                                                  "pairs": len(pairs), "h5": h5, "whole_model": whole,
                                                  "seconds": round(time.time() - t0, 1)}), out, indent=1)
        return
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif os.environ.get("REALPPI_ALLOW_CPU") == "1":     # smoke test on synthetic sequences only
        device = torch.device("cpu")
    else:
        sys.exit("no GPU visible; refusing to score on CPU")
    if mode == "g1":
        pairs_path, fasta, out_dir = sys.argv[2:5]
        os.makedirs(out_dir, exist_ok=False)
        rows = [line.rstrip("\n").split("\t") for line in open(pairs_path) if line.strip()]
        seqs = read_fasta(fasta)
        sc = Scorer(device)
        res, info = run_pairs(sc, [(r[0], r[1]) for r in rows], seqs)
        stem = os.path.splitext(os.path.basename(pairs_path))[0]
        with open(os.path.join(out_dir, f"{stem}_scores.tsv"), "w") as out:
            out.write("protein_a\tprotein_b\tlabel\tprob\n")
            for r in rows:
                out.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{res[(r[0], r[1])]!r}\n")
        m = meta(device, {"mode": "g1", "pairs_file": pairs_path, "model_built": describe(sc.model), **info})
    elif mode == "main":
        in_dir, out_dir = sys.argv[2:4]
        os.makedirs(out_dir, exist_ok=False)
        seqs = read_fasta(os.path.join(in_dir, "sequences.fasta"))
        per = {}
        for lab in LABELS:
            with open(os.path.join(in_dir, f"pairs_{lab}.tsv")) as fh:
                assert fh.readline().rstrip("\n") == "protein_a\tprotein_b"
                per[lab] = [tuple(line.rstrip("\n").split("\t")) for line in fh if line.strip()]
        sc = Scorer(device)
        res, info = run_pairs(sc, [p for lab in LABELS for p in per[lab]], seqs)
        for lab in LABELS:
            with open(os.path.join(out_dir, f"scores_tt_{lab}.tsv"), "w") as out:
                out.write("protein_a\tprotein_b\tprob\n")
                for a, b in per[lab]:
                    out.write(f"{a}\t{b}\t{res[(a, b)]!r}\n")
        m = meta(device, {"mode": "main", "model_built": describe(sc.model), **info,
                          "pairs_per_set": {k: len(v) for k, v in per.items()}})
    else:
        sys.exit(f"unknown mode {mode}")
    m["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "manifest.json"), "w") as out:
        json.dump(m, out, indent=1, default=str)
    print(json.dumps(m, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
