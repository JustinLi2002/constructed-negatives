"""D-SCRIPT scoring for the real-model PPI test (DESIGN.md, Amendment 1, A4).

Classifier: human_v1.sav loaded as the pickled model object.  Language model:
the package's get_pretrained("lm_v1"), set up exactly as in
dscript.language_model.lm_embed.  Each protein's language-model output is passed
once through the model's own projection (model.embedding) and the result is
cached; map_predict is then called with model.embedding = None, which skips
only that already-applied projection.  TF32 is off.

    python -I score_dscript.py loadtest
    python -I score_dscript.py g1   PAIRS_TSV FASTA OUT_DIR    (full human_test.tsv)
    python -I score_dscript.py cli  SUBSET_TSV FASTA OUT_DIR   (official CLI on CPU)
    python -I score_dscript.py main INPUT_DIR OUT_DIR

Environment: DSCRIPT_SAV (human_v1.sav), DSCRIPT_LM_SAV (lm_v1.sav, loadtest only).
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch

SAV = os.environ["DSCRIPT_SAV"]
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


class Scorer:
    def __init__(self, device):
        from dscript.alphabets import Uniprot21
        from dscript.pretrained import get_pretrained
        self.device = device
        model = torch.load(SAV, map_location="cpu", weights_only=False)
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
    m = {"model": "D-SCRIPT human_v1", "sav_sha256": sha256(SAV), "dscript": dscript.__version__,
         "torch": torch.__version__, "python": platform.python_version(), "device": str(device),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
        info = {"pickled_model": describe(sc.model)}
        from dscript.pretrained import get_state_dict_path
        pkg_lm = get_state_dict_path("lm_v1")
        info["package_lm_file"] = pkg_lm
        info["package_lm_sha256"] = sha256(pkg_lm)
        lm_sav = os.environ.get("DSCRIPT_LM_SAV")
        if lm_sav:
            try:      # a record only, not a gate
                old = torch.load(lm_sav, map_location="cpu", weights_only=False)
                old_sd = {}          # state_dict() fails on this old pickled LSTM; read raw parameters
                for name, mod in old.named_modules():
                    for pn, p in mod._parameters.items():
                        if p is not None:
                            old_sd[f"{name}.{pn}" if name else pn] = p.detach()
                new_sd = torch.load(pkg_lm, map_location="cpu")
                shared = sorted(set(old_sd) & set(new_sd))
                diffs = [float((old_sd[k].float() - new_sd[k].float()).abs().max()) for k in shared
                         if old_sd[k].shape == new_sd[k].shape and "proj" not in k]
                info["lm_sav_vs_package_lm"] = {
                    "keys_old": len(old_sd), "keys_new": len(new_sd), "shared": len(shared),
                    "max_abs_diff_excluding_proj": max(diffs) if diffs else None}
            except Exception as exc:
                info["lm_sav_vs_package_lm"] = f"comparison not possible: {type(exc).__name__}: {exc}"
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
        # Official two-step route on CPU: `dscript embed`, then `predict_serial --embeddings`.
        # In dscript 0.3.1 from PyPI, -d/--device is an int (-1 = CPU) and the one-step
        # `predict_serial --seqs` route fails (parse() is called with two arguments).
        tmp = os.path.join(os.environ["REALPPI_WORK"], "cli_h5_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(tmp, exist_ok=False)          # the h5 file stays out of the fetched outputs
        h5 = os.path.join(tmp, "cli_subset.h5")
        cmd_embed = [exe, "embed", "--seqs", sub_fa, "-o", h5, "--device=-1"]
        cmd_pred = [exe, "predict_serial", "--pairs", subset, "--embeddings", h5, "--model", SAV,
                    "--device=-1", "--load_proc", "1", "-o", os.path.join(out_dir, "cli_check")]
        t0 = time.time()
        subprocess.run(cmd_embed, check=True, cwd=tmp)
        subprocess.run(cmd_pred, check=True, cwd=tmp)
        with open(os.path.join(out_dir, "cli_manifest.json"), "w") as out:
            json.dump(meta(torch.device("cpu"), {"mode": "cli", "commands": [cmd_embed, cmd_pred],
                                                  "pairs": len(pairs), "h5": h5,
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
        with open(os.path.join(out_dir, "human_test_scores.tsv"), "w") as out:
            out.write("protein_a\tprotein_b\tlabel\tprob\n")
            for r in rows:
                out.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{res[(r[0], r[1])]!r}\n")
        m = meta(device, {"mode": "g1", "pickled_model": describe(sc.model), **info})
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
            with open(os.path.join(out_dir, f"scores_dscript_{lab}.tsv"), "w") as out:
                out.write("protein_a\tprotein_b\tprob\n")
                for a, b in per[lab]:
                    out.write(f"{a}\t{b}\t{res[(a, b)]!r}\n")
        m = meta(device, {"mode": "main", "pickled_model": describe(sc.model), **info,
                          "pairs_per_set": {k: len(v) for k, v in per.items()}})
    else:
        sys.exit(f"unknown mode {mode}")
    m["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "manifest.json"), "w") as out:
        json.dump(m, out, indent=1, default=str)
    print(json.dumps(m, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
