"""Primary analysis of the real-model PPI test (DESIGN.md sections 6-7, Amendment 1 A10).

    python -I analyze.py TARGET_DIR SCORES_DIR OUT_JSON

TARGET_DIR holds pairs_{positives,O,U}.tsv and target_summary.json (W).
SCORES_DIR holds scores_tuna_{lab}.tsv and scores_dscript_{lab}.tsv.
Model A = TUnA-R, model B = D-SCRIPT; Delta_R = AUROC_B(R) - AUROC_A(R).
"""
import json
import os
import site
import sys

sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

SEED = 20261118
B = 2000
EPS = (0.01, 0.02, 0.05)
LABELS = ("positives", "O", "U")


def read_pairs(path):
    with open(path) as fh:
        assert fh.readline().rstrip("\n") == "protein_a\tprotein_b"
        return [tuple(line.rstrip("\n").split("\t")) for line in fh if line.strip()]


def read_scores(path):
    out = {}
    with open(path) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        k = head.index("prob")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            out[(f[0], f[1])] = float(f[k])
    return out


class WeightedAUC:
    """AUROC with pair weights; the sort is done once, each replicate is O(n)."""

    def __init__(self, pos, neg):
        order = np.argsort(pos, kind="mergesort")
        self.order = order
        ps = pos[order]
        self.hi = np.searchsorted(ps, neg, side="right")
        self.lo = np.searchsorted(ps, neg, side="left")

    def __call__(self, wpos, wneg):
        cw = np.concatenate([[0.0], np.cumsum(wpos[self.order])])
        tot = cw[-1]
        greater = tot - cw[self.hi]
        ties = cw[self.hi] - cw[self.lo]
        return float(np.dot(wneg, greater + 0.5 * ties) / (tot * wneg.sum()))


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def excludes_zero(iv):
    return iv[0] > 0 or iv[1] < 0


def main():
    target_dir, scores_dir, out_json = sys.argv[1:4]
    W = json.load(open(os.path.join(target_dir, "target_summary.json")))["target"]["W_T"]
    pairs = {lab: read_pairs(os.path.join(target_dir, f"pairs_{lab}.tsv")) for lab in LABELS}
    sc = {}
    for model, tag in (("A", "tuna"), ("B", "dscript")):
        for lab in LABELS:
            s = read_scores(os.path.join(scores_dir, f"scores_{tag}_{lab}.tsv"))
            v = np.array([s[p] for p in pairs[lab]])
            if not np.isfinite(v).all():
                sys.exit(f"{tag} {lab}: {int((~np.isfinite(v)).sum())} missing scores; analysis not run")
            sc[(model, lab)] = v
    prots = sorted({x for lab in LABELS for p in pairs[lab] for x in p})
    idx = {p: i for i, p in enumerate(prots)}
    ends = {lab: (np.array([idx[a] for a, _ in pairs[lab]]), np.array([idx[b] for _, b in pairs[lab]]))
            for lab in LABELS}
    auc = {(m, r): WeightedAUC(sc[(m, "positives")], sc[(m, r)]) for m in "AB" for r in ("O", "U")}

    def stats(w):
        a = {(m, r): auc[(m, r)](w["positives"], w[r]) for m in "AB" for r in ("O", "U")}
        d_o = a[("B", "O")] - a[("A", "O")]
        d_u = a[("B", "U")] - a[("A", "U")]
        d_t = (1 - W) * d_o + W * d_u
        return {"AUROC_A_O": a[("A", "O")], "AUROC_A_U": a[("A", "U")],
                "AUROC_B_O": a[("B", "O")], "AUROC_B_U": a[("B", "U")],
                "AUROC_A_target": (1 - W) * a[("A", "O")] + W * a[("A", "U")],
                "AUROC_B_target": (1 - W) * a[("B", "O")] + W * a[("B", "U")],
                "Delta_O": d_o, "Delta_U": d_u, "Delta_target": d_t, "G": d_t - d_o,
                "ratio_target_over_O": d_t / d_o if d_o != 0 else float("nan")}

    point = stats({lab: np.ones(len(pairs[lab])) for lab in LABELS})
    rng = np.random.default_rng(SEED)
    reps, redraws = {k: [] for k in point}, 0
    while len(reps["G"]) < B:
        om = rng.poisson(1.0, len(prots)).astype(float)
        w = {lab: om[ends[lab][0]] * om[ends[lab][1]] for lab in LABELS}
        if min(w[lab].sum() for lab in LABELS) == 0:
            redraws += 1
            continue
        for k, v in stats(w).items():
            reps[k].append(v)
    intervals = {k: ci(np.array(v)) for k, v in reps.items() if k != "ratio_target_over_O"}

    if (excludes_zero(intervals["Delta_O"]) and excludes_zero(intervals["Delta_target"])
            and np.sign(point["Delta_O"]) != np.sign(point["Delta_target"])):
        reading = "reversal"
    elif excludes_zero(intervals["Delta_O"]):
        reading = "amplification or shrinkage"
        intervals["ratio_target_over_O"] = ci(np.array(reps["ratio_target_over_O"]))
    else:
        reading = "no statement about the order"
    band = {str(e): {"G_plus_minus": 4 * W * e, "Delta_R_plus_minus": 2 * e,
                     "Delta_target_plus_minus": 2 * e} for e in EPS}
    res = {"W": W, "B": B, "seed": SEED, "redraws": redraws,
           "pairs": {lab: len(pairs[lab]) for lab in LABELS}, "proteins": len(prots),
           "point": point, "ci95": intervals, "reading": reading,
           "label_band_separate_from_ci": band,
           "model_A": "TUnA-R", "model_B": "D-SCRIPT", "Delta": "AUROC_B - AUROC_A"}
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    for k in ("AUROC_A_O", "AUROC_A_U", "AUROC_B_O", "AUROC_B_U", "Delta_O", "Delta_U", "Delta_target", "G"):
        print(f"{k:15s} {point[k]:+.4f}  [{intervals[k][0]:+.4f}, {intervals[k][1]:+.4f}]")
    if "ratio_target_over_O" in intervals:
        r = intervals["ratio_target_over_O"]
        print(f"ratio          {point['ratio_target_over_O']:+.3f}  [{r[0]:+.3f}, {r[1]:+.3f}]")
    print(f"W {W:.4f}; reading: {reading}; redraws {redraws}")


if __name__ == "__main__":
    main()
