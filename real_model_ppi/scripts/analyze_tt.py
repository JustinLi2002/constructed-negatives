"""Secondary analysis of the real-model PPI test: Topsy-Turvy extension
(DESIGN.md, Amendment 4, A22).

    python -I analyze_tt.py TARGET_DIR SCORES_DIR PRIMARY_JSON OUT_JSON

SCORES_DIR holds scores_{tuna,dscript,tt}_{lab}.tsv.  Pairs (Delta = second - first):
  (TUnA-R, D-SCRIPT)    the primary pair, recomputed here and required to match
                        PRIMARY_JSON (the output of analyze.py) exactly;
  (TUnA-R, Topsy-Turvy) and (D-SCRIPT, Topsy-Turvy), secondary.
Same functions, seed, protein list and bootstrap replicates as analyze.py.
"""
import json
import os
import site
import sys

sys.path.append(site.getusersitepackages())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import analyze as A  # noqa: E402

MODELS = {"tuna": "TUnA-R", "dscript": "D-SCRIPT", "tt": "Topsy-Turvy"}
PAIRS = (("tuna", "dscript"), ("tuna", "tt"), ("dscript", "tt"))
REGIONS = ("O", "U")


def main():
    target_dir, scores_dir, primary_json, out_json = sys.argv[1:5]
    W = json.load(open(os.path.join(target_dir, "target_summary.json")))["target"]["W_T"]
    pairs = {lab: A.read_pairs(os.path.join(target_dir, f"pairs_{lab}.tsv")) for lab in A.LABELS}
    sc = {}
    for tag in MODELS:
        for lab in A.LABELS:
            s = A.read_scores(os.path.join(scores_dir, f"scores_{tag}_{lab}.tsv"))
            v = np.array([s[p] for p in pairs[lab]])
            if not np.isfinite(v).all():
                sys.exit(f"{tag} {lab}: {int((~np.isfinite(v)).sum())} missing scores; analysis not run")
            sc[(tag, lab)] = v
    prots = sorted({x for lab in A.LABELS for p in pairs[lab] for x in p})
    idx = {p: i for i, p in enumerate(prots)}
    ends = {lab: (np.array([idx[a] for a, _ in pairs[lab]]), np.array([idx[b] for _, b in pairs[lab]]))
            for lab in A.LABELS}
    auc = {(m, r): A.WeightedAUC(sc[(m, "positives")], sc[(m, r)]) for m in MODELS for r in REGIONS}

    def stats(w):
        a = {(m, r): auc[(m, r)](w["positives"], w[r]) for m in MODELS for r in REGIONS}
        out = {}
        for first, second in PAIRS:
            d_o = a[(second, "O")] - a[(first, "O")]
            d_u = a[(second, "U")] - a[(first, "U")]
            d_t = (1 - W) * d_o + W * d_u
            out[(first, second)] = {
                "AUROC_A_O": a[(first, "O")], "AUROC_A_U": a[(first, "U")],
                "AUROC_B_O": a[(second, "O")], "AUROC_B_U": a[(second, "U")],
                "AUROC_A_target": (1 - W) * a[(first, "O")] + W * a[(first, "U")],
                "AUROC_B_target": (1 - W) * a[(second, "O")] + W * a[(second, "U")],
                "Delta_O": d_o, "Delta_U": d_u, "Delta_target": d_t, "G": d_t - d_o,
                "ratio_target_over_O": d_t / d_o if d_o != 0 else float("nan")}
        return out

    point = stats({lab: np.ones(len(pairs[lab])) for lab in A.LABELS})
    rng = np.random.default_rng(A.SEED)
    reps = {pr: {k: [] for k in point[pr]} for pr in PAIRS}
    redraws, accepted = 0, 0
    while accepted < A.B:
        om = rng.poisson(1.0, len(prots)).astype(float)
        w = {lab: om[ends[lab][0]] * om[ends[lab][1]] for lab in A.LABELS}
        if min(w[lab].sum() for lab in A.LABELS) == 0:
            redraws += 1
            continue
        accepted += 1
        for pr, d in stats(w).items():
            for k, v in d.items():
                reps[pr][k].append(v)

    results = {}
    for pr in PAIRS:
        iv = {k: A.ci(np.array(v)) for k, v in reps[pr].items() if k != "ratio_target_over_O"}
        p = point[pr]
        if (A.excludes_zero(iv["Delta_O"]) and A.excludes_zero(iv["Delta_target"])
                and np.sign(p["Delta_O"]) != np.sign(p["Delta_target"])):
            reading = "reversal"
        elif A.excludes_zero(iv["Delta_O"]):
            reading = "amplification or shrinkage"
            iv["ratio_target_over_O"] = A.ci(np.array(reps[pr]["ratio_target_over_O"]))
        else:
            reading = "no statement about the order"
        results[f"{pr[0]}_vs_{pr[1]}"] = {
            "model_A": MODELS[pr[0]], "model_B": MODELS[pr[1]], "Delta": "AUROC_B - AUROC_A",
            "role": "primary (recomputed)" if pr == PAIRS[0] else "secondary (Amendment 4)",
            "point": p, "ci95": iv, "reading": reading}

    # The primary pair must reproduce analyze.py exactly.
    prim = json.load(open(primary_json))
    mine = results["tuna_vs_dscript"]
    bad = [k for k in prim["point"] if abs(prim["point"][k] - mine["point"][k]) > 1e-12]
    bad += [k for k in prim["ci95"] if any(abs(x - y) > 1e-12 for x, y in zip(prim["ci95"][k], mine["ci95"][k]))]
    if bad or prim["reading"] != mine["reading"] or prim["redraws"] != redraws:
        sys.exit(f"primary pair does not reproduce {primary_json}: {bad}; extension not reported")

    band = {str(e): {"G_plus_minus": 4 * W * e, "Delta_R_plus_minus": 2 * e,
                     "Delta_target_plus_minus": 2 * e} for e in A.EPS}
    res = {"W": W, "B": A.B, "seed": A.SEED, "redraws": redraws, "primary_reproduced": True,
           "pairs": {lab: len(pairs[lab]) for lab in A.LABELS}, "proteins": len(prots),
           "comparisons": results, "label_band_separate_from_ci": band}
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"primary pair reproduced exactly; W {W:.4f}; redraws {redraws}")
    for name, r in results.items():
        print(f"\n{r['model_B']} - {r['model_A']}  ({r['role']}); reading: {r['reading']}")
        for k in ("AUROC_A_O", "AUROC_A_U", "AUROC_B_O", "AUROC_B_U", "AUROC_A_target", "AUROC_B_target",
                  "Delta_O", "Delta_U", "Delta_target", "G"):
            print(f"  {k:16s} {r['point'][k]:+.4f}  [{r['ci95'][k][0]:+.4f}, {r['ci95'][k][1]:+.4f}]")
        if "ratio_target_over_O" in r["ci95"]:
            q = r["ci95"]["ratio_target_over_O"]
            print(f"  ratio            {r['point']['ratio_target_over_O']:+.3f}  [{q[0]:+.3f}, {q[1]:+.3f}]")


if __name__ == "__main__":
    main()
