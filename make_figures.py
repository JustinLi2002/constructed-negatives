#!/usr/bin/env python3
"""Main figures.

Every panel is drawn from a file in this repository, and the few small tables
that live only in printed output are transcribed at the top of this script with
the run that produced them named, so that a reader can check them against
results_*.txt rather than trusting the plot.

    python make_figures.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

TASKS = ["Phospho S/T", "Phospho Y", "Ubiquitination K", "Sumoylation K",
         "Acetylation K", "Methylation K/R", "Methylation R", "N-Glyc N"]

# results_ptm_guarantee.txt / results_ranking_identification.txt
W_DEPLOY = dict(zip(TASKS, [0.605, 0.642, 0.750, 0.913, 0.921, 0.974, 0.977,
                            0.977]))
DELTA_O_BENCH = dict(zip(TASKS, [0.0165, 0.0532, 0.0497, 0.0667, 0.0578,
                                 0.0378, 0.0534, 0.0100]))
# results_measured_c.txt  (esm2, train=replica, eval=rebuilt, T=10)
MEASURED_C = {"Phospho S/T": 0.0000, "Phospho Y": 0.0258,
              "Acetylation K": 0.0177, "Methylation K/R": 0.0474,
              "Methylation R": 0.1374, "Sumoylation K": 0.0033,
              "Ubiquitination K": 0.0134, "N-Glyc N": 0.0040}
# results_deployment_scope.txt
SCOPE = {"Phospho S/T": (1, 66.1), "Phospho Y": (2, 57.7),
         "Ubiquitination K": (3, 41.5), "Sumoylation K": (5, 12.0),
         "Acetylation K": (5, 10.6), "Methylation K/R": (7, 2.1),
         "Methylation R": (7, 2.1), "N-Glyc N": (6, 1.5)}
# (C) own gap under the measured c -- a different question from (A), so plotted
# as a separate mark rather than a second bar
SCOPE_MEAS = {"Phospho S/T": 100.0, "Phospho Y": 100.0,
              "Ubiquitination K": 100.0, "Sumoylation K": 100.0,
              "Acetylation K": 100.0, "Methylation K/R": 5.7,
              "Methylation R": 1.6, "N-Glyc N": 100.0}
# results_ptm_guarantee.txt, restricted ("seen") universe
W_RESTRICT = {"Phospho S/T": 0.472, "Phospho Y": 0.244, "Acetylation K": 0.817,
              "Methylation K/R": 0.936, "Methylation R": 0.915,
              "Sumoylation K": 0.779, "Ubiquitination K": 0.612,
              "N-Glyc N": 0.889}
# results_shift_check.txt, r2 = 0.90, STRING physical >= 700
# (W, unobserved-mass term, within-region shift term)
SHIFT = [(0.4789, -0.0262, +0.0018), (0.7880, -0.1451, +0.0001),
         (0.9178, -0.2991, -0.0044)]
# results_metric_linearity.txt
PRC_RESID = {"Phospho S/T": (2.75e-1, 0.6155), "Phospho Y": (2.77e-1, 0.4516),
             "Acetylation K": (3.57e-1, 0.4120),
             "Methylation K/R": (3.15e-1, 0.3764),
             "Methylation R": (2.97e-1, 0.4138),
             "Sumoylation K": (3.63e-1, 0.4058),
             "Ubiquitination K": (3.57e-1, 0.4910),
             "N-Glyc N": (1.73e-1, 0.7454)}
# results_threshold_tradeoff.txt, alpha = 0, mean over tasks
TRADE_T = [0, 1, 2, 3, 5, 10, 20]
TRADE_W = [0.000, 0.504, 0.613, 0.679, 0.757, 0.845, 0.916]
TRADE_FN = [0.2117, 0.1733, 0.1542, 0.1394, 0.1159, 0.0784, 0.0346]

NETS = [("string_phys_900", "STRING ≥900"), ("string_phys_700", "STRING ≥700"),
        ("biogrid_human", "BioGRID"), ("hippie_063", "HIPPIE")]
CB = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]


def read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def find(*cands):
    for c in cands:
        if os.path.exists(c):
            return c
    return None


# ---------------------------------------------------------------- figure 1
def figure1():
    """The mechanism: a propensity that reaches zero, and what follows.

    Panel a is the only invented panel in these figures; b and c are measured.
    """
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.8))

    d = np.linspace(0, 25, 400)
    T = 10
    viol = np.where(d >= T, 0.85, 0.0)
    shift = 0.9 * np.exp(-0.12 * (25 - d)) + 0.02
    ax[0].plot(d, shift, lw=1.4, color=CB[0], label="covariate shift")
    ax[0].plot(d, viol, lw=1.6, color=CB[1], label="positivity violation")
    ax[0].axvline(T, color="0.5", lw=0.8, ls=":")
    ax[0].fill_between(d, 0, 1.0, where=(d < T), color=CB[1], alpha=0.07)
    ax[0].text(T + 0.5, 0.06, "$T$", fontsize=7, color="0.35")
    ax[0].annotate("no negatives\ndrawn here", xy=(4.5, 0.30),
                   fontsize=6, color=CB[1], ha="center")
    ax[0].set_ylim(-0.03, 1.05)
    ax[0].set_xlabel("annotation depth")
    ax[0].set_ylabel("propensity of being a negative")
    ax[0].set_title("a   two ways to distort a benchmark", loc="left",
                    fontsize=8)
    ax[0].legend(frameon=False, fontsize=6, loc="upper left")

    y = np.arange(len(TASKS))
    ax[1].barh(y + 0.19, [W_DEPLOY[t] for t in TASKS], height=0.36,
               color=CB[1], label="whole proteome")
    ax[1].barh(y - 0.19, [W_RESTRICT[t] for t in TASKS], height=0.36,
               color=CB[0], label="proteins carrying a site")
    ax[1].axvline(0.5, color="0.4", lw=0.9, ls="--")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(TASKS, fontsize=6)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 1.05)
    ax[1].set_xlabel("unobserved weight $W$")
    ax[1].set_title("b   deployment mass never sampled", loc="left", fontsize=8)
    ax[1].legend(frameon=False, fontsize=5.5, loc="lower left",
                 bbox_to_anchor=(0.0, -0.50), ncol=2, handlelength=1.2,
                 columnspacing=1.0)

    W = [r[0] for r in SHIFT]
    unobs = [abs(r[1]) for r in SHIFT]
    shiftt = [abs(r[2]) for r in SHIFT]
    x = np.arange(len(W))
    ax[2].bar(x, unobs, width=0.5, color=CB[1],
              label="unobserved mass, unreachable")
    ax[2].bar(x, shiftt, width=0.5, bottom=unobs, color=CB[0],
              label="shift inside O, repairable")
    for i, (u, sh) in enumerate(zip(unobs, shiftt)):
        ax[2].text(i, u + sh + 0.012, "%.1f%%" % (100 * sh / (u + sh)),
                   ha="center", fontsize=5.5, color=CB[0])
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(["%.2f" % w for w in W])
    ax[2].set_ylim(0, 0.36)
    ax[2].set_xlabel("unobserved weight $W$")
    ax[2].set_ylabel("|contribution to the error|")
    ax[2].set_title("c   which half is repairable", loc="left", fontsize=8)
    # the blue segment is invisible at this scale, which is the point; the
    # percentages label it
    ax[2].legend(frameon=False, fontsize=5.5, loc="lower left",
                 bbox_to_anchor=(0.0, -0.50), ncol=1, handlelength=1.2)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_mechanism.png")
    plt.close(fig)
    print("fig1_mechanism.png")


# ---------------------------------------------------------------- figure 2
def figure2():
    """What a benchmark determines: worst case, measured c, and scope."""
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.8))

    # (a) required gap against W, worst case, with the eight tasks
    w = np.linspace(0.01, 0.985, 400)
    ax[0].plot(w, w / (1 - w), color="0.25", lw=1.2)
    ax[0].axhline(1.0, color=CB[1], lw=0.8, ls="--")
    ax[0].axvline(0.5, color=CB[1], lw=0.8, ls=":")
    for t in TASKS:
        W = W_DEPLOY[t]
        ax[0].plot(W, W / (1 - W), "o", ms=3.5, color=CB[0], zorder=3)
    ax[0].scatter([W_DEPLOY[t] for t in TASKS],
                  [DELTA_O_BENCH[t] for t in TASKS], s=9, marker="s",
                  color=CB[2], zorder=3, label="largest gap observed")
    ax[0].legend(frameon=False, fontsize=5.5, loc="lower left",
                 bbox_to_anchor=(0.0, -0.50), handlelength=1.2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("unobserved weight $W$")
    ax[0].set_ylabel("gap required, $W/(1-W)$")
    ax[0].set_title("a   worst case", loc="left", fontsize=8)
    # the dashed line is the ceiling a difference of AUROCs cannot pass; label it
    # in the empty upper-left rather than on top of the curve
    ax[0].text(0.03, 30, "max possible gap = 1", color=CB[1], fontsize=5.5)

    # (b) required gap under the measured c
    y = np.arange(len(TASKS))
    need = [W_DEPLOY[t] * MEASURED_C[t] for t in TASKS]
    ax[1].barh(y, need, color=CB[0], height=0.6, label="required, measured $c$")
    ax[1].plot([DELTA_O_BENCH[t] for t in TASKS], y, "s", ms=3.5,
               color=CB[2], label="gap observed")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(TASKS, fontsize=6)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("AUROC gap")
    ax[1].set_title("b   with $c$ measured", loc="left", fontsize=8)
    ax[1].legend(frameon=False, fontsize=6, loc="lower right", ncol=1)

    # (c) the population the benchmark supports
    share = [SCOPE[t][1] for t in TASKS]
    ax[2].barh(y, share, color=[CB[1] if v < 20 else CB[0] for v in share],
               height=0.55, label="(A) any gap, worst case")
    ax[2].plot([SCOPE_MEAS[t] for t in TASKS], y, "D", ms=3.5, color="0.2",
               label="(C) own gap, measured $c$")
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([])
    ax[2].invert_yaxis()
    ax[2].set_xlim(0, 108)
    ax[2].set_xlabel("% of proteome covered")
    ax[2].legend(frameon=False, fontsize=5.5, loc="lower left",
                 bbox_to_anchor=(0.0, -0.50), handlelength=1.2)
    ax[2].set_title("c   population supported", loc="left", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_identified.png")
    plt.close(fig)
    print("fig2_identified.png")


# ---------------------------------------------------------------- figure 3
def figure3():
    """Where shape restrictions help, and the resolution collapse."""
    path = find("sweep_T_20seed.tsv")
    if not path:
        print("fig3 skipped: sweep_T_20seed.tsv missing")
        return
    rows = read_tsv(path)
    by = defaultdict(list)
    for r in rows:
        by[int(r["threshold"])].append(r)
    Ts = sorted(by)

    def share(T, key):
        v = [float(r[key]) for r in by[T] if r[key] not in ("", "nan")]
        return float(np.mean(v)) if v else np.nan

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    # both are upper bounds, so they are comparable; the two-sided cond
    # interval is a different object and is not what the text quotes
    for key, lab, col in (("naive_valid", "endpoint, direction assumed", CB[1]),
                          ("lin_upper", "direction read from data", CB[0])):
        ax[0].plot(Ts, [100 * share(T, key) for T in Ts], "o-", ms=3.5, lw=1.2,
                   color=col, label=lab)
    ax[0].set_xlabel("donor threshold $T$")
    ax[0].set_ylabel("% of replicates the bound holds")
    ax[0].set_ylim(-4, 104)
    ax[0].set_title("a   validity", loc="left", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6, loc="lower center")

    ax[1].plot(Ts, [share(T, "w_unobs") for T in Ts], "o-", ms=3.5, lw=1.2,
               color="0.25")
    ax[1].axhline(0.5, color=CB[1], lw=0.8, ls=":")
    ax[1].set_xlabel("donor threshold $T$")
    ax[1].set_ylabel("unobserved weight $W$")
    ax[1].set_title("b   mass lost", loc="left", fontsize=8)

    nobs = [share(T, "n_obs") for T in Ts]
    ax[2].bar([str(T) for T in Ts], nobs, color=CB[0], width=0.6)
    ax[2].axhline(2, color=CB[1], lw=0.8, ls="--")
    ax[2].axhline(3, color=CB[3], lw=0.8, ls=":")
    ax[2].text(0.1, 2.15, "slope needs 2", color=CB[1], fontsize=5.5)
    ax[2].text(0.1, 3.15, "curvature needs 3", color=CB[3], fontsize=5.5)
    ax[2].set_xlabel("donor threshold $T$")
    ax[2].set_ylabel("strata the benchmark supplies")
    ax[2].set_title("c   resolution", loc="left", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_restrictions.png")
    plt.close(fig)
    print("fig3_restrictions.png")


# ---------------------------------------------------------------- figure 4
def figure4():
    """The rule, not the domain: propensity by degree, and the moving cut."""
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for i, (net, lab) in enumerate(NETS):
        for tag, style, a in (("cl3", "--", 0), ("cl3full", "-", 1)):
            p = find(f"clusterC/results_uniform/{tag}_{net}_by_degree.tsv",
                     f"cluster2/results/{tag}_{net}_by_degree.tsv",
                     f"{net.replace('string_phys_', 'string')}"
                     f"{'_full' if tag == 'cl3full' else ''}_by_degree.tsv")
            if not p:
                continue
            rows = read_tsv(p)
            x = np.arange(len(rows))
            y = [float(r["propensity"]) for r in rows]
            ax[a].plot(x, np.maximum(y, 1e-6), style, marker="o", ms=2.5,
                       lw=1.1, color=CB[i], label=lab if a == 1 else None)
            if a == 1:
                ax[a].set_xticks(x)
                ax[a].set_xticklabels([r["stratum"] for r in rows],
                                      rotation=45, fontsize=5.5)
            else:
                ax[a].set_xticks(x)
                ax[a].set_xticklabels([r["stratum"] for r in rows],
                                      rotation=45, fontsize=5.5)
    for a, ttl in ((0, "a   Contrastive-L3 alone"),
                   (1, "b   with the configuration-model prefilter")):
        ax[a].set_yscale("log")
        ax[a].set_ylim(5e-7, 2)
        ax[a].set_xlabel("protein degree")
        ax[a].set_title(ttl, loc="left", fontsize=8)
    ax[0].set_ylabel("propensity of appearing as a negative")
    ax[1].axhline(1e-6, color="0.4", lw=0.8, ls=":")
    ax[1].text(4.6, 1.5e-6, "exactly zero", fontsize=6, color="0.3")
    ax[1].legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_taxonomy.png")
    plt.close(fig)
    print("fig4_taxonomy.png")


# ---------------------------------------------------------------- figure 5
def figure5():
    """The consequence plane, and what the threshold was buying."""
    fig = plt.figure(figsize=(7.2, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
    axp = fig.add_subplot(gs[0, 0])

    for i, (net, lab) in enumerate(NETS):
        xs, ys = [], []
        for frac in ("1.0", "0.5", "0.2", "0.064"):
            p = find(f"clusterC/results_uniform/ppi_{net}_f{frac}.tsv")
            if not p:
                continue
            for r in read_tsv(p):
                if abs(float(r["r2"]) - 0.90) < 1e-9:
                    xs.append(float(r["w_unobs"]))
                    ys.append(float(r["delta_deploy"]) - float(r["delta_bench"]))
        if xs:
            o = np.argsort(xs)
            axp.plot(np.array(xs)[o], np.array(ys)[o], "o-", ms=3.5, lw=1.2,
                     color=CB[i], label=lab)
    axp.axhline(0, color="0.6", lw=0.6)
    axp.set_xlabel("unobserved weight $W$")
    axp.set_ylabel("deployment $-$ benchmark")
    axp.set_title("a   the benchmark's error", loc="left", fontsize=8)
    axp.legend(frameon=False, fontsize=6, loc="lower left")

    axc = fig.add_subplot(gs[0, 1])
    y = np.arange(len(TASKS))
    axc.barh(y, [PRC_RESID[t][0] / PRC_RESID[t][1] * 100 for t in TASKS],
             color=CB[1], height=0.6)
    axc.set_yticks(y)
    axc.set_yticklabels(TASKS, fontsize=5.5)
    axc.invert_yaxis()
    axc.set_xlabel("AUPRC residual, % of value")
    axc.set_title("b   decomposition fails\n     for AUPRC", loc="left",
                  fontsize=8)

    axt = fig.add_subplot(gs[0, 2])
    axt.plot(TRADE_T, TRADE_W, "o-", ms=3.5, lw=1.2, color=CB[1],
             label="$W$ (cost)")
    axt.plot(TRADE_T, TRADE_FN, "s-", ms=3.5, lw=1.2, color=CB[0],
             label="false-negative rate")
    axt.axhline(0.5, color="0.6", lw=0.8, ls=":")
    axt.axvline(1, color="0.4", lw=0.8, ls="--")
    axt.text(1.3, 0.62, "$T=1$ already\ncrosses $W=0.5$", fontsize=5.5)
    axt.set_xlabel("donor threshold $T$")
    axt.set_title("c   what the threshold buys", loc="left", fontsize=8)
    axt.legend(frameon=False, fontsize=6, loc="center right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_consequence.png")
    plt.close(fig)
    print("fig5_consequence.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print("\nwritten to %s/" % OUT)
