"""Regenerate every figure in the paper and the technical report from results/paper_results.json."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = [os.path.join(ROOT, "figures"),
       os.path.join(ROOT, "paper", "figures"),
       os.path.join(ROOT, "technical_report", "figures")]

plt.rcParams.update({
    # Emit embedded TrueType fonts in PDF figures; avoids Type-3 figure fonts.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

COL = {"B2": "#1f77b4", "B7": "#ff7f0e", "B6": "#2ca02c",
       "B6c": "#d62728", "B8": "#7f7f7f", "B8c": "#9467bd"}
LAB = {"B2": "B2 confidence", "B7": "B7 scalar cascade", "B6": "B6 full (7 coord.)",
       "B6c": "B6c compact (4 coord.)", "B8": "B8 union (full)",
       "B8c": "B8c union (compact)"}


def save(fig, stem):
    for d in OUT:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, stem + ".pdf"))
        fig.savefig(os.path.join(d, stem + ".png"))
    plt.close(fig)


def _roc(a, ax, fam, title, legend, adv=None):
    if adv is not None:
        e = adv["probe"]["by_family"][f"f{fam}"]
        ax.plot(e["obs"]["ni_grid"], e["obs"]["rejection"], "--", lw=1.1,
                color="#ff7f0e", alpha=0.85, label="supervised probe on $o$")
        ax.plot(e["struct"]["ni_grid"], e["struct"]["rejection"], "--", lw=1.1,
                color="#2ca02c", alpha=0.85, label="supervised probe on structural")
    for pol in ("B2", "B7", "B6", "B6c", "B8c"):
        pts = sorted(a["frontier"][pol], key=lambda p: p["NomInt"])
        x = [p["NomInt"] for p in pts]
        y = [1.0 - p[f"Cov_f{fam}"] for p in pts]
        ax.plot(x, y, "-o", ms=2.2, lw=1.2, color=COL[pol], label=LAB[pol])
    ax.axvline(0.10, color="0.4", lw=0.8, ls=":")
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Nominal intervention")
    ax.set_title(title, pad=3)
    if legend:
        ax.legend(loc="lower right", frameon=False, handlelength=1.1, fontsize=6.4, ncol=1)


def fig_evasion(adv):
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    order = [("B7", "#ff7f0e"), ("B6c", "#d62728"), ("B8c", "#9467bd")]
    for k, c in order:
        vs = adv["evasion"][k]
        xs = [0.0] + [v["eps"] for v in vs]
        ys = [vs[0]["evaded_clean"]] + [v["evaded_attacked"] for v in vs]
        ax.plot(xs, ys, "-o", ms=3, lw=1.2, color=c, label=LAB[k])
    ax.set_xlabel(r"Perturbation budget $\varepsilon$")
    ax.set_ylabel("Evasion rate")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", frameon=False, handlelength=1.3, fontsize=7)
    save(fig, "fig1d_evasion")


def fig_frontier(a, adv=None):
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    _roc(a, ax, 7, "Correlated residual", True, adv=adv)
    ax.set_ylabel("Mechanism rejection rate")
    save(fig, "fig1a_roc_f7")

    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    _roc(a, ax, 6, "Coordinated spoof", False, adv=adv)
    ax.set_ylabel("Mechanism rejection rate")
    save(fig, "fig1b_roc_f6")

    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    for pol in ("B2", "B7", "B6", "B6c", "B8c"):
        pts = sorted(a["frontier"][pol], key=lambda p: p["Cov"])
        ax.plot([p["Cov"] for p in pts], [p["HAR"] for p in pts],
                "-o", ms=2.5, lw=1.2, color=COL[pol], label=LAB[pol])
    ax.set_xlabel("Risk coverage")
    ax.set_ylabel("HAR")
    ax.set_xlim(0, 0.42)
    ax.set_ylim(0, 0.20)
    ax.legend(loc="upper left", frameon=False, handlelength=1.3, fontsize=7)
    save(fig, "fig1c_risk_coverage")


NAMES = {"gf": "Fused purity", "C": "Centred alignment", "J": "Bipartite score",
         "S": "Entropy", "gm": "Marginal purity", "Dv": "Relative entropy",
         "F": "Fidelity"}


def fig_ablation(a):
    items = sorted(a["ablation"].items(), key=lambda kv: kv[1]["increase"])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.barh([NAMES[k] for k, _ in items], [v["increase"] for _, v in items],
            color=["#d62728" if v["increase"] > 0.005 else "#9ecae1"
                   for _, v in items])
    ax.set_xlabel("HAR increase over full B6")
    ax.grid(axis="y", alpha=0)
    save(fig, "fig1c_gate_removal")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 1.7))
    ax.axis("off")
    boxes = [("seed", 0.02), ("8 families\n1000 rec. each", 0.15),
             ("stratified\n30/15/55 split", 0.31), ("frozen upstream\nemulator + T_cal", 0.47),
             ("density-state\nencoder + gates", 0.63), ("authority cascade\nB0-B8", 0.79),
             ("HAR / Cov / SR\nNomInt", 0.93)]
    for text, x in boxes:
        ax.add_patch(plt.Rectangle((x, 0.32), 0.115, 0.38, fill=True,
                                   facecolor="#eef3f8", edgecolor="#33628f", lw=1.0,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.0575, 0.51, text, ha="center", va="center", fontsize=7.2,
                transform=ax.transAxes)
    for _, x in boxes[:-1]:
        ax.annotate("", xy=(x + 0.128, 0.51), xytext=(x + 0.117, 0.51),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#33628f"))
    ax.text(0.5, 0.10, "envelope, moments and gate thresholds are fitted on nominal "
                       "TRAINING records only; policies are frozen before evaluation",
            ha="center", fontsize=7, transform=ax.transAxes, style="italic")
    save(fig, "clw_pipeline")


def main():
    with open(os.path.join(ROOT, "results", "paper_results.json")) as fh:
        a = json.load(fh)
    advp = os.path.join(ROOT, "results", "adversarial_results.json")
    adv = json.load(open(advp)) if os.path.exists(advp) else None
    fig_frontier(a, adv)
    fig_ablation(a)
    fig_pipeline()
    if adv is not None:
        fig_evasion(adv)
    print("figures written")


if __name__ == "__main__":
    sys.exit(main())
