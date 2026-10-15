"""Run the i.i.d. simulation grid and write its table and figure:  python -m src.run_part3"""
import time

import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from . import evaluate as E
from . import theory as T
from .data import ROOT

TABLE = ROOT / "reports" / "tables" / "sim_iid_coverage.csv"
FIG = ROOT / "paper" / "figures"
COLORS = {"simple": "#D55E00", "mean_uncertainty": "#0072B2"}
LABELS = {"simple": "Simple bootstrap", "mean_uncertainty": "Mean-uncertainty bootstrap"}
MARKERS = {250: "o", 500: "s", 750: "^", 1000: "D"}


def hero_fig(df):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    x = np.linspace(0.02, 1.05, 200)
    for ax, level in zip(axes, C.LEVELS):
        ax.plot(x, T.coverage(level, x, 1), color=COLORS["simple"], lw=1.2, alpha=0.7)
        ax.axhline(level, color="k", ls="--", lw=0.8)
        for method in ("simple", "mean_uncertainty"):
            for n, g in df[(df.method == method) & (df.level == level)].groupby("n"):
                ax.errorbar(g.H_over_n, g.coverage, yerr=1.96 * g.se, fmt=MARKERS[n], ms=4, lw=0.8,
                            color=COLORS[method], mfc="white" if method == "mean_uncertainty" else COLORS[method])
        ax.set(title=f"Nominal {int(level * 100)}% band", xlabel="H / n")
    axes[0].set_ylabel("Coverage")
    handles = [plt.Line2D([], [], color=COLORS[m], marker="o", ls="", label=LABELS[m]) for m in COLORS]
    handles += [plt.Line2D([], [], color=COLORS["simple"], lw=1.2, label="Theory, simple bootstrap"),
                plt.Line2D([], [], color="k", ls="--", lw=0.8, label="Nominal level")]
    handles += [plt.Line2D([], [], color="grey", marker=m, ls="", label=f"n = {n}") for n, m in MARKERS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def hero_figure(df, path):
    fig = hero_fig(df)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def latex_outputs(df):
    """A table with every cell at both levels, and number macros for the text."""
    s = df[df.method == "simple"].set_index(["n", "H", "level"])
    m = df[df.method == "mean_uncertainty"].set_index(["n", "H", "level"])
    lines = [r"\newcommand{\simTable}{%", r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{95\% band} & \multicolumn{3}{c}{80\% band} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"$n$ & $H$ & Theory & Simple & Mean-unc. & Theory & Simple & Mean-unc. \\", r"\midrule"]
    for n in C.GRID_N:
        for H in C.GRID_H:
            row = f"{n} & {H}"
            for level in (0.95, 0.80):
                row += f" & {s.loc[(n, H, level), 'theory']:.3f} & {s.loc[(n, H, level), 'coverage']:.3f}"
                row += f" & {m.loc[(n, H, level), 'coverage']:.3f}"
            lines.append(row + r" \\")
        if n != C.GRID_N[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    simple = df[df.method == "simple"]
    simple = simple.assign(gap=(simple.coverage - simple.theory).abs())
    mu = df[df.method == "mean_uncertainty"].assign(gap=lambda t: (t.coverage - t.level).abs())
    cell = {lvl: s.loc[(250, 252, lvl)] for lvl in C.LEVELS}        # the cell with H / n close to 1
    z_s = ((simple.coverage - simple.theory) / simple.se).abs()
    z_m = ((mu.coverage - mu.level) / mu.se).abs()
    lines += [rf"\newcommand{{\simCells}}{{{len(simple)}}}",
              rf"\newcommand{{\simWithinTwoSeSimple}}{{{int((z_s <= 2).sum())}}}",
              rf"\newcommand{{\simWithinTwoSeMu}}{{{int((z_m <= 2).sum())}}}",
              rf"\newcommand{{\simOverTwoPoints}}{{{int((simple.gap > 0.02).sum())}}}",
              rf"\newcommand{{\simMaxWidthGap}}{{{(df[df.method == 'mean_uncertainty'].width_vs_simple - (1 + df[df.method == 'mean_uncertainty'].H_over_n) ** 0.5).abs().max():.3f}}}"]
    lines += [rf"\newcommand{{\simReps}}{{{int(df.reps.iloc[0]):,}}}".replace(",", "{,}"),
              rf"\newcommand{{\simPaths}}{{{C.B_SIM:,}}}".replace(",", "{,}"),
              rf"\newcommand{{\simMeanGapSimple}}{{{100 * abs((simple.coverage - simple.theory).mean()):.1f}}}",
              rf"\newcommand{{\simMaxGapSimple}}{{{100 * simple.gap.max():.1f}}}",
              rf"\newcommand{{\simMaxGapMu}}{{{100 * mu.gap.max():.1f}}}",
              rf"\newcommand{{\simMinCoverageSimple}}{{{100 * s.coverage.min():.0f}}}",
              rf"\newcommand{{\simCovHnnNinetyFive}}{{{100 * cell[0.95].coverage:.0f}}}",
              rf"\newcommand{{\simCovHnnEighty}}{{{100 * cell[0.80].coverage:.0f}}}"]
    (ROOT / "paper" / "tables" / "sim_iid.tex").write_text("\n".join(lines) + "\n")


def main(reps=None):
    plt.switch_backend("Agg")                  # scripts write files; notebooks keep their own inline backend
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = E.run_grid(reps=reps)
    df.to_csv(TABLE, index=False)
    hero_figure(df, FIG / "hero_coverage.pdf")
    latex_outputs(df)
    print(f"{len(df)} rows, {int(df.reps.iloc[0])} replications per cell, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
