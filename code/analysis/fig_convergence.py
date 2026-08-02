r"""
fig_convergence.py - SCRIPT 6. Per-iteration convergence on the physical system.

WHY THIS FIGURE EXISTS.

  Tables 2 and 3 report where each run ENDED and how many exposures it took to
  get there. Neither shows the shape of the descent, and the shape is what the
  capture-cost claim actually rests on: simulation seeding is asserted to buy
  convergence speed rather than final quality. That is a statement about
  trajectories, and only a trajectory plot can support or refute it.

WHAT IS PLOTTED, AND WHY IT IS NOT PSNR.

  The per-iteration quantity that exists is the TRAINING LOSS - the DC-masked
  mean squared error evaluated on the camera measurement, i.e. the quantity the
  optimizer is actually minimizing. Per-iteration PSNR does not exist and cannot
  be recovered: each run retained only three captures (iteration 0, the best
  iteration, and the last), so there is no reconstruction to score at iteration
  k. The caption must say this. A reader who assumes the y-axis is image quality
  will misread the whole figure.

NO NORMALIZATION. Loss is plotted raw, on a log axis.

  Targets differ in achievable MSE - a dense photograph and a binary chart do
  not converge to the same floor - so the traces do not collapse onto one
  another and the band is correspondingly wide. That is reported rather than
  removed: rescaling each target to a common start would hide both the spread in
  achievable loss and the seeded run's head start, and the head start is the
  point. The log axis carries the between-target range; the median and
  interquartile band describe the ensemble at each iteration.

BAND CHOICE. Interquartile range across the 18 targets, not a bootstrap CI.
  This figure is descriptive - it shows what the runs did. The inferential
  claims are made in Tables 2 and 3 with paired tests. An inferential band here
  would imply a per-iteration hypothesis test that is not being performed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator

import figstyle
import gdcitl
from figstyle import C_RANDOM, C_WARM, C_MUTED, C_GRID, C_INK

FIG_W = getattr(figstyle, "TEXT_W", 5.25)

# TYPE SIZES, in one place, and all at or above Optica's 7pt floor.
#
# This figure is authored at the text width, so nothing is scaled and these are
# the sizes that reach the page - which is why they can be stated plainly here
# rather than derived through figstyle.title_size(). They were 7/8/6: the panel
# titles sat at the floor and the legend a point under it. Four panels of dense
# traces need type at the upper end of the range, not the lower.
FS_TITLE = 8.0
FS_AXLABEL = 9.0
FS_TICK = 8.0
FS_LEGEND = 7.5

INITS = (("random_init", "Random init", C_RANDOM),
         ("warm_start", "Simulation-seeded", C_WARM))
# Taken from gdcitl, not restated here. gdcitl declares these under its
# Terminology heading as the canonical labels, but nothing imported them, so
# this file kept its own copy and the two drifted: the deposit said "Ideal
# model" while the working repository said "Idealized model", and the figures
# disagreed across trees. A constant that is copied is a constant that will
# diverge.
MODELS = (("faithful", gdcitl.FAITHFUL_LABEL),
          ("ideal", gdcitl.IDEAL_LABEL))

CSV_NAME = "gd_camera_feedback_convergence.csv"


# --------------------------------------------------------------------------
def _read_trace(run_dir: Path) -> np.ndarray | None:
    """Per-iteration training loss for one run, ordered by iteration."""
    path = Path(run_dir) / CSV_NAME
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["iteration"]))
    return np.array([float(r["loss"]) for r in rows], dtype=float)


def compute(root=None) -> dict:
    out = {}
    for physics, _ in MODELS:
        runs = {init: {r["target"]: r for r in gdcitl.citl_runs(init, physics, root)}
                for init, _, _ in INITS}

        # Targets without BOTH initializations are dropped, so every panel in a
        # row is built from the same target set and the rows stay comparable.
        targets = sorted(set(runs["random_init"]) & set(runs["warm_start"]))
        traces, dropped = {init: {} for init, _, _ in INITS}, []
        for t in targets:
            per_init = {}
            for init, _, _ in INITS:
                tr = _read_trace(runs[init][t]["run_dir"])
                if tr is None:
                    break
                per_init[init] = tr
            else:
                for init, tr in per_init.items():
                    traces[init][t] = tr
                continue
            dropped.append(t)

        if dropped:
            print(f"  note: {physics}: no convergence log for {', '.join(dropped)}")

        out[physics] = {
            init: {"traces": traces[init],
                   "best": [runs[init][t]["best_iteration"] for t in traces[init]]}
            for init, _, _ in INITS}
    return out


def _span_under_a_decade(summ: dict) -> bool:
    """True when every panel in this row fits inside one power of ten."""
    vals = [v for s in summ.values() for v in (s[2].min(), s[3].max()) if v > 0]
    return bool(vals) and max(vals) / min(vals) < 10.0


def _stack(traces: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and interquartile band over targets, truncated to the shortest run."""
    n = min(len(v) for v in traces.values())
    m = np.vstack([v[:n] for v in traces.values()])
    x = np.arange(n)
    return x, np.median(m, axis=0), np.percentile(m, 25, axis=0), np.percentile(m, 75, axis=0)


# --------------------------------------------------------------------------
def draw(d: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.apply()
    # Taller than 3.6 to carry the larger type without crowding the traces.
    fig, axes = plt.subplots(len(MODELS), len(INITS), figsize=(FIG_W, 4.1),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for i, (physics, model_label) in enumerate(MODELS):
        summ = {init: _stack(d[physics][init]["traces"])
                for init, _, _ in INITS if d[physics][init]["traces"]}
        # Bound now, not at draw time: a lambda closing over `summ` would see
        # whichever row the loop finished on, not this one.
        label_minor = _span_under_a_decade(summ)
        for j, (init, init_label, colour) in enumerate(INITS):
            ax = axes[i, j]
            if init not in summ:
                ax.set_visible(False)
                continue
            x, med, lo, hi = summ[init]

            for tr in d[physics][init]["traces"].values():
                ax.plot(np.arange(len(tr)), tr, color=colour, linewidth=0.3,
                        alpha=0.13, zorder=2)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.20, linewidth=0, zorder=3)
            # White halo under the median. Eighteen traces plus a band is enough
            # ink that a plain line of the same hue disappears into its own data.
            ax.plot(x, med, color="white", linewidth=2.8, solid_capstyle="round",
                    zorder=5)
            ax.plot(x, med, color=colour, linewidth=1.6, solid_capstyle="round",
                    zorder=6)

            best = d[physics][init]["best"]
            if best:
                ax.axvline(float(np.median(best)), color=C_INK, linewidth=0.8,
                           linestyle=figstyle.LS_DASH, alpha=0.55, zorder=4)

            # Every run does its work in the first few dozen exposures and then
            # sits flat. On a linear axis nine tenths of the width shows nothing,
            # and the part that matters is compressed against the origin. symlog
            # keeps iteration 0 - the seed's own loss, which is the whole point
            # of the seeded panels - while giving the early decay real estate.
            ax.set_xscale("symlog", linthresh=1.0, linscale=0.4)
            ax.set_xticks([0, 1, 3, 10, 30, 100, 300, 750])
            ax.set_xticklabels(["0", "1", "3", "10", "30", "100", "300", "750"])
            ax.set_xlim(0, 750)
            ax.set_yscale("log")
            # A log axis spanning less than a decade labels a single tick and is
            # unreadable. Label the 2/3/5 subdivisions too when that happens.
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0),
                                                  numticks=12))
            ax.yaxis.set_minor_formatter(FuncFormatter(
                lambda v, _pos, on=label_minor: f"{v:g}" if on else ""))
            # y only. The x gridlines fell on the same symlog ticks and competed
            # with the best-iteration marker, which is the vertical that carries
            # information.
            ax.grid(True, axis="y", which="major", color=C_GRID, linewidth=0.5,
                    zorder=0)
            ax.set_title(f"{model_label} - {init_label}", loc="left",
                         fontsize=FS_TITLE, color=C_INK)
            ax.tick_params(axis="both", which="both", labelsize=FS_TICK)

    # One legend, in the first panel only. Five elements repeat across all four
    # panels in two hues, so the proxies are drawn in ink rather than in either
    # series colour.
    handles = [
            Line2D([], [], color=C_INK, linewidth=1.6, label="median over targets"),
            Patch(facecolor=C_INK, alpha=0.20, label="interquartile range"),
            Line2D([], [], color=C_INK, linewidth=0.4, alpha=0.35,
                   label="individual target"),
            Line2D([], [], color=C_INK, linewidth=0.8, linestyle=figstyle.LS_DASH,
                   alpha=0.55, label="median best iteration"),
    ]

    # One shared label per axis. Per-axes labels collide under sharey once the
    # y-label needs two lines, and repeating them four times says nothing extra.
    fig.supylabel("Training loss (DC-masked MSE on the measurement)",
                  fontsize=FS_AXLABEL)

    # Legend and x-label live below the panels. Inside any panel the legend
    # either sits on the median or on the reference curve, and which one it
    # covers changes with the data.
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    fig.supxlabel("Iteration (one camera exposure each)", fontsize=FS_AXLABEL,
                  y=0.105)
    # Two columns, not three: four entries across three columns strand the
    # fourth on a row of its own, and the labels are too long to fit on one.
    fig.legend(handles=handles, fontsize=FS_LEGEND, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.005),
               handlelength=2.4, columnspacing=1.6, labelspacing=0.3)
    figstyle.save(fig, str(out_dir / "fig_convergence"))


def report(d: dict) -> None:
    for physics, label in MODELS:
        print(f"\n{label.upper()}")
        for init, init_label, _ in INITS:
            tr = d[physics][init]["traces"]
            if not tr:
                print(f"  {init_label:20s} no traces")
                continue
            x, med, lo, hi = _stack(tr)
            print(f"  {init_label:20s} n={len(tr):2d}  median loss at iter 0 "
                  f"{med[0]:.4g}  at {len(med)-1} {med[-1]:.4g}  "
                  f"median best iteration {np.median(d[physics][init]['best']):.0f}")
            # A run whose loss never moved is a dead run, not a converged one,
            # and on a three-decade log axis the two look identical. Name them.
            flat = [t for t, v in tr.items()
                    if v[0] > 0 and abs(v.min() - v[0]) / v[0] < 0.01]
            if flat:
                print(f"    ! loss never moved by 1%: {', '.join(sorted(flat))}")


def run(out_dir, root=None, figure: bool = True) -> dict:
    """Notebook entry point: fig_convergence.run(FIGS, RESULTS)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = compute(root)
    report(d)
    if figure:
        draw(d, out_dir)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    run(a.out, a.root, figure=not a.no_figure)


if __name__ == "__main__":
    main()