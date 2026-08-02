r"""
fig_capture_cost.py - SCRIPT 6. What camera-in-the-loop costs, and how to cut it.

Every CITL iteration is one physical camera exposure. The iteration at which the
best reconstruction occurs is therefore not a convergence curiosity - it IS the
cost of the method, and it is the standing objection to camera-in-the-loop
optimization as a practical technique.

THE CLAIM IS A CONJUNCTION, so this script tests both halves:

    (a) simulation seeding reaches the best reconstruction in fewer exposures, AND
    (b) it reaches the same quality when it gets there.

Half (a) alone is worthless - any method can terminate early if you accept a
worse result. Half (b) is what makes the saving real. The two halves come from
different measurements (summary.log iteration counts; scored.json PSNR), so the
script consumes both and reports them together rather than letting a reader
assemble the conjunction themselves from two separate figures.

A CAUTION ON HALF (b). Absence of a significant difference is NOT evidence of
equivalence, and this script does not claim it is. It reports the paired median
difference in quality with its bootstrap interval, so a reader can judge whether
the interval excludes a difference they would consider meaningful. The honest
statement is "the quality difference is at most X dB", not "quality is
unchanged".

WITHIN-CONDITION. Iteration counts are not resolution- or normalization-
dependent, so they do not carry the cross-condition hazard that PSNR does.
Conditions are still reported separately, because the loss surfaces differ and
there is no question the comparison across them would answer.

Consumes summary.log via gdcitl.run_metadata, and optionally scored.json for
half (b). Emits a figure per forward model and tables/capture_cost.tex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import figstyle
import gdcitl
# The two condition labels, in ONE place. These used to be a hardcoded string
# literal inside draw(), which meant renaming the condition in the other scripts
# left this figure disagreeing with the manuscript.
LABELS = {"random": "Random init", "warm": "Simulation-seeded"}

from figstyle import C_RANDOM, C_WARM, C_INK, C_MUTED

MODELS = ("faithful", "ideal")     # faithful first: it is the reported condition


# --------------------------------------------------------------------------
def compute(physics: str, scored: list[dict] | None = None) -> dict:
    rand = gdcitl.run_metadata("random_init", physics)
    warm = gdcitl.run_metadata("warm_start", physics)
    targets = sorted(set(rand) & set(warm))
    if len(targets) != gdcitl.N_TARGETS:
        print(f"  ! {physics}: {len(targets)} paired targets, "
              f"expected {gdcitl.N_TARGETS}")

    r = np.array([rand[t]["best_iteration"] for t in targets], float)
    w = np.array([warm[t]["best_iteration"] for t in targets], float)
    ratios = r / np.maximum(w, 1.0)
    stat, p = gdcitl.wilcoxon(w, r)
    lo, hi = gdcitl.bootstrap_median_ci(ratios)

    spi = [v["seconds_per_iteration"] for v in list(rand.values()) + list(warm.values())
           if v["seconds_per_iteration"]]
    sec_per_iter = float(np.median(spi)) if spi else float("nan")

    d = {
        "physics": physics, "targets": targets, "random": r, "warm": w,
        "ratios": ratios,
        "med_random": float(np.median(r)), "med_warm": float(np.median(w)),
        "med_ratio": float(np.median(ratios)),
        "ratio_ci": (lo, hi),
        "ratio_min": float(ratios.min()), "ratio_max": float(ratios.max()),
        "wins": int((w < r).sum()), "n": len(targets), "W": stat, "p": p,
        "seconds_per_iteration": sec_per_iter,
        "minutes_random": float(np.median(r)) * sec_per_iter / 60.0,
        "minutes_warm": float(np.median(w)) * sec_per_iter / 60.0,
        "quality": None,
    }

    # - half (b): does the saving cost quality?
    if scored is not None:
        by = {}
        for rec in scored:
            if rec["physics"] == physics:
                by.setdefault(rec["source"], {})[rec["target"]] = rec
        a = by.get("bench_citl_random_init", {})
        b = by.get("bench_citl_warm_start", {})
        common = sorted(set(a) & set(b) & set(targets))
        if common:
            qa = np.array([a[t]["psnr"] for t in common], float)
            qb = np.array([b[t]["psnr"] for t in common], float)
            delta = qb - qa
            qlo, qhi = gdcitl.bootstrap_median_ci(delta)
            qstat, qp = gdcitl.wilcoxon(qb, qa)
            d["quality"] = {
                "n": len(common), "median": float(np.median(delta)),
                "ci_lo": qlo, "ci_hi": qhi, "p": qp,
                "median_random": float(np.median(qa)),
                "median_warm": float(np.median(qb)),
            }
    return d


# --------------------------------------------------------------------------
def draw(d: dict, out_dir: Path) -> None:
    r"""(a) paired slopes, (b) per-target reduction factor.

    A slope plot is the chosen form for a paired comparison at n=18: it shows
    every pair, the direction of every change, and the spread. A bar chart of two
    medians hides all three, and the fact that every line moves the same way is
    itself the result.

    Log y on (a) because the distribution spans an order of magnitude (63-703
    iterations under the faithful model) and is strongly right-skewed - on a
    linear axis the three slow targets would set the scale and compress the other
    fifteen into the floor.

    No numbers are drawn. Medians, ratios and intervals are in the table.

    READABILITY PASS. Five changes:
      1. EXPLICIT log ticks on (a). Matplotlib's default LogLocator put a single
         labelled major tick on the whole axis - "10^2" and nothing else - so
         the reader could see the lines move but could not read what they moved
         from or to. Ticks are now named decades and half-decades in plain
         numerals.
      2. Both reference lines in (b) are LABELLED in the axes. There were two
         vertical lines, unity and the median, differing only in colour and dash;
         nothing said which was which.
      3. The solid/dotted encoding in (a) is stated in the axes. Dotted meant
         "this target got worse"; that was recoverable only from the source.
      4. Target labels in (b) go 5.5 -> 6.5 pt and lose their underscores.
         5.5 pt was under Optica's 7-9 pt guidance before any page scaling, and
         serif at that size is worse than the sans it replaced.
      5. The x tick label on (a) was the string "Warm start", hardcoded here and
         NOT read from INITS - so renaming the condition elsewhere left this
         figure disagreeing with the manuscript. It now comes from a module
         constant, LABELS, alongside its partner.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib.lines import Line2D

    figstyle.apply()
    # AUTHORED AT THE TEXT WIDTH, not at the two-column width. Placed with
    # width=\linewidth, a 6.85in figure is scaled to the 5.25in text block --
    # 0.77x - and every font goes with it: the 8pt base landed at 6.1pt and the
    # legends at 4.2pt, under Optica's 7-9pt guidance. Authoring at TEXT_W means
    # the scale factor is 1.0 and the rcParams sizes are what reaches the page.
    # Taller than before because the type no longer shrinks to make room.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(figstyle.TEXT_W, 3.15),
        gridspec_kw={"width_ratios": [1.30, 1.0]})

    # - (a) paired slopes ---------------------------------------------------
    any_worse = False
    for ri, wi in zip(d["random"], d["warm"]):
        improved = wi < ri
        any_worse |= not improved
        ax1.plot([0, 1], [ri, wi], color=C_MUTED,
                 alpha=0.4 if improved else 0.9, linewidth=0.9,
                 linestyle="-" if improved else figstyle.LS_DOT,
                 zorder=2, solid_capstyle="round")
    ax1.scatter(np.zeros(d["n"]), d["random"], s=17, color=C_RANDOM,
                zorder=3, linewidths=0)
    ax1.scatter(np.ones(d["n"]), d["warm"], s=17, color=C_WARM,
                zorder=3, linewidths=0)
    ax1.plot([0, 1], [d["med_random"], d["med_warm"]], color=C_INK,
             linewidth=2.0, zorder=4, solid_capstyle="round")

    ax1.set_xlim(-0.3, 1.3)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([LABELS["random"], LABELS["warm"]])
    ax1.set_ylabel("Iteration of best reconstruction")
    ax1.set_yscale("log")

    # Named ticks. The default locator labels one major tick on this range,
    # which makes the axis decorative rather than readable.
    lo = min(d["random"].min(), d["warm"].min())
    hi = max(d["random"].max(), d["warm"].max())
    cand = [10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 1000]
    ticks = [t for t in cand if lo * 0.85 <= t <= hi * 1.15]
    ax1.yaxis.set_major_locator(FixedLocator(ticks))
    ax1.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax1.yaxis.set_minor_locator(FixedLocator([]))
    ax1.grid(axis="x", visible=False)
    ax1.set_title("(a)", loc="left", color=C_MUTED)

    # State the line encoding, ALWAYS - not only when a target got worse.
    #
    # It was conditional, so the faithful panel had no key at all while the
    # idealized one did: the same encoding documented in one figure and left
    # implicit in its pair. A reader comparing them cannot tell whether the
    # faithful panel has no dashed lines or simply does not label them. Showing
    # both keys states that no target needed more exposures, which is a result.
    #
    # Placed ABOVE the axes, sharing the row with the "(a)" title. Inside the
    # axes it sat on top of the longest traces, which are the ones a reader is
    # most likely to be looking at when they consult the key.
    ax1.legend(handles=[
        Line2D([], [], color=C_MUTED, lw=0.9, alpha=0.4, label="fewer exposures"),
        Line2D([], [], color=C_MUTED, lw=0.9, alpha=0.9,
               linestyle=figstyle.LS_DOT, label="more exposures"),
    ], loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0,
        frameon=False, handlelength=1.8, labelspacing=0.2, ncol=2,
        columnspacing=1.1)

    # - (b) per-target reduction factor -------------------------------------
    order = np.argsort(d["ratios"])
    y = np.arange(d["n"])
    ax2.scatter(d["ratios"][order], y, s=16, color=C_WARM, zorder=3, linewidths=0)
    # Unity = no saving. Everything to its right is an exposure saved.
    ax2.axvline(1.0, color=C_MUTED, linewidth=0.9, zorder=1)
    ax2.axvline(d["med_ratio"], color=C_INK, linewidth=1.0,
                linestyle=figstyle.LS_DASH, zorder=2)
    # Label both, in a legend at lower right. Rotated in-axes labels were tried
    # and clipped against the top spine; the lower right is empty because the
    # smallest ratios sort to the bottom, so the legend sits in real whitespace.
    ax2.legend(handles=[
        Line2D([], [], color=C_MUTED, lw=0.9, label="no saving"),
        Line2D([], [], color=C_INK, lw=1.0, linestyle=figstyle.LS_DASH,
               label="median"),
    ], loc="lower right", handlelength=1.8, labelspacing=0.2,
        borderaxespad=0.4)
    ax2.set_yticks(y)
    ax2.set_yticklabels([d["targets"][i].replace("_", " ") for i in order])
    ax2.set_xlabel("Reduction in iterations to best ($\\times$)")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("(b)", loc="left", color=C_MUTED)

    fig.tight_layout()
    figstyle.save(fig, str(out_dir / f"fig_capture_cost_{d['physics']}"))



# --------------------------------------------------------------------------
def table(results: list[dict], path: Path) -> None:
    r"""The capture-cost table.

    EVERY COLUMN STATES WHAT IT IS. An earlier version had bare headers
    "Random", "Warm", "Ratio" over numbers like 168.0 and 2.11, and nothing in
    the table said 168 of what, which direction the ratio ran, what Wins counted,
    or which way the dPSNR difference was signed. A reader had to reconstruct all
    four from the prose. The spanning header supplies the unit, and the caption
    defines the other three.

    WIDTH. optica-article.cls sets \textwidth = \columnwidth = 379.4pt: the
    submission format is SINGLE column. Measured widths for this table:

        9 columns, with the two exposure-time columns    465.7pt   over
        7 columns, header "Simulation-seeded"            384.1pt   over
        7 columns, header "Seeded"                       338.5pt   fits

    table* does not help - there is no second column to span - and
    \footnotesize on a two-row table is a worse trade than shortening a header.
    The two exposure-time columns were iterations multiplied by the measured
    per-exposure time, so they carried nothing the Ratio column does not, and
    the caption now states the two minute figures directly.

    The header says "Seeded" where prose and figures say "simulation-seeded".
    Column heads are abbreviations by convention; the caption defines it.
    """
    def f(v, n=2):
        return "--" if v is None or np.isnan(v) else f"{v:.{n}f}"

    def pfmt(p):
        return r"$<10^{-4}$" if p < 1e-4 else (r"$<0.001$" if p < 0.001 else f"{p:.3f}")

    rows = []
    for d in results:
        q = d["quality"]
        qcell = (f"{q['median']:+.2f} [{q['ci_lo']:+.2f}, {q['ci_hi']:+.2f}]"
                 if q else "--")
        rows.append(
            f"{d['physics'].capitalize()} & {f(d['med_random'], 1)} & "
            f"{f(d['med_warm'], 1)} & {f(d['med_ratio'])} & "
            f"{d['wins']}/{d['n']} & {pfmt(d['p'])} & {qcell} \\\\")

    # Exposure-time figures for the reported condition, quoted in the caption
    # since the columns that used to carry them are gone.
    rep = next((d for d in results if d["physics"] == "faithful"), results[0])

    lines = [
        r"% GENERATED by analysis/fig_capture_cost.py - do not hand-edit.",
        r"% Destination: main text.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Capture cost. Every iteration of camera-in-the-loop "
        r"optimization costs one physical camera exposure. The iteration at "
        r"which the best reconstruction occurs is therefore the cost of the "
        r"method. \emph{Random} and \emph{Seeded} are the two initializations "
        r"of Section~\ref{sec:citl}. Random draws the phase uniformly. Seeded "
        r"takes the phase of a converged simulation-only run. \emph{Ratio} is "
        r"random divided by seeded, so a value above one means seeding needed "
        r"fewer exposures. \emph{Wins} counts the targets for which seeding "
        r"reached its best reconstruction in fewer iterations. The final column "
        r"is the paired difference in final reconstruction quality, seeded "
        r"minus random, so a positive value means seeding scored higher. It "
        r"carries a \SI{95}{\percent} interval from \num{10000} bootstrap "
        r"resamples and bounds what the saving costs in fidelity. All values "
        r"are medians across \num{18} paired targets. The distribution is "
        r"strongly right-skewed so medians are reported in preference to means. "
        rf"At the measured \SI{{{rep['seconds_per_iteration']:.2f}}}{{\second}} "
        r"per exposure the faithful-model medians correspond to "
        rf"\num{{{rep['minutes_random']:.1f}}} and "
        rf"\SI{{{rep['minutes_warm']:.1f}}}{{\minute}} of acquisition per "
        r"target. $p$ is from a Wilcoxon signed-rank test on the iteration "
        r"counts.}",
        r"\label{tab:capture}",
        r"\begin{tabular}{l S[table-format=3.1] S[table-format=3.1] "
        r"S[table-format=1.2] c l c}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Iterations to best} & & & & \\",
        r"\cmidrule(lr){2-3}",
        r"Model & {Random} & {Seeded} & {Ratio} & "
        r"{Wins} & {$p$} & {$\Delta$PSNR (dB)} \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  table -> {path}")


def report(results: list[dict]) -> None:
    for d in results:
        print(f"\n{d['physics'].upper()}  (n={d['n']})")
        print(f"  median best iteration : {d['med_random']:.1f} -> {d['med_warm']:.1f}")
        print(f"  median reduction      : {d['med_ratio']:.2f}x "
              f"[{d['ratio_ci'][0]:.2f}, {d['ratio_ci'][1]:.2f}]  "
              f"(range {d['ratio_min']:.2f}-{d['ratio_max']:.2f})")
        print(f"  seeded sooner in      : {d['wins']}/{d['n']}   p={d['p']:.3g}")
        print(f"  exposure time         : {d['minutes_random']:.1f} -> "
              f"{d['minutes_warm']:.1f} min  "
              f"(at {d['seconds_per_iteration']:.2f} s/iteration)")
        q = d["quality"]
        if q:
            print(f"  quality cost          : {q['median']:+.2f} dB "
                  f"[{q['ci_lo']:+.2f}, {q['ci_hi']:+.2f}]  p={q['p']:.3g}")
            print(f"                          (warm {q['median_warm']:.2f} vs "
                  f"random {q['median_random']:.2f} dB)")
        else:
            print("  quality cost          : not evaluated "
                  "(pass scored_path to test half (b))")
    print("\nNote: a non-significant quality difference is not evidence of "
          "equivalence.\n      Read the interval, not the p-value.")


def run(out_dir, tables_dir, scored_path=None, figure: bool = True) -> list[dict]:
    out_dir, tables_dir = Path(out_dir), Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = json.loads(Path(scored_path).read_text()) if scored_path else None

    results = [compute(p, scored) for p in MODELS]
    report(results)
    table(results, tables_dir / "capture_cost.tex")
    if figure:
        for d in results:
            draw(d, out_dir)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--tables", type=Path, default=Path("tables"))
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    run(a.out, a.tables, a.scored, figure=not a.no_figure)


if __name__ == "__main__":
    main()
