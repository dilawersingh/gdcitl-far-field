r"""
fig_citl_vs_plain.py - SCRIPT 4. GD+CITL versus plain GD on hardware.

The result the paper stands on. For each forward model, the same 18 targets are
photographed twice on the same bench: once displaying the hologram gradient
descent produced in simulation, once displaying the hologram gradient descent
produced with the camera in the loop. The difference between those two captures
is what camera feedback buys.

WITHIN-CONDITION ONLY - the methodological rule this script exists to enforce.

  Every comparison here holds the forward model FIXED and varies only the
  hologram. Within a condition, both captures are scored at the same resolution,
  against the same target preparation, with the same zero-order handling, the same
  calibration and the same normalization. The only thing that differs is the
  hologram, so the difference is attributable to it.

  ACROSS conditions nothing is comparable, and this script never compares them:
  ideal scores a native 1000x1000 target, faithful scores that target bicubic-
  upsampled to 2000x2000. Different histogram, hence different min-max
  normalization; different resolution, hence a different SSIM window relative to
  image content. An ideal PSNR and a faithful PSNR are answers to two different
  scoring problems and must never be differenced, plotted on a shared axis, or
  described as one being higher than the other.

  Ideal vs faithful is therefore a ROBUSTNESS AXIS, not a result: the CITL
  advantage is measured twice, independently, and the question is whether it
  survives both. That is what answers the fairness objection - GD has no
  corrector, so its only defense against real optical non-idealities is the
  accuracy of its own internal model, and a sceptic is entitled to ask whether
  the advantage is an artefact of an unrealistic one.

Four primary comparisons: {ideal, faithful} x {random init, simulation-seeded},
each
paired over 18 targets. Benjamini-Hochberg across those four.

Consumes scored.json from script 2. Emits the figure, tables/citl.tex (main
text) and tables/citl_metrics.tex (supplement: the same four comparisons under
every metric in scored.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import figstyle
import gdcitl
from figstyle import C_RANDOM, C_WARM, C_INK, C_MUTED

MODELS = ("faithful", "ideal")   # faithful first: it is the reported condition
# Display labels only; the KEYS index scored.json's source field and must not
# change. "Simulation-seeded" replaces "Warm start" - warm start conventionally
# implies reusing optimizer state, which this deliberately does not do, and the
# point of the seed is that it costs no camera exposures, which the new name
# states and the old one hid.
INITS = (("random_init", "Random init", C_RANDOM),
         ("warm_start", "Simulation-seeded", C_WARM))
METRIC = "psnr"
TABLE_METRICS = ("psnr", "ssim", "nmse", "diffraction_efficiency")


# --------------------------------------------------------------------------
def compute(scored: list[dict]) -> dict:
    """Paired statistics for all four comparisons. No cross-condition arithmetic
    happens anywhere in this function - see the module docstring."""
    by = {}
    for r in scored:
        by.setdefault((r["physics"], r["source"]), {})[r["target"]] = r

    comparisons, pvals, keys = {}, [], []

    for physics in MODELS:
        plain = by.get((physics, "bench_plain_gd"), {})
        for init, label, _ in INITS:
            citl = by.get((physics, f"bench_citl_{init}"), {})
            targets = sorted(set(plain) & set(citl))
            if len(targets) != gdcitl.N_TARGETS:
                print(f"  ! {physics}/{init}: {len(targets)} paired targets, "
                      f"expected {gdcitl.N_TARGETS}")
            if not targets:
                continue

            p_vals = np.array([plain[t][METRIC] for t in targets], float)
            c_vals = np.array([citl[t][METRIC] for t in targets], float)
            delta = c_vals - p_vals
            stat, p = gdcitl.wilcoxon(c_vals, p_vals)
            lo, hi = gdcitl.bootstrap_median_ci(delta)

            key = (physics, init)
            comparisons[key] = {
                "physics": physics, "init": init, "label": label,
                "targets": targets, "plain": p_vals, "citl": c_vals,
                "delta": delta, "median": float(np.median(delta)),
                "ci_lo": lo, "ci_hi": hi,
                "wins": int((delta > 0).sum()), "n": len(targets),
                "W": stat, "p": p,
                "plain_stats": gdcitl.summarize(p_vals),
                "citl_stats": gdcitl.summarize(c_vals),
                # all metrics, for the table
                "metrics": {
                    m: {"plain": gdcitl.summarize([plain[t][m] for t in targets]),
                        "citl": gdcitl.summarize([citl[t][m] for t in targets])}
                    for m in TABLE_METRICS
                },
            }
            pvals.append(p)
            keys.append(key)

    for key, sig in zip(keys, gdcitl.benjamini_hochberg(pvals)):
        comparisons[key]["significant"] = bool(sig)

    return {"comparisons": comparisons, "order": keys}


# --------------------------------------------------------------------------
def draw(d: dict, out_dir: Path) -> None:
    r"""Per-target paired differences, ONE FIGURE PER FORWARD MODEL.

    Form: differences, not absolute levels. The claim is "CITL beats plain GD",
    which is a statement about the sign of a paired difference, so the figure
    plots that difference against a zero reference rather than two columns of
    absolute PSNR the reader has to subtract by eye.

    ONE FILE PER MODEL, and that is the point. The previous version put all four
    comparisons on a single shared x-axis. Plotting differences does remove the
    worst version of the cross-condition trap - a difference is at least not an
    absolute level - but it does not license the shared axis, because a dPSNR
    measured at 1000x1000 and a dPSNR measured at 2000x2000 are differences of
    quantities that were min-max normalized differently. A reader WILL compare
    the rows, and "ideal gains more than faithful" is not a supported reading.
    The table caption says values are not comparable between blocks; the figure
    used to invite exactly that comparison. Separate files, separate axes, and
    the main text carries faithful only.

    ONE ROW PER TARGET, sorted. This is the substantive change. The old form was
    one row per comparison with 18 anonymous jittered dots, which showed the
    sign result and nothing else. Sorting by target and labelling makes three
    claims legible at once:
      1. the sign result   - everything right of zero but one target;
      2. content dependence - the ordering itself, which the prose otherwise
                               has to spend six enumerated numbers on;
      3. initialization parity - the two dots per row nearly coincide, which is
                               the same claim Section 4.4 makes from the medians.

    NO CATEGORY BRACKETS, and no content descriptors in the tick labels. The
    ordering is real but the "dense natural at the top, sparse and synthetic at
    the bottom" reading has genuine exceptions in this target set - a bright sun
    on a dark sky ranks first, the densest urban scene ranks tenth, and a sparse
    dot pattern ranks fifth. Only the BOTTOM of the distribution is clean.
    Labelling by content, or bracketing the two groups, would make the figure
    assert a partition the Discussion explicitly declines to test. Filenames
    keep the figure honest and are the traceability key into Supplement 1.

    NO POOLED MEDIAN LINE. An earlier draft drew the median of all 36 values
    across both initializations. That is not a quantity reported anywhere in the
    paper. The real paired medians are in tables/citl.tex.
    """
    import matplotlib.pyplot as plt

    for physics in MODELS:
        keys = [k for k in d["order"] if k[0] == physics]
        if len(keys) != len(INITS):
            print(f"  ! {physics}: {len(keys)} comparisons, expected {len(INITS)}")
            continue
        _draw_one(d, physics, keys, out_dir)


def _draw_one(d: dict, physics: str, keys: list, out_dir: Path) -> None:
    """One forward model, one file. Targets on y, sorted by mean improvement."""
    import matplotlib.pyplot as plt

    comps = d["comparisons"]
    targets = comps[keys[0]]["targets"]
    series = [(comps[k]["label"],
               dict(random_init=C_RANDOM, warm_start=C_WARM)[comps[k]["init"]],
               comps[k]["delta"]) for k in keys]

    # Sort by the mean across initializations so neither series dictates the
    # order. Ascending, because matplotlib's y axis runs upward and the largest
    # improvement should sit at the top.
    mean_delta = np.mean([s[2] for s in series], axis=0)
    order = np.argsort(mean_delta)

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(figstyle.COL_W, 3.55))
    ypos = np.arange(len(targets))

    # Connector: the within-target spread between initializations. Recessive -
    # it is context, not a series, so it never competes with the marks.
    stack = np.array([s[2] for s in series])
    ax.hlines(ypos, stack.min(axis=0)[order], stack.max(axis=0)[order],
              color=figstyle.C_GRID, linewidth=1.4, zorder=1)

    for label, colour, delta in series:
        ax.scatter(delta[order], ypos, s=17, color=colour, linewidths=0,
                   zorder=3, label=label)

    # Zero = camera feedback made no difference. The whole claim is distance
    # from this line, so it is drawn over the grid and never cropped out.
    ax.axvline(0.0, color=C_MUTED, linewidth=1.0, zorder=2)

    ax.set_yticks(ypos)
    ax.set_yticklabels([targets[i] for i in order], fontsize=6)
    ax.set_ylim(-0.7, len(targets) - 0.3)
    ax.set_xlabel(r"$\Delta$PSNR, GD+CITL $-$ plain GD (dB)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.4,
              borderaxespad=0.4, labelspacing=0.25, markerscale=1.1)

    fig.tight_layout(pad=0.3)
    figstyle.save(fig, str(out_dir / f"fig_citl_vs_plain_{physics}"))



# --------------------------------------------------------------------------
def table(d: dict, path: Path) -> None:
    comps, order = d["comparisons"], d["order"]

    def m(v, n=2):
        return "--" if v is None or np.isnan(v) else f"{v:.{n}f}"

    rows = []
    for i, physics in enumerate(MODELS):
        keys = [k for k in order if k[0] == physics]
        if not keys:
            continue
        # Absolute levels are quoted WITHIN a block and never compared across
        # blocks - the \midrule between models is doing methodological work,
        # not just typographic.
        plain = comps[keys[0]]["metrics"]
        rows.append(
            f"{physics.capitalize()} & Plain GD & "
            f"{m(plain['psnr']['plain']['median'])} & "
            f"{m(plain['ssim']['plain']['median'], 3)} & "
            f"{m(plain['nmse']['plain']['median'], 3)} & & \\\\")
        for k in keys:
            c = comps[k]
            dag = r"$^\dagger$" if c.get("significant") else ""
            rows.append(
                f" & GD+CITL, {c['label'].lower()} & "
                f"{m(c['metrics']['psnr']['citl']['median'])} & "
                f"{m(c['metrics']['ssim']['citl']['median'], 3)} & "
                f"{m(c['metrics']['nmse']['citl']['median'], 3)} & "
                f"{m(c['median'], 2)} [{m(c['ci_lo'])}, {m(c['ci_hi'])}] & "
                f"{c['wins']}/{c['n']}{dag} \\\\")
        if i < len(MODELS) - 1:
            rows.append(r"\midrule")

    lines = [
        r"% GENERATED by analysis/fig_citl_vs_plain.py - do not hand-edit.",
        r"% Destination: main text.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{GD+CITL against plain GD on the physical system. All "
        r"values are medians across \num{18} paired targets. Arrows mark the "
        r"direction of improvement. $\Delta$PSNR is the median paired "
        r"difference, GD+CITL minus plain GD, so a positive value means camera "
        r"feedback scored higher. It carries a \SI{95}{\percent} interval from \num{10000} "
        r"bootstrap resamples. \emph{Wins} counts the targets improved by "
        r"camera feedback out of \num{18}. The plain GD row is "
        r"the baseline the other two are measured against, so it has no "
        r"$\Delta$PSNR of its own. $\dagger$ marks survival of "
        r"Benjamini-Hochberg correction at $q=0.05$ across the four "
        r"comparisons. Each block is a self-contained comparison at a single "
        r"forward model. Within a block both holograms are scored at the same "
        r"resolution against the same target preparation, so only the hologram "
        r"differs. Values are not comparable between blocks. The ideal "
        r"condition scores a native \num{1000}$\times$\num{1000} target and "
        r"the faithful condition scores that target upsampled to "
        r"\num{2000}$\times$\num{2000}. The two blocks answer whether the "
        r"advantage survives both internal models. They do not answer which "
        r"model scores higher.}",
        r"\label{tab:citl}",
        # \footnotesize is required, not cosmetic. optica-article.cls sets
        # \textwidth = \columnwidth = 379.4pt (the submission format is SINGLE
        # column). Measured: 425.9pt at normal size without arrows, 442.0pt with
        # them, 372.1pt with arrows at \footnotesize. table* does not help --
        # there is no second column to span.
        r"\footnotesize",
        r"\begin{tabular}{l l c c c c c}", r"\toprule",
        # Arrows give SSIM and NMSE a direction. NMSE is the one readers invert.
        r"Model & Hologram & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & "
        r"NMSE $\downarrow$ & $\Delta$PSNR (dB) & Wins \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  table -> {path}")


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# All metrics: the four comparisons repeated under every metric in scored.json
# --------------------------------------------------------------------------
# (key, column label, decimals). LPIPS and HaarPSI are included only if
# scored.json carries them.
ALL_METRICS = (("psnr", r"PSNR (dB) $\uparrow$", 2),
               ("ssim", r"SSIM $\uparrow$", 3),
               ("nmse", r"NMSE $\downarrow$", 3),
               ("lpips", r"LPIPS $\downarrow$", 3),
               ("haarpsi", r"HaarPSI $\uparrow$", 3))
LOWER_IS_BETTER = {"nmse", "lpips"}


def compute_all_metrics(scored: list[dict]) -> dict:
    """Median paired difference, interval, wins and p for every metric.

    Kept separate from compute() so that Table 2 is untouched. Benjamini-Hochberg
    is applied across the four comparisons within each metric, the same family
    the main table uses. A win is an improvement, so for NMSE and LPIPS it is a
    negative difference."""
    by = {}
    for r in scored:
        by.setdefault((r["physics"], r["source"]), {})[r["target"]] = r
    present = [m for m in ALL_METRICS if all(m[0] in r for r in scored)]
    out = {"metrics": present, "cells": {}}
    for metric, _, _ in present:
        keys, pvals = [], []
        for physics in MODELS:
            plain = by.get((physics, "bench_plain_gd"), {})
            for init, label, _ in INITS:
                citl = by.get((physics, f"bench_citl_{init}"), {})
                targets = sorted(set(plain) & set(citl))
                if not targets:
                    continue
                p_vals = np.array([plain[t][metric] for t in targets], float)
                c_vals = np.array([citl[t][metric] for t in targets], float)
                delta = c_vals - p_vals
                _, p = gdcitl.wilcoxon(c_vals, p_vals)
                lo, hi = gdcitl.bootstrap_median_ci(delta)
                better = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
                out["cells"][(metric, physics, init)] = {
                    "median": float(np.median(delta)), "ci_lo": lo, "ci_hi": hi,
                    "wins": int(better.sum()), "n": len(targets), "p": p}
                keys.append((metric, physics, init))
                pvals.append(p)
        for k, sig in zip(keys, gdcitl.benjamini_hochberg(pvals)):
            out["cells"][k]["significant"] = bool(sig)
    return out


def table_all_metrics(a: dict, path: Path) -> None:
    def num(v, n):
        t = f"{v:.{n}f}"
        return t.replace("-", "$-$", 1) if t.startswith("-") else t

    rows = []
    for i, physics in enumerate(MODELS):
        first = True
        for metric, label, n in a["metrics"]:
            cells = []
            for init, _, _ in INITS:
                c = a["cells"].get((metric, physics, init))
                if c is None:
                    cells += ["--", "--"]
                    continue
                dag = r"$^\dagger$" if c.get("significant") else ""
                cells += [f"{num(c['median'], n)} [{num(c['ci_lo'], n)}, {num(c['ci_hi'], n)}]",
                          f"{c['wins']}/{c['n']}{dag}"]
            model = physics.capitalize() if first else ""
            first = False
            rows.append(f"{model} & {label} & " + " & ".join(cells) + r" \\")
        if i < len(MODELS) - 1:
            rows.append(r"\midrule")

    lines = [
        r"% GENERATED by analysis/fig_citl_vs_plain.py - do not hand-edit.",
        r"% Destination: Supplement 1.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{GD+CITL against plain GD on the physical system, scored with "
        r"every metric. Each entry is the median paired difference across "
        r"\num{18} targets, GD+CITL minus plain GD, with a \SI{95}{\percent} interval from \num{10000} "
        r"bootstrap resamples. \emph{Wins} counts the targets improved by camera "
        r"feedback. Arrows mark the direction of improvement, so for NMSE and LPIPS "
        r"an improvement is a negative difference. $\dagger$ marks survival of "
        r"Benjamini-Hochberg correction at $q=0.05$ across the four comparisons "
        r"for that metric. Every metric is computed on the same images as Table 2 "
        r"of the main text: the reconstruction and the target, each min-max "
        r"normalized over the whole frame at the resolution of the forward model. "
        r"LPIPS and HaarPSI are defined in Section 2.6 of the main text. "
        r"As in Table 2, each block is a self-contained comparison at a single "
        r"forward model and values are not comparable between blocks.}",
        r"\label{tab:citlMetrics}",
        r"\footnotesize",
        r"\begin{tabular}{l l c c c c}", r"\toprule",
        r" & & \multicolumn{2}{c}{Random init} & \multicolumn{2}{c}{Simulation-seeded} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Model & Metric & $\Delta$ & Wins & $\Delta$ & Wins \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  table -> {path}")


def report(d: dict) -> None:
    for key in d["order"]:
        c = d["comparisons"][key]
        print(f"\n{c['physics'].upper()} / {c['label']}  (n={c['n']})")
        print(f"  plain GD   : {c['plain_stats']['median']:6.2f} dB")
        print(f"  GD+CITL    : {c['citl_stats']['median']:6.2f} dB")
        print(f"  delta      : {c['median']:+6.2f} dB "
              f"[{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]")
        print(f"  wins       : {c['wins']}/{c['n']}   p={c['p']:.3g}"
              f"{'  (survives BH)' if c.get('significant') else ''}")
    print("\nReminder: ideal and faithful are separate scoring problems. "
          "Compare within a block, never across.")


def run(scored_path, out_dir, tables_dir, figure: bool = True) -> dict:
    scored_path, out_dir, tables_dir = Path(scored_path), Path(out_dir), Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = json.loads(scored_path.read_text())
    d = compute(scored)
    report(d)
    table(d, tables_dir / "citl.tex")
    d["all_metrics"] = compute_all_metrics(scored)
    table_all_metrics(d["all_metrics"], tables_dir / "citl_metrics.tex")
    if figure:
        draw(d, out_dir)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, default=Path("scored.json"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--tables", type=Path, default=Path("tables"))
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    run(a.scored, a.out, a.tables, figure=not a.no_figure)


if __name__ == "__main__":
    main()
