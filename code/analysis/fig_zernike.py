r"""
fig_zernike.py - SCRIPT 5. The aberration operating envelope, in simulation.

WHY THIS EXPERIMENT EXISTS - the framing for the manuscript.

  The hardware result shows that camera feedback improves reconstructions on the
  bench. It does not show WHY, and it cannot: a real optical system's aberration
  is whatever it happens to be, unknown and unsweepable. So the bench alone
  leaves open the possibility that the improvement comes from something other
  than correcting optical error.

  This experiment closes that. We emulate an optical system in simulation, inject
  an aberration of KNOWN type and KNOWN magnitude, and withhold that aberration
  from the optimizer's own forward model - so the optimizer can sense it only
  through the feedback loop, exactly as it senses a real bench's aberration only
  through a real camera. If feedback recovers the target under those conditions,
  then aberration correction is a mechanism the method actually has, and the
  bench result has an explanation rather than just a number.

  Being able to do it in simulation is what makes the experimental result
  understood rather than merely observed. The sweep additionally bounds the
  mechanism: it says how much wavefront error feedback can absorb, and where it
  stops working.

THE COMPARISON. Every condition scores three ways; only one pair is the test:

  aberrated_psnr        CITL's hologram, measured on the aberrated system
  naive_aberrated_psnr  the no-feedback hologram, on that SAME aberrated system
  psnr                  CITL's hologram re-scored on the clean model

  aberrated vs naive_aberrated is the headline: both holograms judged by the
  system that actually exists. Scoring CITL's hologram on the clean model
  instead would flatter it, since that is not the system it was optimized
  against.

  ("naive" is the JSON's field name; the paper calls this condition plain GD.
  See gdcitl.PLAIN_GD_LABEL - nothing labelled "naive" reaches a caption.)

THE NULL CONTROL. Piston (Noll j=1) is a pure global phase and by construction
has exactly zero effect on replay intensity. Any measured improvement on piston
is therefore optimizer refinement, not aberration correction, and must be
subtracted before a correction magnitude is claimed. Under random init this
reads -0.001 dB, confirming the pipeline introduces no artefact; under simulation seeding
it does not, and that offset is a result in its own right.

Consumes the Zernike stress-test JSONs directly (not scored.json - this arm is
pure simulation and never touches a capture). Emits the envelope figure, an
optional per-mode diagnostic for the supplement, and tables/zernike.tex.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import figstyle
import gdcitl
from figstyle import C_RANDOM, C_WARM, C_MUTED, C_INK

# Display labels only. The KEYS stay "random"/"warm": they index the results
# tree, scored.json and LABEL_POS below, and nothing user-facing reads them.
# "Simulation-seeded" replaces "Warm start" because warm start conventionally
# implies reusing optimizer state, which this deliberately does not do, and
# because the point of the seed is that it costs no camera exposures - which
# "warm start" hides and "simulation-seeded" states.
INITS = (("random", "Random init", C_RANDOM),
         ("warm", "Simulation-seeded", C_WARM))

# The combined sweep is the confirmatory analysis: the six image-degrading modes
# present simultaneously at equal RMS, with total RMS swept. Piston and tip/tilt
# are excluded from it by construction so the x-axis reflects only genuinely
# reconstruction-degrading wavefront error.
COMBINED = "combined"
PER_MODE = "per_mode"
# Series-label anchors in DATA coordinates: (x waves RMS, y dB, vertical align).
# Warm start sits just above its peak band; random init sits in the clear gap
# between its lower band edge and the piston floor. Retune here, not inline.
LABEL_POS = {
    "warm":   (0.056, 2.62, "bottom"),
    "random": (0.050, 0.52, "top"),
}
PISTON = "piston"


# --------------------------------------------------------------------------
def paired_delta(records, sweep, magnitude, mode=None) -> np.ndarray:
    """Per-target paired improvement, both holograms on the true aberrated system."""
    rs = [r for r in records
          if r["sweep"] == sweep
          and r["magnitude_waves_rms"] == magnitude
          and (mode is None or r["mode_name"] == mode)]
    return np.array([r["aberrated_psnr"] - r["naive_aberrated_psnr"] for r in rs])


def compute() -> dict:
    runs = {key: gdcitl.zernike_results(key) for key, _, _ in INITS}
    out = {"curves": {}, "piston": {}, "per_mode": {}, "best_iteration": {}}

    for key, label, _ in INITS:
        recs = runs[key]

        # - combined sweep (confirmatory)
        mags = sorted({r["magnitude_waves_rms"] for r in recs
                       if r["sweep"] == COMBINED and r["magnitude_waves_rms"] > 0})
        rows, pvals = [], []
        for m in mags:
            d = paired_delta(recs, COMBINED, m)
            lo, hi = gdcitl.bootstrap_median_ci(d)
            _, p = gdcitl.wilcoxon(d, np.zeros_like(d))
            rows.append({"magnitude": m, "median": float(np.median(d)),
                         "ci_lo": lo, "ci_hi": hi, "p": p,
                         "wins": int((d > 0).sum()), "n": int(d.size)})
            pvals.append(p)
        for row, sig in zip(rows, gdcitl.benjamini_hochberg(pvals)):
            row["significant"] = bool(sig)
        out["curves"][key] = {"label": label, "mags": mags, "rows": rows}

        # - piston null control
        pist = [r["aberrated_psnr"] - r["naive_aberrated_psnr"] for r in recs
                if r["sweep"] == PER_MODE and r["mode_name"] == PISTON
                and r["magnitude_waves_rms"] > 0]
        out["piston"][key] = {"label": label, "mean": float(np.mean(pist)),
                              "sd": float(np.std(pist)), "n": len(pist)}

        # - per-mode diagnostic (supplement)
        modes = sorted({r["mode_name"] for r in recs if r["sweep"] == PER_MODE})
        out["per_mode"][key] = {
            mode: {"mags": (mm := sorted({r["magnitude_waves_rms"] for r in recs
                                          if r["sweep"] == PER_MODE
                                          and r["mode_name"] == mode})),
                   "median": [float(np.median(paired_delta(recs, PER_MODE, m, mode)))
                              for m in mm]}
            for mode in modes}

        # - iterations to best, combined sweep (context for the capture-cost claim)
        out["best_iteration"][key] = {
            m: float(np.median([r["best_iteration"] for r in recs
                                if r["sweep"] == COMBINED
                                and r["magnitude_waves_rms"] == m]))
            for m in mags}

    return out


# --------------------------------------------------------------------------
def draw(d: dict, out_dir: Path) -> None:
    r"""The envelope.

    Form: a continuous ordered x (aberration magnitude) against a paired
    difference, so a line with a bootstrap band - not bars. Two series, each
    direct-labelled as well as coloured, so identity never rests on colour alone.

    Three things the figure must show as geometry rather than as text:
      1. where correction works  - the rise at the low-magnitude end;
      2. where it stops          - the curve crosses zero. The y-axis is never
                                    cropped to hide this; a characterized limit
                                    is a result, not an embarrassment;
      3. the piston floor        - drawn as a horizontal reference so the
                                    aberration-independent component can be
                                    subtracted by eye. Its VALUE is in the table.
    """
    import matplotlib.pyplot as plt

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(figstyle.COL_W, 2.5))

    # Black, not C_WARM. The floor is a shared reference line; drawn in the
    # warm-start hue it read as a property of that curve rather than of the
    # measurement. Reference lines wear ink.
    floor = d["piston"]["warm"]["mean"]
    ax.axhline(floor, color=C_INK, linewidth=0.8, linestyle=figstyle.LS_DOT,
               zorder=2)
    ax.annotate("piston floor", xy=(1.0, floor), xytext=(-2, 3),
                textcoords="offset points", ha="right", va="bottom",
                color=C_MUTED, fontsize=6)

    for key, label, colour in INITS:
        c = d["curves"][key]
        mags = c["mags"]
        med = [r["median"] for r in c["rows"]]
        lo = [r["ci_lo"] for r in c["rows"]]
        hi = [r["ci_hi"] for r in c["rows"]]
        sig = [r["significant"] for r in c["rows"]]

        ax.fill_between(mags, lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax.plot(mags, med, color=colour, linewidth=1.4, zorder=3)
        # Filled = survives BH correction, open = does not. Secondary encoding,
        # so significance is never carried by colour alone.
        for want, kw in ((True, dict(color=colour, linewidths=0)),
                         (False, dict(facecolors="white", edgecolors=colour,
                                      linewidths=1.0))):
            xs = [m for m, s in zip(mags, sig) if s is want]
            ys = [v for v, s in zip(med, sig) if s is want]
            ax.scatter(xs, ys, s=16, zorder=4, **kw)
        # Hand-placed rather than anchored to the first data point. The old
        # anchor put "Warm start" inside its own confidence band and "Random
        # init" on top of the blue band's upper edge. These two positions are
        # the largest genuinely empty regions for each series: warm start above
        # its peak, random init in the gap between its band and the piston
        # floor. Data coordinates, so they track the curves if the data changes.
        lx, ly, lva = LABEL_POS[key]
        ax.annotate(label, xy=(lx, ly), color=colour, fontsize=7,
                    ha="left", va=lva, zorder=5)

    # Zero = feedback made no difference. The crossing is the envelope's edge.
    ax.axhline(0.0, color=C_MUTED, linewidth=0.8, zorder=1)

    # Top-only padding so the warm-start label cannot be clipped. The BOTTOM is
    # never padded away or cropped - the zero crossing is the result.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(ymax, 2.90))

    ax.set_xscale("log")
    ax.set_xlabel("Injected wavefront error (waves RMS)")
    ax.set_ylabel(r"$\Delta$PSNR, GD+CITL $-$ plain GD (dB)")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1.0"])
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    figstyle.save(fig, str(out_dir / "fig_zernike_envelope"))


def draw_per_mode(d: dict, out_dir: Path) -> None:
    r"""Supplementary diagnostic: each Zernike mode swept independently.

    Its job is to show WHICH aberrations feedback handles, and to display piston
    as a flat line at zero - the visual form of the null control.

    THE SPARE PANEL CARRIES THE LEGEND. Seven modes in a 4-wide grid leaves one
    slot empty. Hiding it wasted the only free space in the figure, and the
    figure shipped with two unlabelled colours. The legend now lives there, with
    the piston reading stated beside it, because piston is the panel a reader is
    most likely to misread as a broken axis.

    TICKS MATCH THE ENVELOPE FIGURE. The default log locator labelled only
    10^-1 and 10^0 here while the envelope figure showed 0.05 to 1.0 in plain
    numerals. Two views of the same sweep with differently-formatted axes invite
    the reader to think they cover different ranges.

    TYPE SIZE IS DERIVED, not fixed: authored at COL_W_2 and placed at
    \linewidth, so a hardcoded 7pt would reach the page at 5.4pt.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figstyle.apply()
    fs = figstyle.title_size(figstyle.COL_W_2)

    modes = sorted(d["per_mode"]["warm"])
    ncol = 4
    nrow = int(np.ceil((len(modes) + 1) / ncol))     # +1 reserves the legend slot
    fig, axes = plt.subplots(nrow, ncol, figsize=(figstyle.COL_W_2, 1.75 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, mode in zip(axes, modes):
        for key, label, colour in INITS:
            pm = d["per_mode"][key][mode]
            ax.plot(pm["mags"], pm["median"], color=colour, linewidth=1.3)
        ax.axhline(0.0, color=C_MUTED, linewidth=0.7, zorder=1)
        ax.set_xscale("log")
        ax.set_title(mode.replace("_", " "), loc="left", fontsize=fs, color=C_INK)
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1.0"])
        ax.tick_params(which="minor", length=0)
        ax.tick_params(labelsize=fs * 0.85)

    spare = axes[len(modes):]
    if len(spare):
        lax = spare[0]
        # Blank this panel with tick_params and spines ONLY. Do NOT call
        # set_xticks/set_yticks/set_xscale here: sharex/sharey means the axes
        # share locators and scales, so clearing them on this one panel strips
        # the tick labels from ALL of them.
        for s in lax.spines.values():
            s.set_visible(False)
        lax.tick_params(which="both", length=0,
                        labelbottom=False, labelleft=False)
        lax.grid(False)
        lax.legend(handles=[Line2D([], [], color=c, lw=1.6, label=l)
                            for _, l, c in INITS],
                   loc="upper left", fontsize=fs * 0.95, frameon=False,
                   handlelength=1.8, labelspacing=0.4, borderaxespad=0.2)
        pw = d["piston"]["warm"]["mean"]
        lax.text(0.0, 0.34,
                 "Piston has no effect on\nreplay intensity by\n"
                 f"construction. Its offset\n({pw:+.2f} dB, seeded) is the\n"
                 "null control of the\nmain text.",
                 transform=lax.transAxes, fontsize=fs * 0.8, color=C_MUTED,
                 va="top", ha="left", linespacing=1.4)
        for ax in spare[1:]:
            ax.set_visible(False)

    fig.supxlabel("Injected wavefront error (waves RMS)", fontsize=fs)
    fig.supylabel(r"$\Delta$PSNR, GD+CITL $-$ plain GD (dB)", fontsize=fs)
    fig.tight_layout()
    figstyle.save(fig, str(out_dir / "fig_zernike_per_mode"))


# --------------------------------------------------------------------------
def table(d: dict, tables_dir: Path) -> None:
    r"""Emit the two Zernike tables as SEPARATE files.

    Previously both went into one tables/zernike.tex inside a single `table`
    float, separated by \vspace{1em}. Two problems with that: LaTeX cannot place
    them independently (they are one float, so they travel together), and two
    \caption commands in one float is non-standard enough that journal
    production may query it.

    They now have different destinations, which makes the split necessary rather
    than merely tidy:

      zernike_sweep.tex   -> Supplement 1, as Table S1. Label tab:s-envelope.
                             24 rows, every sampled magnitude.
      zernike_piston.tex  -> main text. Label tab:piston. Two rows.

    The sweep caption therefore cannot refer to the piston table as being
    "below" any more, and says "in the main text" instead.

    Takes a DIRECTORY, not a path - it writes two files. Call site updated in
    run() accordingly.
    """
    def f(v, n=2, signed=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:+.{n}f}" if signed else f"{v:.{n}f}"

    def pfmt(p):
        return r"$<10^{-4}$" if p < 1e-4 else (r"$<0.001$" if p < 0.001 else f"{p:.3f}")

    def write(path: Path, lines) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        print(f"  table -> {path}")

    tables_dir = Path(tables_dir)

    # --- Table S1: the full sweep (supplement) ------------------------------
    rows = []
    for i, (key, label, _) in enumerate(INITS):
        c = d["curves"][key]
        for j, r in enumerate(c["rows"]):
            dag = r"$^\dagger$" if r["significant"] else ""
            rows.append(
                f"{label if j == 0 else ''} & {r['magnitude']:.2f} & "
                f"{f(r['median'], 2, True)} & "
                f"[{f(r['ci_lo'], 2, True)}, {f(r['ci_hi'], 2, True)}] & "
                f"{r['wins']}/{r['n']} & {pfmt(r['p'])}{dag} \\\\")
        if i < len(INITS) - 1:
            rows.append(r"\midrule")

    write(tables_dir / "zernike_sweep.tex", [
        r"% GENERATED by analysis/fig_zernike.py - do not hand-edit.",
        r"% Destination: Supplement 1, Table S1.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Aberration operating envelope at every sampled magnitude. "
        r"Median paired change in PSNR between GD+CITL and plain GD, both "
        r"holograms scored on the true aberrated system, over \num{18} targets, "
        r"with percentile bootstrap \SI{95}{\percent} intervals "
        r"(\num{10000} resamples). The six image-degrading modes are present "
        r"simultaneously at equal RMS and the total is swept. $\dagger$ marks "
        r"survival of Benjamini-Hochberg correction at $q=0.05$ across the "
        r"sweep. Values are not corrected for the piston offset, which is given "
        r"in the main text.}",
        r"\label{tab:s-envelope}",
        r"\begin{tabular}{l S[table-format=1.2] S[table-format=+1.2] c c l}",
        r"\toprule",
        r"Condition & {RMS ($\lambda$)} & {$\Delta$PSNR (dB)} & "
        r"{\SI{95}{\percent} CI} & {Wins} & {$p$} \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])

    # --- Table (main text): the piston null control -------------------------
    piston_rows = [
        f"{p['label']} & {f(p['mean'], 3, True)} & {f(p['sd'], 3)} \\\\"
        for p in d["piston"].values()]

    write(tables_dir / "zernike_piston.tex", [
        r"% GENERATED by analysis/fig_zernike.py - do not hand-edit.",
        r"% Destination: main text.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Piston null control. Piston (Noll $j=1$) is a pure global "
        r"phase and by construction has no effect on replay intensity, so any "
        r"measured change on this mode is optimizer refinement rather than "
        r"aberration correction. Mean and standard deviation across the "
        r"magnitude sweep.}",
        r"\label{tab:piston}",
        # \footnotesize matches the manuscript.
        r"\footnotesize",
        r"\begin{tabular}{l S[table-format=+1.3] S[table-format=1.3]}",
        r"\toprule",
        r"Condition & {Mean $\Delta$PSNR (dB)} & {SD across magnitudes} \\",
        r"\midrule", *piston_rows, r"\bottomrule", r"\end{tabular}",
        r"\end{table}",
    ])



# --------------------------------------------------------------------------
def report(d: dict) -> None:
    for key, label, _ in INITS:
        c = d["curves"][key]
        print(f"\n{label.upper()}  (combined sweep, n=18 per magnitude)")
        print(f"  {'RMS':>6} {'dPSNR':>8} {'95% CI':>18} {'wins':>7}  BH")
        for r in c["rows"]:
            print(f"  {r['magnitude']:>6.2f} {r['median']:>+8.2f} "
                  f"  [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] "
                  f"  {r['wins']:>2d}/{r['n']:<2d}  {'*' if r['significant'] else ''}")
        crossings = [r["magnitude"] for r in c["rows"] if r["median"] <= 0]
        print(f"  envelope edge: "
              + (f"first non-positive at {min(crossings):.2f} lambda RMS"
                 if crossings else "positive across the whole sweep"))

    print("\nPiston null control (zero effect by construction):")
    for p in d["piston"].values():
        print(f"  {p['label']:12s} {p['mean']:+.3f} dB  (sd {p['sd']:.3f}, n={p['n']})")


def run(out_dir, tables_dir, figure: bool = True, per_mode: bool = True) -> dict:
    out_dir, tables_dir = Path(out_dir), Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = compute()
    report(d)
    table(d, tables_dir)          # writes TWO files; see table()
    if figure:
        draw(d, out_dir)
        if per_mode:
            draw_per_mode(d, out_dir)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--tables", type=Path, default=Path("tables"))
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    run(a.out, a.tables, figure=not a.no_figure)


if __name__ == "__main__":
    main()