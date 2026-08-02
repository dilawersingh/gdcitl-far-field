r"""
fig_gap.py - SCRIPT 3. The simulation-to-experiment gap (manuscript 3.1).

The paper's opening result and the motivation for everything after it: a hologram
that scores well in simulation does not score well when displayed. This script
measures that gap, and asks how much of it is a modelling deficiency by repeating
the measurement under both forward models.

  simulation      the GD hologram scored against its own simulated replay
  bench_plain_gd  that SAME hologram displayed on the SLM and photographed

Two questions, two tests, both pre-specified:

  Q1  Is there a gap?              Wilcoxon, simulation vs bench, per model.
  Q2  Does the faithful forward    Wilcoxon, gap_ideal vs gap_faithful, paired
      model narrow it?             per target. THIS is the interesting one --
                                   Q1's answer is not in doubt.

Q2 is what justifies carrying the ideal/faithful distinction through the rest of
the paper. If the faithful model closed the gap entirely there would be no case
for camera feedback; if it closes none of it, the ideal column is the only one
worth reporting. The measured answer decides how 3.2 and 3.3 are framed.

Figure: paired slope plots (simulation -> bench) per model, plus the per-target
gap compared across models. A slope plot is the correct form for a paired
comparison at n=18 - it shows every pair, the direction of every change, and the
spread, where a bar of two medians hides all three.

Consumes scored.json from script 2. Emits the figure and tables/gap.tex.

Usage:
    python fig_gap.py --scored scored.json --out figures --tables tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import figstyle
import gdcitl
from figstyle import C_SIMULATION, C_BENCH, C_IDEAL, C_FAITHFUL, C_INK, C_MUTED

METRIC = "psnr"          # figure metric; the table carries all four
MODELS = ("ideal", "faithful")


# --------------------------------------------------------------------------
def compute(scored: list[dict]) -> dict:
    """Paired arrays and tests. Separated from drawing so the numbers can be
    checked without matplotlib."""
    out: dict = {"models": {}, "metrics": {}}

    for physics in MODELS:
        sim = {r["target"]: r for r in scored
               if r["physics"] == physics and r["source"] == "simulation"}
        bench = {r["target"]: r for r in scored
                 if r["physics"] == physics and r["source"] == "bench_plain_gd"}
        targets = sorted(set(sim) & set(bench))
        if len(targets) != gdcitl.N_TARGETS:
            print(f"  ! {physics}: {len(targets)} paired targets, "
                  f"expected {gdcitl.N_TARGETS}")

        s = np.array([sim[t][METRIC] for t in targets], float)
        b = np.array([bench[t][METRIC] for t in targets], float)
        stat, p = gdcitl.wilcoxon(b, s)

        out["models"][physics] = {
            "targets": targets, "sim": s, "bench": b, "gap": s - b,
            "sim_stats": gdcitl.summarize(s),
            "bench_stats": gdcitl.summarize(b),
            "gap_stats": gdcitl.summarize(s - b),
            "W": stat, "p": p,
        }

        # every metric, for the table
        for m in ("psnr", "ssim", "nmse", "diffraction_efficiency"):
            out["metrics"][(physics, "simulation", m)] = gdcitl.summarize(
                [sim[t][m] for t in targets])
            out["metrics"][(physics, "bench_plain_gd", m)] = gdcitl.summarize(
                [bench[t][m] for t in targets])

    # Q2 - does the faithful model narrow the gap? Paired over the targets
    # common to both models.
    a, b_ = out["models"]["ideal"], out["models"]["faithful"]
    common = sorted(set(a["targets"]) & set(b_["targets"]))
    gi = np.array([a["gap"][a["targets"].index(t)] for t in common], float)
    gf = np.array([b_["gap"][b_["targets"].index(t)] for t in common], float)
    stat, p = gdcitl.wilcoxon(gf, gi)
    out["gap_comparison"] = {
        "targets": common, "gap_ideal": gi, "gap_faithful": gf,
        "narrowed_in": int((gf < gi).sum()), "n": len(common),
        "median_reduction": float(np.median(gi - gf)), "W": stat, "p": p,
    }
    return out


# --------------------------------------------------------------------------
def draw(d: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.apply()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(figstyle.COL_W_2, 2.7),
        gridspec_kw={"width_ratios": [1.5, 1.0]})

    # - (a) simulation -> bench, both models, offset groups
    for gi, (physics, x0) in enumerate(zip(MODELS, (0.0, 2.2))):
        m = d["models"][physics]
        for s, b in zip(m["sim"], m["bench"]):
            ax1.plot([x0, x0 + 1], [s, b], color=C_MUTED, alpha=0.35,
                     linewidth=0.8, zorder=2, solid_capstyle="round")
        ax1.scatter(np.full(m["sim"].size, x0), m["sim"], s=16,
                    color=C_SIMULATION, zorder=3, linewidths=0)
        ax1.scatter(np.full(m["bench"].size, x0 + 1), m["bench"], s=16,
                    color=C_BENCH, zorder=3, linewidths=0)
        ax1.plot([x0, x0 + 1],
                 [np.median(m["sim"]), np.median(m["bench"])],
                 color=C_INK, linewidth=2.0, zorder=4, solid_capstyle="round")
        ax1.annotate(physics.capitalize(), xy=(x0 + 0.5, 1.0),
                     xycoords=("data", "axes fraction"), ha="center",
                     va="bottom", fontsize=7, color=C_MUTED)

    ax1.set_xticks([0, 1, 2.2, 3.2])
    ax1.set_xticklabels(["Simulation", "Bench", "Simulation", "Bench"])
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(axis="x", visible=False)
    ax1.set_title("(a)", loc="left", color=C_MUTED)

    # - (b) the gap itself, ideal vs faithful, paired per target
    g = d["gap_comparison"]
    for a_, f_ in zip(g["gap_ideal"], g["gap_faithful"]):
        ax2.plot([0, 1], [a_, f_], color=C_MUTED, alpha=0.35, linewidth=0.8,
                 zorder=2, solid_capstyle="round")
    ax2.scatter(np.zeros(g["n"]), g["gap_ideal"], s=16, color=C_IDEAL,
                zorder=3, linewidths=0)
    ax2.scatter(np.ones(g["n"]), g["gap_faithful"], s=16, color=C_FAITHFUL,
                zorder=3, linewidths=0)
    ax2.plot([0, 1], [np.median(g["gap_ideal"]), np.median(g["gap_faithful"])],
             color=C_INK, linewidth=2.0, zorder=4, solid_capstyle="round")
    # Zero gap = simulation and bench agree. The distance from this line is the
    # quantity the whole paper is about, so it must be visible.
    ax2.axhline(0.0, color=C_MUTED, linewidth=0.8, zorder=1)
    ax2.set_xlim(-0.35, 1.35)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Ideal", "Faithful"])
    ax2.set_ylabel("Gap, simulation $-$ bench (dB)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("(b)", loc="left", color=C_MUTED)

    fig.tight_layout()
    figstyle.save(fig, str(out_dir / "fig_gap"))


# --------------------------------------------------------------------------
def table(d: dict, path: Path) -> None:
    def f(v, n=2):
        return "--" if v is None or np.isnan(v) else f"{v:.{n}f}"

    rows = []
    for physics in MODELS:
        for src, label in (("simulation", "Simulation"), ("bench_plain_gd", "Bench")):
            g = lambda m: d["metrics"][(physics, src, m)]
            rows.append(
                f"{physics.capitalize() if src == 'simulation' else ''} & {label} & "
                f"{f(g('psnr')['median'])} ({f(g('psnr')['q1'])}--{f(g('psnr')['q3'])}) & "
                f"{f(g('ssim')['median'], 3)} & {f(g('nmse')['median'], 3)} & "
                f"{f(g('diffraction_efficiency')['median'], 3)} \\\\")
        m = d["models"][physics]
        rows.append(
            f" & \\textit{{Gap}} & \\textit{{{f(m['gap_stats']['median'])} "
            f"[{f(m['gap_stats']['ci_lo'])}, {f(m['gap_stats']['ci_hi'])}]}} & & & \\\\")
        if physics != MODELS[-1]:
            rows.append(r"\midrule")

    g = d["gap_comparison"]
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{The simulation-to-experiment gap. Median (interquartile range) "
        r"across \num{18} targets for the same gradient-descent holograms scored "
        r"in simulation and displayed on the bench, under both forward models. "
        r"\emph{Gap} is the median paired difference with a bootstrap "
        r"\SI{95}{\percent} interval. The faithful model narrows the gap in "
        rf"{g['narrowed_in']}/{g['n']} targets (median reduction "
        rf"\SI{{{g['median_reduction']:.2f}}}{{\dB}}, Wilcoxon "
        rf"$p={g['p']:.3g}$).}}",
        r"\label{tab:gap}",
        r"\begin{tabular}{l l c c c c}", r"\toprule",
        r"Model & Source & PSNR (dB) & SSIM & NMSE & Diff. eff. \\", r"\midrule",
        *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  table -> {path}")


# --------------------------------------------------------------------------
def run(scored_path, out_dir, tables_dir, figure: bool = True) -> dict:
    r"""Notebook entry point. Every figure script in this pipeline exposes the
    same signature, so a notebook cell is one call and the returned dict stays
    available for inspection.

    Returns the computed statistics - the same dict the figure and table were
    built from, so anything quoted in the manuscript can be traced back to it
    without re-deriving it.
    """
    scored_path, out_dir, tables_dir = Path(scored_path), Path(out_dir), Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = compute(json.loads(scored_path.read_text()))
    report(d)
    table(d, tables_dir / "gap.tex")
    if figure:
        draw(d, out_dir)
    return d


def report(d: dict) -> None:
    """Print the headline numbers. Called by run(); safe to call again."""
    for physics in MODELS:
        m = d["models"][physics]
        print(f"\n{physics.upper()} (n={len(m['targets'])})")
        print(f"  simulation PSNR : {m['sim_stats']['median']:.2f} dB")
        print(f"  bench PSNR      : {m['bench_stats']['median']:.2f} dB")
        print(f"  gap             : {m['gap_stats']['median']:.2f} dB "
              f"[{m['gap_stats']['ci_lo']:.2f}, {m['gap_stats']['ci_hi']:.2f}], "
              f"p={m['p']:.3g}")

    g = d["gap_comparison"]
    print(f"\nDoes the faithful model narrow the gap?")
    print(f"  narrowed in {g['narrowed_in']}/{g['n']} targets, "
          f"median reduction {g['median_reduction']:.2f} dB, p={g['p']:.3g}")


def main() -> None:
    """CLI entry point. The notebook uses run() instead."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, default=Path("scored.json"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--tables", type=Path, default=Path("tables"))
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    run(a.scored, a.out, a.tables, figure=not a.no_figure)


if __name__ == "__main__":
    main()