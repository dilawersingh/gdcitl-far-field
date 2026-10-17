r"""
dc_mask_diagnostic.py - read-only sensitivity check for the zero-order (DC) disc.

Question it answers: the DC disc is excluded from the training loss and from
the capture normalization sum, but NOT from compute_metrics. Do the paper's
conclusions change if the disc is also excluded from the reported metrics?

What it does, for all 144 records in scored.json:
  1. Loads every input exactly as score_captures.score_all does (same work list,
     same calibration, same capture loader, same target preparation).
  2. Scores each record twice:
       orig   - evaluate_methods.compute_metrics, unchanged. This must reproduce
                scored.json; the script checks it and says so.
       masked - the same metrics with the DC disc removed from BOTH capture and
                target before min-max normalization. PSNR/MSE/NMSE over the
                remaining pixels; SSIM map averaged over pixels whose 7x7
                window does not touch the disc.
  3. Recomputes the four headline paired comparisons (GD+CITL vs plain GD,
     {ideal, faithful} x {random, seeded}) under both scorings, with the
     paper's own statistics (gdcitl.wilcoxon, bootstrap_median_ci,
     benjamini_hochberg), plus the seeded-vs-random quality check from the
     capture-cost section.

Nothing under results/ is modified. Everything is written to --out (default
output/dc_diagnostic/), and with more than one radius the Supplement 1 table
zero_order.tex is written to output/tables/.

The masked metric reduces EXACTLY to compute_metrics when the mask is empty;
the script verifies this on the first record before scoring anything.

Usage:
    python dc_mask_diagnostic.py
    python dc_mask_diagnostic.py --radius 10 20 30 40   # mask-radius sensitivity

or from the analysis notebook:
    import dc_mask_diagnostic as dcd
    dc = dcd.run(radius=20)
    dc = dcd.run(radius=[10, 20, 30, 40])   # one pass, all radii
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths - every default resolves inside the deposit (see paths.py)
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent            # code/analysis
for _d in (_HERE, _HERE.parent / "method"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import paths                                    # noqa: E402

RESULTS = Path(os.environ.get("GDCITL_RESULTS", paths.RESULTS))
REPLAY = Path(os.environ.get("GDCITL_REPLAY", paths.REPLAY))
STOCK = Path(os.environ.get("GDCITL_STOCK", paths.TARGETS))
SCORED = paths.SCORED
OUT = paths.OUTPUT / "dc_diagnostic"
REPO = paths.ROOT / "code" / "method"             # evaluate_methods, raw_camera_processor

# gdcitl reads these at import time, so set them first
os.environ.setdefault("GDCITL_RESULTS", str(RESULTS))
os.environ.setdefault("GDCITL_REPLAY", str(REPLAY))
os.environ.setdefault("GDCITL_STOCK", str(STOCK))

import gdcitl                                   # noqa: E402
import score_captures as sc                     # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402
from scipy.ndimage import binary_dilation       # noqa: E402

SSIM_WIN = 7          # skimage default, as used by compute_metrics
EPS = 1e-12


# --------------------------------------------------------------------------
# Masked metrics
# --------------------------------------------------------------------------
def disc_mask(shape, pad_factor: int, radius_native: float) -> np.ndarray:
    """Same disc the loss and the normalisation use, scaled by pad_factor."""
    cx = gdcitl.DC_CENTER_NATIVE[0] * pad_factor
    cy = gdcitl.DC_CENTER_NATIVE[1] * pad_factor
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (radius_native * pad_factor) ** 2


def masked_metrics(recon, target, pad_factor: int, mask: np.ndarray,
                   perceptual_fn=None) -> dict:
    """compute_metrics with the pixels in `mask` excluded everywhere.

    With an all-False mask this returns exactly what compute_metrics returns
    (checked by self_test).

    perceptual_fn (perceptual.perceptual_metrics) adds LPIPS and HaarPSI. A
    network cannot skip pixels, so for these the disc is normalised out (the
    min-max uses unmasked pixels only) and filled with the same value in both
    images, so the zero-order region is identical in capture and target."""
    recon = np.asarray(recon, np.float32)
    target = np.asarray(target, np.float32)
    valid = ~mask

    def minmax(x):
        lo, hi = x[valid].min(), x[valid].max()
        y = (x - lo) / (hi - lo + EPS)
        y[mask] = 0.0                    # identical fill in both images
        return y

    r01, t01 = minmax(recon), minmax(target)
    diff2 = (r01 - t01)[valid] ** 2
    mse = float(diff2.mean())
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
    nmse = float(diff2.sum() / (np.sum(t01[valid] ** 2) + EPS))

    # SSIM: skimage's scalar SSIM is the mean of the SSIM map with a
    # (win-1)//2 border cropped. Do the same, and additionally drop every
    # pixel whose window touches the disc.
    _, smap = structural_similarity(t01, r01, data_range=1.0, full=True)
    pad = (SSIM_WIN - 1) // 2
    keep = ~binary_dilation(mask, structure=np.ones((SSIM_WIN, SSIM_WIN), bool))
    keep[:pad, :] = keep[-pad:, :] = False
    keep[:, :pad] = keep[:, -pad:] = False
    ssim = float(smap[keep].mean())

    r_sum = np.clip(recon, 0.0, None) * valid
    r_sum = r_sum / (r_sum.sum() + EPS)
    t_sum = np.clip(target, 0.0, None) * valid
    t_sum = t_sum / (t_sum.sum() + EPS)
    support = t_sum > (1e-6 / (pad_factor ** 2))
    eff = float(r_sum[support].sum() / (r_sum.sum() + EPS))

    out = {"psnr": psnr, "ssim": ssim, "mse": mse, "nmse": nmse,
           "diffraction_efficiency": eff}
    if perceptual_fn is not None:
        out.update(perceptual_fn(r01, t01))   # r01/t01 already in [0, 1], disc filled
    return out


def self_test(compute_metrics, recon, target, pf, perceptual_fn=None):
    """Empty mask must reproduce the unmasked metrics to float precision."""
    a = compute_metrics(recon, target, pf)
    if perceptual_fn is not None:
        a.update(perceptual_fn(recon, target))
    b = masked_metrics(recon, target, pf, np.zeros(recon.shape, bool), perceptual_fn)
    for k in a:
        if not np.isclose(a[k], b[k], rtol=1e-4, atol=1e-5):
            raise RuntimeError(f"self-test failed on {k}: {a[k]} vs {b[k]}")
    print("  self-test OK: empty mask reproduces the unmasked metrics"
          + (" (incl. LPIPS, HaarPSI)" if perceptual_fn else ""))


# --------------------------------------------------------------------------
# Work list - same construction as score_captures.score_all
# --------------------------------------------------------------------------
def work_list():
    run = sc.replay_run(REPLAY)
    work = []
    for physics in ("ideal", "faithful"):
        pf = gdcitl.FORWARD_MODELS[physics]["pad_factor"]
        for init in ("random_init", "warm_start"):
            for r in gdcitl.citl_runs(init, physics, RESULTS):
                cap = (r["capture_raw"] if r["capture_raw"].exists()
                       else r["capture_processed"] if r["capture_processed"].exists()
                       else None)
                work.append({"target": r["target"], "physics": physics, "pad_factor": pf,
                             "source": f"bench_citl_{init}", "capture": cap,
                             "target_image": sc.target_path(r["target"], STOCK)})
        for d in sorted((run / physics).iterdir()):
            if not d.is_dir():
                continue
            gd = d / "gd.jpg"
            work.append({"target": d.name, "physics": physics, "pad_factor": pf,
                         "source": "bench_plain_gd", "capture": gd if gd.exists() else None,
                         "target_image": sc.target_path(d.name, STOCK)})
            work.append({"target": d.name, "physics": physics, "pad_factor": pf,
                         "source": "simulation", "capture": None,
                         "recon_npy": sc.simulation_recon(RESULTS, physics, d.name),
                         "target_image": sc.target_path(d.name, STOCK)})
    return work


# --------------------------------------------------------------------------
# Statistics - same as fig_citl_vs_plain.compute, for both scorings
# --------------------------------------------------------------------------
COMPARISONS = [  # (label, physics, better source, baseline source)
    ("ideal / random  vs plain", "ideal", "bench_citl_random_init", "bench_plain_gd"),
    ("ideal / seeded  vs plain", "ideal", "bench_citl_warm_start", "bench_plain_gd"),
    ("faith / random  vs plain", "faithful", "bench_citl_random_init", "bench_plain_gd"),
    ("faith / seeded  vs plain", "faithful", "bench_citl_warm_start", "bench_plain_gd"),
]
QUALITY_CHECK = [  # capture-cost half (b): does seeding cost quality?
    ("ideal / seeded  vs random", "ideal", "bench_citl_warm_start", "bench_citl_random_init"),
    ("faith / seeded  vs random", "faithful", "bench_citl_warm_start", "bench_citl_random_init"),
]


# For these, lower is better, so a "win" is a negative difference.
LOWER_IS_BETTER = {"mse", "nmse", "lpips"}


def paired(rows, scoring, metric, physics, a_src, b_src):
    by = {(r["physics"], r["source"], r["target"]): r for r in rows}
    targets = sorted({r["target"] for r in rows if r["physics"] == physics})
    a = np.array([by[(physics, a_src, t)][f"{scoring}_{metric}"] for t in targets])
    b = np.array([by[(physics, b_src, t)][f"{scoring}_{metric}"] for t in targets])
    d = a - b
    _, p = gdcitl.wilcoxon(a, b)
    lo, hi = gdcitl.bootstrap_median_ci(d)
    return {"targets": targets, "delta": d, "median": float(np.median(d)),
            "ci_lo": lo, "ci_hi": hi, "n": len(d), "p": p,
            "wins": int((d < 0).sum() if metric in LOWER_IS_BETTER else (d > 0).sum())}


def stats_block(rows, comps, metric):
    out = {}
    for scoring in ("orig", "masked"):
        res = [paired(rows, scoring, metric, ph, a, b) for _, ph, a, b in comps]
        sig = gdcitl.benjamini_hochberg([r["p"] for r in res])
        for (label, *_), r, s in zip(comps, res, sig):
            r["bh_significant"] = bool(s)
            out[(label, scoring)] = r
    return out


def fmt(r, nd=2):
    return (f"{r['median']:+.{nd}f} [{r['ci_lo']:+.{nd}f}, {r['ci_hi']:+.{nd}f}]  "
            f"wins {r['wins']:2d}/{r['n']}  p={r['p']:.2g}  "
            f"BH {'sig' if r.get('bh_significant') else 'NS '}")


# --------------------------------------------------------------------------
def _write_outputs(rows, radius, out_dir: Path) -> str:
    """per_record.csv and summary.txt for one radius. Returns the summary text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_record.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    L = []
    worst = max(r["repro_abs_err_psnr"] for r in rows)
    L.append(f"DC mask diagnostic  |  radius {radius:g} px native  |  {len(rows)} records")
    L.append(f"Reproduction of published scored.json: max |PSNR error| = {worst:.2e} dB"
             + ("   OK" if worst < 1e-3 else "   ** MISMATCH - stop and check **"))
    L.append("")
    metrics = [("psnr", 2), ("ssim", 3), ("nmse", 3)]
    if "orig_lpips" in rows[0]:
        metrics += [("lpips", 3), ("haarpsi", 3)]
    for metric, nd in metrics:
        L.append(f"=== {metric.upper()}: GD+CITL minus plain GD, paired, n=18 "
                 f"(median delta [95% CI], wins, Wilcoxon p, BH q=0.05 across the 4) ===")
        st = stats_block(rows, COMPARISONS, metric)
        for label, *_ in COMPARISONS:
            L.append(f"  {label}  orig   {fmt(st[(label, 'orig')], nd)}")
            L.append(f"  {'':24s}  masked {fmt(st[(label, 'masked')], nd)}")
        L.append("")
    L.append("=== Capture-cost quality check: seeded minus random (PSNR) ===")
    L.append("    (the paper reports the median and CI only; p and BH here are extra)")
    st = stats_block(rows, QUALITY_CHECK, "psnr")
    for label, *_ in QUALITY_CHECK:
        L.append(f"  {label}  orig   {fmt(st[(label, 'orig')])}")
        L.append(f"  {'':24s}  masked {fmt(st[(label, 'masked')])}")
    L.append("")
    L.append("=== Median PSNR by source (orig -> masked) ===")
    for physics in ("ideal", "faithful"):
        for src in ("simulation", "bench_plain_gd", "bench_citl_random_init", "bench_citl_warm_start"):
            sub = [r for r in rows if r["physics"] == physics and r["source"] == src]
            o = np.median([r["orig_psnr"] for r in sub])
            m = np.median([r["masked_psnr"] for r in sub])
            L.append(f"  {physics:8s} {src:24s} {o:6.2f} -> {m:6.2f}   ({m - o:+.2f})")
    L.append("")
    L.append("=== Per-target PSNR delta, faithful / random vs plain (orig -> masked) ===")
    st = stats_block(rows, COMPARISONS, "psnr")
    o, m = st[("faith / random  vs plain", "orig")], st[("faith / random  vs plain", "masked")]
    for t, do, dm in sorted(zip(o["targets"], o["delta"], m["delta"]), key=lambda x: x[1]):
        flag = "   <- sign flip" if np.sign(do) != np.sign(dm) else ""
        L.append(f"  {t:16s} {do:+6.2f} -> {dm:+6.2f}{flag}")

    text = "\n".join(L)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    return text


def _cross_radius_table(rows_by_r) -> str:
    """One line per (metric, comparison, radius): the headline paired result."""
    any_r = next(iter(rows_by_r.values()))
    metrics = [("psnr", 2), ("ssim", 3)]
    if "orig_lpips" in any_r[0]:
        metrics += [("lpips", 3), ("haarpsi", 3)]
    L = ["Mask-radius sensitivity. radius 0 = published metric (no mask).",
         "Wins count improvements (lower is better for LPIPS).", ""]
    for metric, nd in metrics:
        L.append(f"=== {metric.upper()}: GD+CITL minus plain GD ===")
        base = stats_block(any_r, COMPARISONS, metric)
        per_r = {r: stats_block(rows, COMPARISONS, metric) for r, rows in rows_by_r.items()}
        for label, *_ in COMPARISONS:
            L.append(f"  {label}")
            L.append(f"    r =  0  {fmt(base[(label, 'orig')], nd)}")
            for r in rows_by_r:
                L.append(f"    r = {r:2g}  {fmt(per_r[r][(label, 'masked')], nd)}")
        L.append("")
    L.append("=== Seeded minus random quality (PSNR, masked), median [95% CI] ===")
    base = stats_block(any_r, QUALITY_CHECK, "psnr")
    for label, *_ in QUALITY_CHECK:
        b = base[(label, "orig")]
        L.append(f"  {label}")
        L.append(f"    r =  0  {b['median']:+.2f} [{b['ci_lo']:+.2f}, {b['ci_hi']:+.2f}]")
        for r, rows in rows_by_r.items():
            q = stats_block(rows, QUALITY_CHECK, "psnr")[(label, "masked")]
            L.append(f"    r = {r:2g}  {q['median']:+.2f} [{q['ci_lo']:+.2f}, {q['ci_hi']:+.2f}]")
    return "\n".join(L)


# --------------------------------------------------------------------------
# LaTeX table for Supplement 1 (zero-order sensitivity)
# --------------------------------------------------------------------------
# Same row order and labels as tables/citl_metrics.tex.
TEX_ROWS = [  # (model, initialization, comparison label)
    ("Faithful", "Random init", "faith / random  vs plain"),
    ("Faithful", "Simulation-seeded", "faith / seeded  vs plain"),
    ("Ideal", "Random init", "ideal / random  vs plain"),
    ("Ideal", "Simulation-seeded", "ideal / seeded  vs plain"),
]
TEX_METRIC_NAMES = {"psnr": "PSNR", "ssim": "SSIM", "nmse": "NMSE",
                    "lpips": "LPIPS", "haarpsi": "HaarPSI"}


def _zero_order_tex(rows_by_r) -> str:
    """Table S4: PSNR paired difference at every radius, with the other
    metrics, wins and significance summarised in the caption. Every number
    in the table and the caption is computed here."""
    radii = list(rows_by_r)
    any_r = rows_by_r[radii[0]]
    metrics = ["psnr", "ssim", "nmse"]
    if "orig_lpips" in any_r[0]:
        metrics += ["lpips", "haarpsi"]

    # orig (radius 0) is identical in every radius's rows, so take it from any.
    base = {m: stats_block(any_r, COMPARISONS, m) for m in metrics}
    per_r = {r: {m: stats_block(rows, COMPARISONS, m) for m in metrics}
             for r, rows in rows_by_r.items()}

    def cell(res):
        dag = r"$^\dagger$" if res["bh_significant"] else ""
        v = f"{res['median']:.2f}"
        v = v.replace("-", "$-$", 1) if v.startswith("-") else v
        return f"{v} ({res['wins']}){dag}"

    body = []
    for i, (model, init, label) in enumerate(TEX_ROWS):
        cells = [cell(base["psnr"][(label, "orig")])]
        cells += [cell(per_r[r]["psnr"][(label, "masked")]) for r in radii]
        body.append(f"{model if i % 2 == 0 else ''} & {init} & " + " & ".join(cells) + r" \\")
        if i == 1:
            body.append(r"\midrule")

    frac = [100.0 * float(np.median([x["masked_fraction"] for x in rows_by_r[r]]))
            for r in radii]
    pct = [rf"\SI{{{f:.2f}}}{{\percent}}" for f in frac]
    frac_txt = ", ".join(pct[:-1]) + " and " + pct[-1] if len(pct) > 1 else pct[0]
    radii_txt = (", ".join(f"{r:g}" for r in radii[:-1]) + f" and {radii[-1]:g}"
                 if len(radii) > 1 else f"{radii[0]:g}")

    # Largest change in the median paired difference, any comparison, any radius.
    def max_shift(m):
        return max(abs(per_r[r][m][(lab, "masked")]["median"] - base[m][(lab, "orig")]["median"])
                   for r in radii for _, _, lab in TEX_ROWS)

    others = [m for m in metrics if m != "psnr"]
    def shift_str(v):
        return "less than 0.001" if v < 0.0005 else f"{v:.3f}"

    # tiny ("less than 0.001") entries last, so "less than" cannot read as
    # applying to the whole list
    vals = sorted(((max_shift(m), m) for m in others), key=lambda x: x[0] < 0.0005)
    shifts = [f"{shift_str(v)} in {TEX_METRIC_NAMES[m]}" for v, m in vals]
    shift_txt = ", ".join(shifts[:-1]) + " and " + shifts[-1] if len(shifts) > 1 else shifts[0]

    # Any change in wins or significance, any metric, any comparison, any radius.
    changes = []
    for m in metrics:
        for model, init, lab in TEX_ROWS:
            b = base[m][(lab, "orig")]
            for r in radii:
                q = per_r[r][m][(lab, "masked")]
                if q["wins"] != b["wins"] or q["bh_significant"] != b["bh_significant"]:
                    k = abs(q["wins"] - b["wins"])
                    what = "one target" if k == 1 else rf"\num{{{k}}} targets"
                    if q["bh_significant"] != b["bh_significant"]:
                        what += " and its significance"
                    start = {"Random init": "random initialization",
                             "Simulation-seeded": "simulation seeding"}[init]
                    changes.append(f"{TEX_METRIC_NAMES[m]} under the {model.lower()} model "
                                   f"from {start} at $r={r:g}$, which changes by {what}")
    if not changes:
        change_txt = ("The number of improved targets and the outcome of every "
                      "significance test are unchanged at every radius, for every metric.")
    else:
        change_txt = ("The number of improved targets and the outcome of every "
                      "significance test are unchanged at every radius, for every metric, "
                      "except " + " and ".join(changes) + ".")

    bench = [x for x in rows_by_r[20.0 if 20.0 in rows_by_r else radii[0]]
             if x["source"].startswith("bench")]
    ratios = [x["dc_peak_over_p999"] for x in bench]
    loss_r = gdcitl.DC_RADIUS_NATIVE

    caption = (
        r"\caption{Sensitivity of the comparison between GD+CITL and plain GD to the "
        r"zero order. Each entry is the median paired difference in PSNR (dB) across "
        r"\num{18} targets, GD+CITL minus plain GD, with the number of targets "
        r"improved by camera feedback in parentheses. \emph{Whole frame} is the "
        r"scoring used throughout the main text, with the zero order included. A "
        r"column headed $r$ removes a disc of radius $r$ native pixels, centered on "
        r"the zero order, from both the capture and the target before min-max "
        rf"normalization and scoring. Radii of {radii_txt} remove {frac_txt} "
        rf"of the replay field, and the loss uses $r=\num{{{loss_r:g}}}$. "
        r"$\dagger$ marks survival of "
        r"Benjamini-Hochberg correction at $q=0.05$ across the four comparisons in "
        r"that column. Across all radii and comparisons, the largest change in the "
        rf"median paired difference is {shift_txt}. {change_txt} In every capture "
        r"the brightest pixel of "
        rf"the zero order is between \num{{{min(ratios):.2f}}} and "
        rf"\num{{{max(ratios):.2f}}} times the 99.9th percentile of the rest of the "
        r"frame, so the zero order does not set the min-max scale. As in Table 2 of "
        r"the main text, values are not comparable between forward models.}"
    )

    header = (r" & & \multicolumn{" + str(len(radii) + 1) + r"}{c}{$\Delta$PSNR (dB)} \\")
    lines = [
        r"% GENERATED by analysis/dc_mask_diagnostic.py - do not hand-edit.",
        r"% Destination: Supplement 1.",
        r"\begin{table}[htbp]", r"\centering",
        caption,
        r"\label{tab:zeroOrder}",
        r"\footnotesize",
        r"\begin{tabular}{l l " + "c " * (len(radii) + 1) + "}",
        r"\toprule",
        header,
        rf"\cmidrule(lr){{3-{len(radii) + 3}}}",
        r"Model & Initialization & Whole frame & "
        + " & ".join(f"$r={r:g}$" for r in radii) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    return "\n".join(lines)


def run(radius=gdcitl.DC_RADIUS_NATIVE, out: Path = OUT, perceptual: bool = True,
        tables: Path | None = None) -> dict:
    """Notebook entry point.

    radius: one value or a list, in NATIVE pixels. Each capture is loaded once
    and scored at every radius, so a list costs about the same as one run.
    Writes out/radius_<r>/ for each radius, plus out/radius_sensitivity.txt
    and the Supplement 1 table zero_order.tex when more than one radius is
    given. The table goes to `tables` (default: a "tables" folder next to
    `out`, which in the notebook is analysis/tables, beside citl_metrics.tex).

    Returns {"rows": {r: [...]}, "summary": {r: str}, "sensitivity": str}.
    """
    radii = [float(r) for r in (radius if np.iterable(radius) else [radius])]
    out = Path(out)

    compute_metrics, prepare_target, target_for_pad, load_capture = sc.build_scorer(REPO)
    pfn = None
    if perceptual:
        from perceptual import perceptual_metrics as pfn
    cal = gdcitl.calibration(RESULTS)
    roi = {k: cal["roi"][k] for k in ("y0", "x0", "h", "w")}
    angle = cal["rotation_deg"]
    published = {(r["physics"], r["source"], r["target"]): r
                 for r in json.loads(SCORED.read_text())}

    work = work_list()
    print(f"{len(work)} records, DC radius {', '.join(f'{r:g}' for r in radii)} px native")

    rows_by_r = {r: [] for r in radii}
    tested = False
    for w in work:
        if not (w.get("capture") or w.get("recon_npy")) or not w["target_image"]:
            print(f"  ! skipped {w['physics']} {w['source']} {w['target']} (missing input)")
            continue
        pf = w["pad_factor"]
        tgt = prepare_target(image_path=str(w["target_image"]), device="cpu")
        H, W = tgt["H"], tgt["W"]
        target_eval = target_for_pad(tgt["target_eval_np"], pf)

        if w["source"] == "simulation":
            recon = np.load(w["recon_npy"]).astype(np.float32)
        else:
            I_cam, *_ = load_capture(
                image_path=str(w["capture"]), roi=roi, out_hw=(H * pf, W * pf),
                device="cpu", angle=angle,
                dc_radius=gdcitl.DC_RADIUS_NATIVE * pf, auto_center=False,
                dc_center=(gdcitl.DC_CENTER_NATIVE[0] * pf, gdcitl.DC_CENTER_NATIVE[1] * pf),
                subtract_min=True,
            )
            recon = I_cam[0].detach().cpu().numpy().astype(np.float32)

        if not tested:
            self_test(compute_metrics, recon, target_eval, pf, pfn)
            tested = True

        m0 = compute_metrics(recon, target_eval, pf)
        if pfn is not None:
            m0.update(pfn(recon, target_eval))
        pub = published.get((w["physics"], w["source"], w["target"]))
        repro = abs(m0["psnr"] - pub["psnr"]) if pub else float("nan")

        line = f"  {w['physics']:8s} {w['source']:24s} {w['target']:14s} PSNR {m0['psnr']:6.2f} ->"
        for r in radii:
            mask = disc_mask(recon.shape, pf, r)
            m1 = masked_metrics(recon, target_eval, pf, mask, pfn)
            # how dominant is the DC region in this frame?
            dc_peak_ratio = float(recon[mask].max() / (np.percentile(recon[~mask], 99.9) + EPS))
            row = {"target": w["target"], "physics": w["physics"], "source": w["source"],
                   "pad_factor": pf, "radius_native": r, "masked_fraction": float(mask.mean()),
                   "dc_peak_over_p999": dc_peak_ratio, "repro_abs_err_psnr": repro}
            row.update({f"orig_{k}": v for k, v in m0.items()})
            row.update({f"masked_{k}": v for k, v in m1.items()})
            rows_by_r[r].append(row)
            line += f"  r{r:g}: {m1['psnr']:6.2f}"
        print(line)

    summaries = {}
    for r in radii:
        d = out / f"radius_{r:g}"
        summaries[r] = _write_outputs(rows_by_r[r], r, d)
        print(f"\nwrote {d}")

    sens = ""
    if len(radii) > 1:
        sens = _cross_radius_table(rows_by_r)
        (out / "radius_sensitivity.txt").write_text(sens, encoding="utf-8")
        print("\n" + sens)
        print(f"\nwrote {out / 'radius_sensitivity.txt'}")
        tdir = Path(tables) if tables is not None else out.parent / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        tex = _zero_order_tex(rows_by_r)
        (tdir / "zero_order.tex").write_text(tex, encoding="utf-8")
        print(f"  table -> {tdir / 'zero_order.tex'}")
    else:
        print("\n" + summaries[radii[0]])

    return {"rows": rows_by_r, "summary": summaries, "sensitivity": sens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, nargs="+", default=[gdcitl.DC_RADIUS_NATIVE],
                    help="DC disc radius in NATIVE pixels, one or more (default: 20)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-perceptual", action="store_true", help="skip LPIPS/HaarPSI")
    a = ap.parse_args()
    run(a.radius, a.out, perceptual=not a.no_perceptual)


if __name__ == "__main__":
    main()
