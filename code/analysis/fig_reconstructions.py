r"""
fig_reconstructions.py - SCRIPT 7. What the improvement actually looks like.

Two layouts, both rendered for every target.

  ROI layout (1x4) - the manuscript figure
      target (full frame, green ROI box) | target ROI | plain GD ROI | GD+CITL ROI

      The zoom is not decoration. At column width a 1000x1000 reconstruction
      renders about an inch across, and the difference being shown is speckle and
      fine detail - invisible at that size. The full frame on the left
      establishes what the scene is and where the crop came from; the three crops
      carry the comparison.

  FULL layout (1x3) - full-scale comparison
      target | plain GD | GD+CITL, whole frame

Outputs:
    review_<physics>/roi/<target>.{png,pdf}    ROI layout
    review_<physics>/full/<target>.{png,pdf}   full frame
    grid/<target>.{png,pdf}                    2x2, method x forward model

One file per target per layout. Pick what the manuscript needs from these rather
than having the script assemble a montage, so what is published is literally a
file you looked at.

------------------------------------------------------------------------------
FOUR CONSIDERATIONS, each fixed here rather than left to care.
------------------------------------------------------------------------------
1. DISPLAY SCALE. Every panel in a row shares ONE scale, computed jointly over
   the plain-GD and GD+CITL captures for that target. Per-panel normalization
   would let either method be flattered by its own scale and would make a dim
   capture look identical to a bright one. Brightness differences stay visible.

2. ZERO ORDER. The panels show the camera capture AS CAPTURED, zero order
   included - these are direct comparisons of what the sensor recorded, and
   masking the DC would leave a hole in the middle of every reconstruction that
   is not a property of the system.

   What the DC region is excluded from is the PERCENTILE THAT SETS THE DISPLAY
   SCALE. The zero-order spike is orders of magnitude above the image content, so
   letting it participate in the scale would crush everything else toward black.
   Excluding it from the scale and then clipping means the zero order renders as
   the saturated spot it actually is, while the image content stays legible. The
   same treatment is applied to both captures, so neither is favoured.

3. RESAMPLING. Full frames are DOWNsampled for display and use antialiased
   resampling; zooms are UPsampled and use nearest. Downsampling speckle by
   nearest-neighbour aliases it, and the aliasing pattern differs between two
   panels showing genuinely similar content - which would manufacture a visible
   difference that is not in the data. This distinction is not cosmetic.

4. ROI SELECTION. Crops come from roi.json, one entry per target, each entirely
   independent - setting one never affects another, and an unlisted target is
   shown at full frame rather than being cropped to somebody else's region.
   Choose each crop by looking at the TARGET, on where the detail is, never by
   flipping between the two reconstructions (which would let you pick the region
   where CITL happens to win). roi.json itself is the record - version it with
   the paper repo.

PSNR INSETS are read from scored.json, never recomputed, so an inset cannot
disagree with a table. This is the one figure where numbers are drawn on an
image: here a number labels the panel it sits on rather than being a statistic
read off an axis, and it is the convention for reconstruction montages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import figstyle
import gdcitl
from figstyle import C_INK, C_MUTED

# WHICH CITL VARIANT THE FIGURES SHOW, in one place.
#
# Random initialization, NOT simulation-seeded. The seeded run begins from the
# very plain-GD hologram these figures place beside it, so a seeded panel shows
# aberration correction and 750 further iterations of ordinary descent at once,
# and a reader cannot tell which produced the difference they are looking at.
# Random init is budget-matched against the baseline and answers the same
# question: +1.82 dB against plain GD under the faithful model, versus +1.80 dB
# seeded. The tables are unaffected - they read scored.json, which carries all
# four sources.
CITL_SOURCE = "bench_citl_random_init"

PANELS = (("target", "Target"),
          ("bench_plain_gd", "Plain GD"),
          (CITL_SOURCE, "GD+CITL"))

DISPLAY_PERCENTILE = 99.5

# Annotation colour, not a series colour: it is drawn OVER greyscale imagery, so
# it is chosen for luminance contrast against a dark field rather than from the
# categorical palette.
ROI_BOX_COLOUR = "#2ee06a"

# Per-target ROI, loaded from a JSON file (see load_roi). Values are FRACTIONS
# of the frame, [y0, x0, side], so one entry works at any pad_factor.
#
#   {
#     "landscape": [0.30, 0.42, 0.18],
#     "face":      [0.22, 0.38, 0.16]
#   }
#
# A target NOT listed is shown at FULL FRAME - no zoom. Each target is entirely
# independent; setting one never affects another.
ROI: dict[str, tuple[float, float, float]] = {}
ROI_FILENAME = "roi.json"


def load_roi(path):
    """Load per-target ROIs from JSON, replacing anything currently set."""
    global ROI
    path = Path(path)
    ROI = {k: tuple(v) for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
    print(f"  ROI file: {path}  ({len(ROI)} target(s) specified, rest full frame)")
    return ROI


# --------------------------------------------------------------------------
def _roi_pixels(target: str, shape, **_):
    """(y0, x0, side, source) for one target.

    Listed in ROI -> that crop. Not listed -> the FULL FRAME, so an unspecified
    target is never silently zoomed into some other target's region.
    """
    H, W = shape
    if target not in ROI:
        side = min(H, W)
        return (H - side) // 2, (W - side) // 2, side, "full"
    y0f, x0f, sf = ROI[target]
    side = int(round(sf * min(H, W)))
    y0 = max(0, min(int(round(y0f * H)), H - side))
    x0 = max(0, min(int(round(x0f * W)), W - side))
    return y0, x0, side, "manual"


def _load_capture(path, roi, angle, out_hw, pad_factor, repo):
    import sys
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from raw_camera_processor import load_camera_capture_for_citl
    except ImportError:
        from citl_capture import load_camera_capture_for_citl

    I_cam, _, dc_mask, _ = load_camera_capture_for_citl(
        image_path=str(path), roi=roi, out_hw=out_hw, device="cpu", angle=angle,
        dc_radius=gdcitl.DC_RADIUS_NATIVE * pad_factor, auto_center=False,
        dc_center=(gdcitl.DC_CENTER_NATIVE[0] * pad_factor,
                   gdcitl.DC_CENTER_NATIVE[1] * pad_factor),
        subtract_min=True,
    )
    return (I_cam[0].numpy().astype(np.float32),
            dc_mask.numpy().astype(bool))


def _load_target(target, pad_factor, repo):
    import sys
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from evaluate_methods import prepare_target_arrays, _target_eval_for_padding
    p = gdcitl.target_image(target)
    if p is None:
        return None
    t = prepare_target_arrays(image_path=str(p), device="cpu")
    return _target_eval_for_padding(t["target_eval_np"], pad_factor)


def render_rows(targets, physics, repo, citl_source=CITL_SOURCE):
    """Load every panel and compute the ONE shared scale per target."""
    cal = gdcitl.calibration()
    roi = {k: cal["roi"][k] for k in ("y0", "x0", "h", "w")}
    angle = cal["rotation_deg"]
    pf = gdcitl.FORWARD_MODELS[physics]["pad_factor"]

    plain_caps = gdcitl.plain_gd_captures(physics)
    citl = {r["target"]: r for r in gdcitl.citl_runs(
        citl_source.replace("bench_citl_", ""), physics)}

    rows = []
    for t in targets:
        tgt = _load_target(t, pf, repo)
        if tgt is None or t not in plain_caps or t not in citl:
            print(f"  ! {t}: missing input, skipped")
            continue
        H, W = tgt.shape
        p_img, dc = _load_capture(plain_caps[t], roi, angle, (H, W), pf, repo)
        c = citl[t]
        cap = c["capture_raw"] if c["capture_raw"].exists() else c["capture_processed"]
        c_img, _ = _load_capture(cap, roi, angle, (H, W), pf, repo)

        keep = ~dc
        vmax = float(np.percentile(np.concatenate([p_img[keep], c_img[keep]]),
                                   DISPLAY_PERCENTILE))
        rows.append({"target": t, "dc": dc, "vmax": max(vmax, 1e-9),
                     "target_img": tgt, "plain": p_img, "citl": c_img,
                     "best_iteration": c["best_iteration"],
                     "shape": (H, W)})
        y0, x0, side, how = _roi_pixels(t, (H, W))
        rows[-1]["roi"] = (y0, x0, side)
        rows[-1]["roi_source"] = how
        print(f"  rendered {t:16s} roi={how}: y0={y0} x0={x0} side={side}")
    return rows


# --------------------------------------------------------------------------
def _show(ax, img, vmax, zoom=False):
    r"""Render one panel.

    The image is shown as captured - zero order included (see note 2). `vmax`
    was computed with the DC region excluded, so clipping renders the spike as
    saturated white rather than letting it set the scale.

    See note 3 on interpolation: downsampled full frames must NOT use nearest.
    """
    ax.imshow(np.clip(np.asarray(img, dtype=np.float32) / vmax, 0, 1),
              cmap="gray", vmin=0, vmax=1,
              interpolation="nearest" if zoom else "antialiased")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _psnr(scored, target, physics, source):
    for r in scored:
        if (r["target"] == target and r["physics"] == physics
                and r["source"] == source):
            return r["psnr"]
    return None



def _title(ax, label, psnr=None, fontsize=6.5, pad=3):
    """Panel label, with PSNR in the title rather than overlaid on the image --
    an inset sits on unpredictable content and can be unreadable depending on
    the crop.

    fontsize is passed in, not fixed here, because these layouts are authored at
    different widths and are all placed at \\linewidth. See figstyle.title_size.
    """
    txt = label if psnr is None else f"{label}\nPSNR {psnr:.2f} dB"
    ax.set_title(txt, fontsize=fontsize, color=C_INK, pad=pad, linespacing=1.35)


def _panel_source(row, src):
    """(image, vmax) for one panel. The target has no zero order and so gets its
    own scale; the two captures SHARE one, computed jointly (note 1)."""
    if src == "target":
        return (row["target_img"],
                float(np.percentile(row["target_img"], DISPLAY_PERCENTILE)))
    return (row["plain"] if src == "bench_plain_gd" else row["citl"]), row["vmax"]


def draw_roi_panel(row, scored, physics, out_path, width=9.0, dpi=400):
    r"""1x4: target full frame with the ROI box, then the three ROI crops."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figstyle.apply()
    fs = figstyle.title_size(width)
    y0, x0, side = row["roi"]
    full_frame = side >= min(row["shape"])
    fig, axes = plt.subplots(1, 4, figsize=(width, width / 4 * 1.14), squeeze=False)
    axes = axes[0]

    img, vmax = _panel_source(row, "target")
    _show(axes[0], img, vmax, zoom=False)
    if not full_frame:
        axes[0].add_patch(Rectangle((x0, y0), side, side, fill=False,
                                    edgecolor=ROI_BOX_COLOUR, linewidth=1.0))
    _title(axes[0], "Target", fontsize=fs, pad=5)

    suffix = "" if full_frame else " (ROI)"
    for j, (src, label) in enumerate(PANELS, start=1):
        img, vmax = _panel_source(row, src)
        _show(axes[j], img[y0:y0 + side, x0:x0 + side], vmax, zoom=not full_frame)
        for sp in axes[j].spines.values():
            sp.set_visible(True); sp.set_color(ROI_BOX_COLOUR); sp.set_linewidth(1.0)
        _title(axes[j], f"{label}{suffix}",
               None if src == "target" else _psnr(scored, row["target"], physics, src), fontsize=fs, pad=5)

    fig.subplots_adjust(wspace=0.03)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_path}.pdf")
    fig.savefig(f"{out_path}.png", dpi=dpi)
    plt.close(fig)


def draw_full_panel(row, scored, physics, out_path, width=9.0, dpi=400):
    """1x3 at full frame: target, plain GD, GD+CITL, whole captures as recorded."""
    import matplotlib.pyplot as plt

    figstyle.apply()
    fs = figstyle.title_size(width)
    fig, axes = plt.subplots(1, 3, figsize=(width, width / 3 * 1.14), squeeze=False)
    axes = axes[0]
    for j, (src, label) in enumerate(PANELS):
        img, vmax = _panel_source(row, src)
        _show(axes[j], img, vmax, zoom=False)
        _title(axes[j], label,
               None if src == "target" else _psnr(scored, row["target"], physics, src), fontsize=fs, pad=5)
    fig.subplots_adjust(wspace=0.03)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_path}.pdf")
    fig.savefig(f"{out_path}.png", dpi=dpi)
    plt.close(fig)


def run(scored_path, out_dir, repo, physics="faithful", select=None,
        citl_source=CITL_SOURCE, review_width=9.0,
        review_dpi=400, roi_file=None) -> list[dict]:
    out_dir = Path(out_dir)
    roi_dir = out_dir / f"review_{physics}" / "roi"
    full_dir = out_dir / f"review_{physics}" / "full"
    for d in (roi_dir, full_dir):
        d.mkdir(parents=True, exist_ok=True)

    load_roi(Path(roi_file) if roi_file else out_dir / ROI_FILENAME)

    scored = json.loads(Path(scored_path).read_text())
    targets = sorted({r["target"] for r in scored})
    if select:
        # select limits EVERYTHING that gets rendered, not just the montage --
        # each target loads two ~7 MB raw captures, so rendering all 18 when you
        # asked for one is minutes of wasted work.
        missing = [t for t in select if t not in targets]
        if missing:
            print(f"  ! not in scored.json, ignored: {', '.join(missing)}")
        targets = [t for t in targets if t in set(select)]
    print(f"  rendering {len(targets)} target(s)")
    rows = render_rows(targets, physics, Path(repo), citl_source)

    for row in rows:
        draw_roi_panel(row, scored, physics, roi_dir / row["target"],
                       width=review_width, dpi=review_dpi)
        draw_full_panel(row, scored, physics, full_dir / row["target"],
                        width=review_width, dpi=review_dpi)
    print(f"  {len(rows)} ROI panels  -> {roi_dir}")
    print(f"  {len(rows)} full panels -> {full_dir}")

    return rows



# --------------------------------------------------------------------------
# 2x2 grid: {plain GD, GD+CITL} x {ideal, faithful}
# --------------------------------------------------------------------------
GRID_CELLS = (("bench_plain_gd", "ideal",    "Plain GD, ideal"),
              ("bench_plain_gd", "faithful", "Plain GD, faithful"),
              (CITL_SOURCE,      "ideal",    "GD+CITL, ideal"),
              (CITL_SOURCE,      "faithful", "GD+CITL, faithful"))


def render_grid_rows(targets, repo, citl_source=CITL_SOURCE):
    r"""Rows carrying BOTH forward models for each target.

    Display scale is computed PER MODEL, not across all four cells. The arrays
    are sum-normalized to (H*W), which differs between pad_factor 1 and 2, so a
    single shared scale would compare quantities that are not on the same
    footing. Within a model the two captures still share one scale, as before,
    so neither method is flattered.
    """
    per = {ph: {r["target"]: r for r in render_rows(targets, ph, repo, citl_source)}
           for ph in ("ideal", "faithful")}
    rows = []
    for t in targets:
        if t not in per["ideal"] or t not in per["faithful"]:
            print(f"  ! {t}: missing in one model, skipped")
            continue
        rows.append({"target": t, "ideal": per["ideal"][t], "faithful": per["faithful"][t]})
    return rows


def draw_grid_panel(row, scored, out_path, width=7.0, dpi=400, annotate=True,
                    use_roi=False):
    r"""One target as a 2x2: rows are method, columns are forward model.

    READ DOWN A COLUMN, NOT ACROSS. Within a column both captures are scored at
    the same resolution against the same target preparation, so the comparison
    is sound. Across columns the ideal condition scores a native
    1000x1000 target and the faithful one scores it upsampled to 2000x2000 --
    different histogram, hence different min-max normalization, and a different
    SSIM window relative to image content. The two columns are two independent
    comparisons placed side by side, not one comparison spanning four cells.
    """
    import matplotlib.pyplot as plt

    figstyle.apply()
    fs = figstyle.title_size(width)
    fig, axes = plt.subplots(2, 2, figsize=(width, width * 1.04), squeeze=False)

    for k, (src, phys, label) in enumerate(GRID_CELLS):
        ax = axes[k // 2][k % 2]
        sub = row[phys]
        # The grid exists to compare whole reconstructions across conditions, so
        # it shows the FULL FRAME by default. Pass use_roi=True to crop instead.
        if use_roi:
            y0, x0, side = sub["roi"]
            full_frame = side >= min(sub["shape"])
        else:
            y0, x0, side, full_frame = 0, 0, min(sub["shape"]), True
        img, vmax = _panel_source(sub, src)
        _show(ax, img[y0:y0 + side, x0:x0 + side], vmax, zoom=not full_frame)
        _title(ax, f"{label}{'' if full_frame else ' (ROI)'}",
               _psnr(scored, row["target"], phys, src) if annotate else None, fontsize=fs, pad=5)

    fig.subplots_adjust(wspace=0.03, hspace=0.26)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_path}.pdf")
    fig.savefig(f"{out_path}.png", dpi=dpi)
    plt.close(fig)


def run_grid(scored_path, out_dir, repo, select=None,
             citl_source=CITL_SOURCE, width=7.0, dpi=400,
             roi_file=None, use_roi=False) -> list[dict]:
    """2x2 grid per target. Same conventions as run(): roi.json for crops,
    unlisted targets at full frame, select limits what renders."""
    out_dir = Path(out_dir)
    grid_dir = out_dir / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    load_roi(Path(roi_file) if roi_file else out_dir / ROI_FILENAME)

    scored = json.loads(Path(scored_path).read_text())
    targets = sorted({r["target"] for r in scored})
    if select:
        missing = [t for t in select if t not in targets]
        if missing:
            print(f"  ! not in scored.json, ignored: {', '.join(missing)}")
        targets = [t for t in targets if t in set(select)]
    print(f"  rendering {len(targets)} target(s), both models")

    rows = render_grid_rows(targets, Path(repo), citl_source)
    for row in rows:
        draw_grid_panel(row, scored, grid_dir / row["target"],
                        width=width, dpi=dpi, use_roi=use_roi)
    print(f"  {len(rows)} grid panels -> {grid_dir}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, default=Path("scored.json"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--repo", type=Path,
                    default=Path.home() / "Documents/GitHub/holography")
    ap.add_argument("--physics", default="faithful", choices=["faithful", "ideal"])
    ap.add_argument("--select", nargs="*", default=None)
    a = ap.parse_args()
    run(a.scored, a.out, a.repo, a.physics, a.select)


if __name__ == "__main__":
    main()
