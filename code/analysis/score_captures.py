r"""
score_captures.py - SCRIPT 2. Builds the single scored dataset the paper rests on.

Produces one tidy record per (target, forward model, source), where source is:

    simulation      the GD hologram scored against its own simulated replay
    bench_plain_gd  that same hologram displayed on the SLM and photographed
    bench_citl      the GD+CITL hologram at its best iteration, photographed

Those three sources are exactly the paper's argument:
    simulation vs bench_plain_gd   ->  the simulation-to-experiment gap  (3.1)
    ideal vs faithful              ->  how much of the gap is modelling  (3.1)
    bench_plain_gd vs bench_citl   ->  what camera feedback recovers     (3.3)

ONE SCORING FUNCTION. Every metric in this paper comes from
`evaluate_methods.compute_metrics` - the project's single scoring function,
imported here, never reimplemented. See build_scorer() for the one place where
this script does something the stock harness does not (resolution handling), and
why that is necessary for the simulation-vs-experiment comparison to mean
anything.

WHY EVERY METRIC IS RECOMPUTED rather than read from stored JSON: the delivered
simulation sweeps were scored BEFORE the diffraction-efficiency support
threshold was made resolution-invariant. Under pad_factor=2 the padded target's
pixels fell below the old fixed 1e-6 threshold, the support mask emptied, and
efficiency silently read 0.0 - for 132 of 198 faithful GD entries, but not all
of them, which makes it look like a result rather than a bug. The current
compute_metrics is correct; the stored numbers predate it. Re-scoring from the
saved arrays is what brings them back into line, so stored metrics are never
read.

CROSS-SESSION COMPARABILITY. The plain-GD bench captures come from a replay run
on 2026-07-24; the CITL captures from 2026-07-17..20. Their calibrations are
byte-identical (manual, ROI 3267x3265 at (1083,129), rotation -0.06 deg), and the
replay run's own configuration states the intent: "The rig is locked and
calibrated; these values must not diverge from the CITL runs or the captures stop
being comparable." Both capture sets are additionally scored HERE with the same
DC exclusion and the same target preparation, so no asymmetry can enter at
scoring time even though the replay run's hardware block carries no DC entries.

Usage:
    python score_captures.py --dry-run                 # inventory only
    python score_captures.py --repo /path/to/holography --out scored.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import gdcitl
import paths

# Where the plain-gradient-descent bench captures live. Acquired in a separate
# session from the camera-in-the-loop runs; see the note in gdcitl.py.
REPLAY_ROOT_DEFAULT = gdcitl.REPLAY_ROOT

# Iteration at which GD is reported, matched across simulation and bench.
GD_ITERATIONS = 750


# --------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------
def replay_run(replay_root: Path) -> Path:
    """Newest replay_simulation_* run directory."""
    runs = sorted(p for p in Path(replay_root).iterdir()
                  if p.is_dir() and p.name.startswith("replay_simulation_"))
    if not runs:
        raise FileNotFoundError(f"no replay_simulation_* under {replay_root}")
    if len(runs) > 1:
        print(f"  note: {len(runs)} replay runs; using newest ({runs[-1].name})")
    return runs[-1]


def simulation_recon(results_root: Path, physics: str, target: str) -> Path | None:
    r"""Saved replay array for GD at GD_ITERATIONS, from the simulation sweep.

    Recomputing from this array rather than reading the sweep's stored metrics is
    what removes the stale-efficiency problem from the paper.

    Layout (verified against the delivered tree):
        <target>_<experiment name>_<timestamp>/
            outputs/gd/gd_iter_750_recon.npy

    IMPORTANT - it must be `outputs/gd/`, never `outputs/gd_intensity_target/`.
    The latter is the additional intensity-target formulation sweep; scoring it
    here would silently substitute a different experimental condition. The
    candidate list below is therefore explicit rather than a `gd*` glob, which
    would match both.
    """
    base = Path(results_root) / "simulations" / f"simulation_comparison_{physics}"
    if not base.exists():
        return None
    for d in sorted(base.glob(f"{target}_simulation_comparison*")):
        for rel in (Path("outputs") / "gd", Path("gd")):   # current, then legacy
            p = d / rel / f"gd_iter_{GD_ITERATIONS}_recon.npy"
            if p.exists():
                return p
    return None


def target_path(target: str, stock_root: Path) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = Path(stock_root) / f"{target}{ext}"
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def build_scorer(repo: Path):
    r"""Import the project's OWN scoring functions. Nothing is reimplemented here.

    `evaluate_methods.compute_metrics` is the single scoring function for the
    whole project - simulation and experiment, every method, both manuscripts.
    This script imports it rather than defining its own so that a change there
    propagates here automatically and the two can never diverge.

    RESOLUTION CONSISTENCY - the reason this script does not simply call
    `evaluate_methods.evaluate_camera_capture_once`:

        that function calls `compute_metrics(recon_np, target_eval_np)` with NO
        pad_factor, and its callers pass out_hw=(H, W). So the experimental path
        scores captures at native 1000x1000 / pad_factor=1, while the simulation
        path scores faithful results at 2000x2000 / pad_factor=2.

    PSNR and (especially) SSIM are resolution-dependent, so scoring the two sides
    of the simulation-vs-experiment comparison at different resolutions would
    make that comparison meaningless. This script therefore resizes captures to
    (H*pad_factor, W*pad_factor) and passes pad_factor into compute_metrics, so
    BOTH sides of every comparison sit at the same resolution. The scoring
    function is shared; only the resolution handling is made explicit here.

    The assertion below is a guard for the case where this script is run against
    an older checkout of the repo - the fix is present in the current
    evaluate_methods.py, but the stored simulation sweeps predate it.
    """
    sys.path.insert(0, str(repo))
    from evaluate_methods import (compute_metrics, prepare_target_arrays,
                                  _target_eval_for_padding)
    # Camera loading lives in raw_camera_processor (evaluate_methods imports it
    # from there); citl_capture is the same code under an older name.
    try:
        from raw_camera_processor import load_camera_capture_for_citl
    except ImportError:
        from citl_capture import load_camera_capture_for_citl
    import inspect

    if not any(s in inspect.getsource(compute_metrics)
               for s in ("pad_factor ** 2", "pad_factor**2")):
        raise RuntimeError(
            "compute_metrics lacks the resolution-invariant support fix\n"
            "  (support = target_sum > 1e-6 / pad_factor**2).\n"
            "  This checkout predates the fix; update before scoring."
        )
    return compute_metrics, prepare_target_arrays, _target_eval_for_padding, \
        load_camera_capture_for_citl


def score_all(results_root: Path, replay_root: Path, stock_root: Path,
              repo: Path, out_path: Path, dry_run: bool) -> None:
    run = replay_run(replay_root)
    print(f"replay run: {run.name}")

    # Assemble the work list first, so a missing input is reported up front
    # rather than discovered halfway through a long scoring pass.
    work: list[dict] = []
    for physics in ("ideal", "faithful"):
        pf = gdcitl.FORWARD_MODELS[physics]["pad_factor"]

        for init in ("random_init", "warm_start"):
            for r in gdcitl.citl_runs(init, physics, results_root):
                cap = (r["capture_raw"] if r["capture_raw"].exists()
                       else r["capture_processed"] if r["capture_processed"].exists()
                       else None)
                # Resolve the target from stock_root, exactly as the other
                # three sources below do - NOT from the run log's "Image:"
                # line. That line records the absolute path on the machine
                # that acquired the data, so it cannot resolve anywhere else,
                # and in this deposit it has been rewritten to a deposit-
                # relative form that only opens from the deposit root. Either
                # way it is provenance, not a usable handle. Resolving all
                # four sources the same way also guarantees they score against
                # the same file.
                work.append({"target": r["target"], "physics": physics, "pad_factor": pf,
                             "source": f"bench_citl_{init}", "capture": cap,
                             "best_iteration": r["best_iteration"],
                             "target_image": str(target_path(r["target"], stock_root) or "")})

        for d in sorted((run / physics).iterdir()):
            if not d.is_dir():
                continue
            gd = d / "gd.jpg"
            work.append({"target": d.name, "physics": physics, "pad_factor": pf,
                         "source": "bench_plain_gd",
                         "capture": gd if gd.exists() else None,
                         "best_iteration": GD_ITERATIONS,
                         "target_image": str(target_path(d.name, stock_root) or "")})
            npy = simulation_recon(results_root, physics, d.name)
            work.append({"target": d.name, "physics": physics, "pad_factor": pf,
                         "source": "simulation", "recon_npy": npy, "capture": None,
                         "best_iteration": GD_ITERATIONS,
                         "target_image": str(target_path(d.name, stock_root) or "")})

    # --- inventory
    print(f"\n{len(work)} records planned")
    for src in sorted({w["source"] for w in work}):
        sub = [w for w in work if w["source"] == src]
        ok = sum(1 for w in sub if w.get("capture") or w.get("recon_npy"))
        miss = [w["target"] for w in sub if not (w.get("capture") or w.get("recon_npy"))]
        print(f"  {src:26s} {ok:3d}/{len(sub):3d} inputs found"
              + (f"   MISSING: {', '.join(sorted(set(miss)))}" if miss else ""))
    bad_target = sorted({w["target"] for w in work if not w["target_image"]})
    if bad_target:
        print(f"  ! no target image resolved for: {', '.join(bad_target)}")

    if dry_run:
        print("\ndry run - nothing scored")
        return

    compute_metrics, prepare_target, target_for_pad, load_capture = build_scorer(repo)
    cal = gdcitl.calibration(results_root)
    roi = {k: cal["roi"][k] for k in ("y0", "x0", "h", "w")}
    angle = cal["rotation_deg"]

    out: list[dict] = []
    for w in work:
        if not (w.get("capture") or w.get("recon_npy")) or not w["target_image"]:
            continue
        pf = w["pad_factor"]
        tgt = prepare_target(image_path=w["target_image"], device="cpu")
        H, W = tgt["H"], tgt["W"]
        target_eval = target_for_pad(tgt["target_eval_np"], pf)

        if w["source"] == "simulation":
            recon = np.load(w["recon_npy"]).astype(np.float32)
        else:
            # Both bench sources go through the SAME calibration and the SAME
            # DC exclusion, scaled by pad_factor. The run logs confirm the
            # faithful runs used dc_center (990, 1020) = 2x native.
            I_cam, *_ = load_capture(
                image_path=str(w["capture"]), roi=roi, out_hw=(H * pf, W * pf),
                device="cpu", angle=angle,
                dc_radius=gdcitl.DC_RADIUS_NATIVE * pf,
                auto_center=False,
                dc_center=(gdcitl.DC_CENTER_NATIVE[0] * pf,
                           gdcitl.DC_CENTER_NATIVE[1] * pf),
                subtract_min=True,
            )
            recon = I_cam[0].detach().cpu().numpy().astype(np.float32)

        m = compute_metrics(recon, target_eval, pf)
        out.append({"target": w["target"], "physics": w["physics"],
                    "source": w["source"], "best_iteration": w["best_iteration"],
                    "pad_factor": pf, **m})
        print(f"  {w['physics']:8s} {w['source']:26s} {w['target']:14s} "
              f"PSNR={m['psnr']:6.2f} SSIM={m['ssim']:.3f} eff={m['diffraction_efficiency']:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}  ({len(out)} records)")

    # The efficiency bug is silent; check the OUTPUT as well as the input.
    for physics in ("ideal", "faithful"):
        eff = [r["diffraction_efficiency"] for r in out if r["physics"] == physics]
        if eff and all(e == 0.0 for e in eff):
            print(f"  ** WARNING: all {physics} efficiencies are 0.0 - still the bug **")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=gdcitl.RESULTS_ROOT)
    ap.add_argument("--replay", type=Path, default=REPLAY_ROOT_DEFAULT)
    ap.add_argument("--stock", type=Path, default=paths.TARGETS)
    # evaluate_methods.py and raw_camera_processor.py live in code/method/,
    # not code/. The old default put code/ on sys.path, where neither is
    # importable, so a standalone CLI run died before scoring anything. The
    # notebook always passed the right value, which is why it went unnoticed.
    ap.add_argument("--repo", type=Path, default=paths.ROOT / "code" / "method")
    ap.add_argument("--out", type=Path, default=Path("scored.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    score_all(a.results, a.replay, a.stock, a.repo, a.out, a.dry_run)


if __name__ == "__main__":
    main()