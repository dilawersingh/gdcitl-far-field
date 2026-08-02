r"""
gdcitl.py - shared foundation for every figure in the manuscript.

SCRIPT 1 of the analysis pipeline. Nothing here draws anything; it is the layer
every figure script sits on, so that loading, statistics, terminology and visual
style are defined exactly once and cannot drift between figures.

Three principles, each of which caused a real problem when it was violated
earlier in this project:

  1. READ-ONLY, FROM SOURCE. The results tree is immutable. Nothing in this
     pipeline writes to it, and no figure is ever built from a hand-edited
     intermediate. Point RESULTS_ROOT at the tree and everything regenerates.

  2. TERMINOLOGY IS ENFORCED HERE, NOT PER-FIGURE. The results JSON stores the
     no-feedback condition under keys prefixed `naive_`. In the paper that
     condition is "plain GD". PLAIN_GD_LABEL is the only string that should ever
     reach an axis or a caption - nothing labelled "naive".

  3. MEDIANS, NOT MEANS. Several of these distributions are strongly
     right-skewed across the target set (best-iteration under random-init: mean
     213 vs median 168, driven by three outlier targets). Every summary
     statistic here is a median with a bootstrap interval, and every test is
     paired and non-parametric, because n = 18 with skew does not support
     normality. These choices are fixed in advance, not selected after
     inspecting results.

Configure with the GDCITL_RESULTS environment variable, or pass root= explicitly.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

import paths

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Every default resolves inside the deposit, which paths.py locates from its own
# file position. The environment variables remain as overrides for anyone
# pointing this at their own results tree.
RESULTS_ROOT = Path(os.environ.get("GDCITL_RESULTS", paths.RESULTS))

# The plain-gradient-descent bench captures were acquired in a separate session
# from the camera-in-the-loop runs: those holograms were computed in simulation
# and replayed on the bench. Reusing them as this paper's no-feedback control is
# legitimate because the rig calibration is byte-identical across the two capture
# sessions (see calibration() below) and both used 750 iterations deliberately,
# so the comparison is on equal footing.
REPLAY_ROOT = Path(os.environ.get("GDCITL_REPLAY", paths.REPLAY))

STOCK_ROOT = Path(os.environ.get("GDCITL_STOCK", paths.TARGETS))

# The 18-target set, in a fixed order so every figure and table lists them
# identically. Derived from the results tree rather than hard-coded, but pinned
# to this count as a guard against a partial run being plotted as if complete.
N_TARGETS = 18

# --------------------------------------------------------------------------
# Terminology (see principle 2)
# --------------------------------------------------------------------------
PLAIN_GD_LABEL = "Plain GD"
CITL_LABEL = "GD+CITL"
IDEAL_LABEL = "Ideal model"
FAITHFUL_LABEL = "Faithful model"

# Forward-model settings, taken from run_configuration.json -> conditions.
# Recorded here so a figure can state them without re-reading a config.
FORWARD_MODELS = {
    "ideal":    dict(pad_factor=1, apply_sinc=False, fill_factor=1.0, fill_is_areal=True),
    "faithful": dict(pad_factor=2, apply_sinc=True, fill_factor=0.91, fill_is_areal=True),
}

# Zero-order geometry at NATIVE resolution (run_configuration.json -> hardware).
# Both centre and radius scale by pad_factor; the run logs confirm the faithful
# runs used (990, 1020), i.e. exactly 2x these values.
DC_CENTER_NATIVE = (495, 510)
DC_RADIUS_NATIVE = 20


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def zernike_results(init: str, root: Path | None = None) -> list[dict]:
    """Load one Zernike stress-test run.

    init: "random" | "warm"

    Returns the raw record list: 1,872 records = 18 images x 8 modes
    (7 per-mode + 1 combined) x 13 magnitudes. Note the sweep is 13 magnitudes,
    not the 9 described in zernike_experiment.md, and contains no tip/tilt --
    the documented mode list is stale relative to the data.
    """
    base = Path(root or RESULTS_ROOT) / "simulations" / "zernike_stress_test"
    prefix = {"random": "random_start", "warm": "warm_start"}[init]
    matches = sorted(base.glob(f"{prefix}_*/zernike_stress_results.json"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}_*/zernike_stress_results.json under {base}")
    if len(matches) > 1:
        print(f"  note: {len(matches)} {prefix} runs; using newest ({matches[-1].parent.name})")
    return json.loads(matches[-1].read_text())


def simulation_sweep(physics: str, root: Path | None = None) -> dict:
    """Load the plain-GD simulation sweep (no camera) for one forward model.

    This is the sweep that provides the `simulation` source in scored.json and
    the converged holograms used as the simulation-seeded initialization.
    """
    base = Path(root or RESULTS_ROOT) / "simulations" / f"simulation_comparison_{physics}"
    matches = list(base.glob("*_all_results.json"))
    if not matches:
        raise FileNotFoundError(f"no *_all_results.json under {base}")
    return json.loads(matches[0].read_text())


def citl_runs(init: str, physics: str, root: Path | None = None) -> list[dict]:
    """Locate every CITL run directory for one condition, with its metadata.

    init: "random_init" | "warm_start";  physics: "ideal" | "faithful"

    Each run retained exactly three captures - iteration 0, the best iteration,
    and the last. Only the BEST-iteration capture is a result; iteration 0 is
    one update into the run and is not a plain-GD control.

    Where a target was run more than once, the newest run wins.
    """
    base = Path(root or RESULTS_ROOT) / "experiments" / init / f"citl_gs_gd_{physics}"
    out: dict[str, dict] = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        log = d / "experimental_log.txt"
        if not log.exists():
            continue
        txt = log.read_text(encoding="utf-8", errors="replace")
        m_t = re.search(r"Target:\s*(\S+)", txt)
        m_b = re.search(r"GD:\s*best iteration\s*(\d+)", txt)
        m_i = re.search(r"Image:\s*(.+?)\s*$", txt, re.M)
        if not (m_t and m_b):
            print(f"  ! {d.name}: no target/best-iteration in log, skipped")
            continue
        best = int(m_b.group(1))
        out[m_t.group(1)] = {
            "target": m_t.group(1),
            "run_dir": d,
            "best_iteration": best,
            "target_image": m_i.group(1).strip() if m_i else None,
            "capture_raw": (d / "dslr captures" / f"gd_iter_{best:04d}.jpg"),
            "capture_processed": (d / "processed" / "gd_captures" / f"gd_iter_{best:04d}.png"),
            "init": init,
            "physics": physics,
        }
    return [out[k] for k in sorted(out)]


def run_metadata(init: str, physics: str, root: Path | None = None) -> dict[str, dict]:
    r"""{target: {best_iteration, gd_seconds, seconds_per_iteration, exposure, run_dir}}

    THE RUN DIRECTORIES ARE THE SINGLE SOURCE OF TRUTH. This reads each surviving
    run's own experimental_log.txt, never the condition-level summary.log.

    Why: a target re-run under different settings leaves ONE directory (the newer
    run) but TWO blocks in summary.log, which is append-only. Deriving iteration
    counts from the log while deriving captures from the directories means two
    independent notions of "which run counts", and they can silently disagree --
    a subtly different regex over summary.log picked the older `guy` block (113)
    where the surviving directory holds the newer one (120). Reading only the
    directories makes that class of bug impossible, and leaves summary.log as
    what it should be: a human-readable provenance record that no code depends
    on, and that must NOT be edited to match the folders.

    gd_seconds comes from "GD camera-feedback wall-clock: X hr for N iterations",
    which times the GD loop specifically rather than the whole run (that also
    covers setup and any other stages in the run). Since every GD iteration is one
    physical exposure, seconds_per_iteration converts an iteration count into
    camera time - the quantity that decides whether the method is deployable.
    """
    out: dict[str, dict] = {}
    for r in citl_runs(init, physics, root):
        log = r["run_dir"] / "experimental_log.txt"
        txt = log.read_text(encoding="utf-8", errors="replace")
        rec = {"best_iteration": r["best_iteration"], "run_dir": r["run_dir"],
               "gd_seconds": None, "gd_iterations": None,
               "seconds_per_iteration": None, "exposure": None}
        m_w = re.search(r"GD camera-feedback wall-clock:\s*([\d.]+)\s*hr\s*for\s*(\d+)",
                        txt)
        if m_w:
            rec["gd_seconds"] = float(m_w.group(1)) * 3600.0
            rec["gd_iterations"] = int(m_w.group(2))
            if rec["gd_iterations"]:
                rec["seconds_per_iteration"] = rec["gd_seconds"] / rec["gd_iterations"]
        m_e = re.search(r"Exposure:\s*ISO=(\S+)\s*\([^)]*\),\s*Tv=(\S+)\s*\(", txt)
        if m_e:
            rec["exposure"] = f"ISO={m_e.group(1)} Tv={m_e.group(2)}"
        out[r["target"]] = rec
    return out


def best_iterations(init: str, physics: str, root: Path | None = None) -> dict[str, int]:
    """{target: best_iteration}. Thin wrapper over run_metadata()."""
    return {t: r["best_iteration"]
            for t, r in run_metadata(init, physics, root).items()}


def replay_run(replay_root: Path | None = None) -> Path:
    """Newest replay_simulation_* directory - the plain-GD bench capture set."""
    base = Path(replay_root or REPLAY_ROOT)
    runs = sorted(p for p in base.iterdir()
                  if p.is_dir() and p.name.startswith("replay_simulation_"))
    if not runs:
        raise FileNotFoundError(f"no replay_simulation_* under {base}")
    if len(runs) > 1:
        print(f"  note: {len(runs)} replay runs; using newest ({runs[-1].name})")
    return runs[-1]


def plain_gd_captures(physics: str, replay_root: Path | None = None) -> dict[str, Path]:
    """{target: gd.jpg} - the simulation-optimized hologram displayed on the
    bench and photographed, with no camera feedback anywhere in the loop.

    This is the paper's no-feedback control. Note it is NOT iteration 0 of a CITL
    run: that capture has already been through one update and is not plain GD.
    """
    base = replay_run(replay_root) / physics
    out = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        p = d / "gd.jpg"
        if p.exists():
            out[d.name] = p
    return out


def target_image(target: str, stock_root: Path | None = None) -> Path | None:
    """Resolve a target name to its image file (extensions vary across the set)."""
    base = Path(stock_root or STOCK_ROOT)
    for ext in (".png", ".jpg", ".jpeg"):
        p = base / f"{target}{ext}"
        if p.exists():
            return p
    return None


def calibration(root: Path | None = None) -> dict:
    """Rig-wide alignment: rotate the full frame, crop a fixed ROI, resize.

    The project reverted from homography/warpPerspective to this path on
    2026-07-16 after a matched-condition comparison (citl_capture.py). The
    handoff document still says homography; the run configurations that produced
    the data say manual. This file is the authority, and Methods must describe
    ROI+rotation.
    """
    path = Path(root or RESULTS_ROOT) / "experiments" / "_rig_calibration" / "calibration.json"
    cal = json.loads(path.read_text())
    if cal.get("mode") != "manual":
        raise ValueError(f"expected manual calibration, found {cal.get('mode')!r}")
    return cal


# --------------------------------------------------------------------------
# Statistics (see principle 3)
# --------------------------------------------------------------------------
def bootstrap_median_ci(x, n_boot: int = 10_000, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the median. Seeded, so every rebuild
    of the manuscript produces identical intervals."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return (float(np.percentile(meds, 100 * alpha / 2)),
            float(np.percentile(meds, 100 * (1 - alpha / 2))))


def wilcoxon(a, b) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank - the pre-specified primary test.

    Paired because every condition is measured on the same 18 targets.
    Non-parametric because n=18 with skew does not support a t-test, and a
    referee will ask why a t-test was used if one is.
    """
    from scipy.stats import wilcoxon as _w
    stat, p = _w(np.asarray(a, float), np.asarray(b, float))
    return float(stat), float(p)


def benjamini_hochberg(pvals, q: float = 0.05) -> np.ndarray:
    """BH step-up FDR control, for sweeps that run many simultaneous tests."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, p.size + 1) / p.size)
    cutoff = int(np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
    out = np.zeros(p.size, dtype=bool)
    out[order[:cutoff]] = True
    return out


def summarize(x) -> dict:
    """Median, IQR and bootstrap CI - the standard summary for every table."""
    x = np.asarray(x, dtype=float)
    lo, hi = bootstrap_median_ci(x)
    return {"median": float(np.median(x)), "q1": float(np.percentile(x, 25)),
            "q3": float(np.percentile(x, 75)), "ci_lo": lo, "ci_hi": hi,
            "n": int(x.size)}


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
def check(root: Path | None = None) -> None:
    """Report what is present and what is missing. Run this first."""
    root = Path(root or RESULTS_ROOT)
    print(f"results root: {root}")
    print(f"  exists: {root.exists()}\n")
    if not root.exists():
        return

    for init in ("random_init", "warm_start"):
        for physics in ("ideal", "faithful"):
            try:
                runs = citl_runs(init, physics, root)
                have = sum(1 for r in runs if r["capture_raw"].exists()
                           or r["capture_processed"].exists())
                flag = "" if len(runs) == N_TARGETS else f"  <- expected {N_TARGETS}"
                print(f"  citl {init:11s}/{physics:8s}: {len(runs):2d} runs, "
                      f"{have:2d} best-iteration captures{flag}")
            except (FileNotFoundError, NotADirectoryError) as e:
                print(f"  citl {init:11s}/{physics:8s}: MISSING ({e})")

    for init in ("random", "warm"):
        try:
            print(f"  zernike {init:6s}: {len(zernike_results(init, root)):5d} records")
        except FileNotFoundError as e:
            print(f"  zernike {init:6s}: MISSING ({e})")

    for physics in ("ideal", "faithful"):
        try:
            simulation_sweep(physics, root)
            print(f"  sim sweep {physics:8s}: present")
        except FileNotFoundError as e:
            print(f"  sim sweep {physics:8s}: MISSING ({e})")

    try:
        cal = calibration(root)
        print(f"  calibration: {cal['mode']}, rotation {cal['rotation_deg']} deg, "
              f"roi {cal['roi']['w']}x{cal['roi']['h']}")
    except Exception as e:
        print(f"  calibration: MISSING ({e})")

    # - exposure consistency between the two capture sets.
    # The plain-GD replay and the CITL runs happened on different days and
    # exposure_settings.json changed in between (guy, tram were re-run). Both
    # sides must use the same exposure per target or the comparison is
    # confounded, so this is checked rather than assumed.
    try:
        replay_ex: dict[tuple[str, str], str] = {}
        log = (replay_run() / "experimental_log.txt").read_text(
            encoding="utf-8", errors="replace")
        for line in log.splitlines():
            m = re.match(r"^(ideal|faithful)/(\w+)/gd <-", line.strip())
            e = re.search(r"\|\s*(ISO=\S+)\s+(Tv=\S+)\s*->", line)
            if m and e:
                replay_ex[(m.group(1), m.group(2))] = f"{e.group(1)} {e.group(2)}"

        mismatches = []
        for init in ("random_init", "warm_start"):
            for physics in ("ideal", "faithful"):
                for t, rec in run_metadata(init, physics, root).items():
                    ref = replay_ex.get((physics, t))
                    if ref and rec["exposure"] and ref != rec["exposure"]:
                        mismatches.append((physics, init, t, ref, rec["exposure"]))
        if mismatches:
            print(f"  ** EXPOSURE MISMATCH in {len(mismatches)} cases - the "
                  f"plain-GD and CITL captures are NOT comparable: **")
            for phys, init, t, a, b in mismatches[:8]:
                print(f"       {phys}/{init}/{t}: replay[{a}] vs citl[{b}]")
        else:
            print(f"  exposure match (plain-GD vs CITL, {len(replay_ex)} targets): OK")
    except Exception as e:
        print(f"  exposure check: SKIPPED ({e})")

    # - plain-GD bench captures (the no-feedback control, in the other tree)
    try:
        run = replay_run()
        print(f"  replay run: {run.name}")
        for physics in ("ideal", "faithful"):
            caps = plain_gd_captures(physics)
            flag = "" if len(caps) == N_TARGETS else f"  <- expected {N_TARGETS}"
            print(f"  plain-GD {physics:8s}: {len(caps):2d} captures{flag}")
        # Cross-session comparability: the plain-GD captures and the CITL
        # captures come from different sessions, so this must match or the
        # comparison is confounded.
        rc = json.loads((run / "calibration.json").read_text())
        same = rc == calibration(root)
        print(f"  replay calibration matches CITL rig: "
              f"{'YES' if same else 'NO - captures are NOT comparable'}")
    except Exception as e:
        print(f"  plain-GD captures: MISSING ({e})")

    # - target images. Enumerated from the capture set, so if that could not be
    # listed we report NOT CHECKED rather than falsely reporting all-clear on an
    # empty loop.
    try:
        names = sorted(plain_gd_captures("ideal"))
    except Exception:
        names = []
    if not names:
        print("  target images: NOT CHECKED (no capture list to check against)")
    else:
        missing = [t for t in names if target_image(t) is None]
        if missing:
            print(f"  ! no target image for: {', '.join(missing)}")
        else:
            print(f"  target images: all {len(names)} resolved")


if __name__ == "__main__":
    check()
