"""
evaluate_methods.py - simulation and experimental evaluation harness for
gradient-descent hologram generation under the shared Fraunhofer forward model
in physics.py.

This is the single file that scores every quantitative result reported in the
manuscript. Simulation and bench results are scored identically: the predicted
phase is propagated through the same forward model
(physics.hologram_intensity_from_phase), re-normalized identically
(stats_torch.normalize_intensity_sum), and scored with the same metric function
(compute_metrics below). Any difference in reported PSNR, SSIM or NMSE therefore
reflects a difference in the hologram itself, not a difference in how it was
evaluated.

  IDEAL AND FAITHFUL RESULTS ARE NEVER COMPARABLE WITH ONE ANOTHER. The faithful
  condition scores a target bicubically upsampled to the padded resolution; the
  ideal condition scores it at native resolution. Different resolutions mean
  different min-max normalization and a different SSIM window relative to image
  content. Each condition is a self-contained comparison in which only the
  hologram differs.

Target formulation:
  Gradient descent is driven by an amplitude target, sqrt(intensity), following
  the convention of the published comparison. gd.run squares it to recover the
  intensity training target, so the loss is formed in intensity either way, and
  the `target_formulation` parameter carried through `prepare_target_arrays` and
  `save_target_artifacts` describes what the caller is passing in rather than the
  domain the loss is evaluated in.

Scope:
  This file covers the gradient-descent method reported in this paper. It is
  maintained upstream as part of a larger evaluation harness; other methods are
  removed here rather than never having existed, so function names and some
  parameters retain a generality this deposit does not exercise.
"""
import os
import time
import json
import math
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from gd import GradientDescentHologram
from physics import hologram_intensity_from_phase
from stats_torch import normalize_intensity_sum
from raw_camera_processor import (
    load_camera_capture_for_citl,
    no_dc_load_camera_capture_for_citl,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_grayscale_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.asarray(img, dtype=np.float32)


def save_display_image(arr: np.ndarray, path: str):
    """Min-max normalize to [0,255] for visualization only - not used for any
    quantitative metric (see compute_metrics, which does its own normalization
    independently for scoring)."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    img = (255.0 * arr).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def _target_eval_for_padding(target_eval_np: np.ndarray, pad_factor: int) -> np.ndarray:
    """
    Return target_eval_np at the resolution compute_metrics needs to match
    recon_np: unchanged if pad_factor==1, bicubic-upsampled to
    (pad_factor*H, pad_factor*W) otherwise. Used identically by
    every sweep and capture path in this file, so all
    three upsample the SAME way whenever pad_factor>1 - kept as a single
    helper rather than duplicated inline in three places so they cannot drift.
    """
    if pad_factor <= 1:
        return target_eval_np
    H, W = target_eval_np.shape
    t = torch.from_numpy(target_eval_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t_up = torch.nn.functional.interpolate(
        t, size=(H * pad_factor, W * pad_factor), mode="bicubic", align_corners=False
    )
    return t_up.squeeze(0).squeeze(0).numpy().astype(np.float32)


def compute_metrics(recon: np.ndarray, target: np.ndarray, pad_factor:int=1) -> dict:
    """
    The single scoring function used for every result in the paper
    in both simulation and experiment. recon/target are independently normalized
    here (both min-max to [0,1] for PSNR/SSIM, both sum-normalized for
    NMSE/diffraction efficiency) - this function does not assume its inputs
    arrive pre-normalized in any particular convention, so it produces consistent
    metrics regardless of which upstream normalization a given caller used before
    this point.
    """
    recon = np.asarray(recon, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    recon_01 = recon - recon.min()
    recon_01 = recon_01 / (recon_01.max() + 1e-12)

    target_01 = target - target.min()
    target_01 = target_01 / (target_01.max() + 1e-12)

    recon_sum = np.clip(recon, 0.0, None)
    recon_sum = recon_sum / (recon_sum.sum() + 1e-12)

    target_sum = np.clip(target, 0.0, None)
    target_sum = target_sum / (target_sum.sum() + 1e-12)

    psnr = peak_signal_noise_ratio(target_01, recon_01, data_range=1.0)
    ssim = structural_similarity(target_01, recon_01, data_range=1.0)
    mse = np.mean((recon_01 - target_01) ** 2)
    nmse = np.sum((recon_01 - target_01) ** 2) / (np.sum(target_01 ** 2) + 1e-12)

    support = target_sum > (1e-6 / (pad_factor ** 2))
    eff = recon_sum[support].sum() / (recon_sum.sum() + 1e-12)

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "mse": float(mse),
        "nmse": float(nmse),
        "diffraction_efficiency": float(eff),
    }


def reconstruct_from_phase_np(phase_np: np.ndarray, device: str,
                              pad_factor: int = 1, apply_sinc: bool = False,
                              fill_factor: float = 1.0, fill_is_areal: bool = True) -> np.ndarray:
    """
    Re-propagate a phase array through the shared forward model and re-normalize
    independently of however the optimizer that produced it normalized things
    internally. This is what makes simulation and bench results comparable: all
    three end up here before compute_metrics is called on their output, so the
    metrics reflect the predicted phase itself, not each method's internal
    training/optimization bookkeeping.

    pad_factor/apply_sinc/fill_factor/fill_is_areal are forwarded, UNCHANGED IN
    MEANING, to physics.hologram_intensity_from_phase. Defaults (1, False, 1.0,
    True) reproduce the original byte-identical ideal-FFT scoring. IMPORTANT:
    callers pass
    the SAME settings used to GENERATE the phase, so generation and scoring are
    always on the same forward model - never call this with different settings
    than whatever produced phase_np, or the reported metric will not reflect
    what was actually optimized/simulated.
    """
    phase_t = torch.from_numpy(phase_np.astype(np.float32)).unsqueeze(0).to(device)
    recon_t = hologram_intensity_from_phase(
        phase_t, pad_factor=pad_factor, apply_sinc=apply_sinc,
        fill_factor=fill_factor, fill_is_areal=fill_is_areal,
    )
    recon_t = normalize_intensity_sum(recon_t)
    return recon_t[0].detach().cpu().numpy().astype(np.float32)

def prepare_target_arrays(image_path: str, device: str, target_formulation: str = "amplitude"):
    """
    Build the optimization target and the (independent) evaluation target for a
    single image.

    target_formulation controls ONLY what is returned as "target_optim_np" below
    - the array that gradient descent uses to drive its optimization:
        - "amplitude" (default, published behaviour):
              target_optim_np = sqrt(clip(X, 0))
              This is the formulation used for every result reported in the
              manuscript.
        - "intensity" (additional condition, run to check whether the choice of
          optimization-target domain disadvantages the baseline):
              target_optim_np = clip(X, 0)
              GD's intensity training target (via target_formulation plumbed
              through to gd.run(), see gd.py) is then driven directly by
              intensity rather than amplitude. Note that gd.run squares an
              amplitude target to recover the intensity target, so the two paths
              agree to floating-point rounding.

    target_eval_np (used for ALL scoring/metrics, regardless of formulation) is
    deliberately computed from the raw image X directly and never from
    target_optim_np, so switching target_formulation cannot change what any
    method is scored against - only what the optimizer is driven toward. This
    independence was verified by trace and by a standalone numerical check before
    this parameter was added.

    Returns a dict containing "target_optim_np" (formulation-aware) and, for
    backward compatibility with any existing external scripts/notebooks that read
    "target_amplitude_np" directly, an alias that is only present when
    target_formulation == "amplitude". New code should read "target_optim_np".
    """
    if target_formulation not in ("amplitude", "intensity"):
        raise ValueError(
            f"Unknown target_formulation: {target_formulation!r}. "
            f"Expected 'amplitude' (published/default) or 'intensity'."
        )

    img = Image.open(image_path).convert("L")
    X = np.asarray(img, dtype=np.float32)

    H, W = X.shape
    X_batched = X.reshape(1, H, W)

    X_clipped = np.clip(X, 0.0, None)
    if target_formulation == "amplitude":
        target_optim_np = np.sqrt(X_clipped).astype(np.float32)
    else:  # "intensity"
        target_optim_np = X_clipped.astype(np.float32)

    # Evaluation target: independent of target_formulation by construction (see
    # docstring above) - computed from X directly, never from target_optim_np.
    target_t = torch.from_numpy(X_batched).to(device=device, dtype=torch.float32)
    target_t = normalize_intensity_sum(target_t)
    target_eval_np = target_t[0].detach().cpu().numpy().astype(np.float32)

    out = {
        "raw_intensity_batched": X_batched,
        "target_optim_np": target_optim_np,
        "target_formulation": target_formulation,
        "target_eval_np": target_eval_np,
        "H": H,
        "W": W,
    }
    if target_formulation == "amplitude":
        # Back-compat alias only - intentionally absent for "intensity" so that
        # any old code path assuming "target_amplitude_np" is amplitude-shaped
        # fails loudly (KeyError) rather than silently treating an intensity
        # array as if it were amplitude.
        out["target_amplitude_np"] = target_optim_np

    return out


def save_target_artifacts(out_dir: str, target_eval_np: np.ndarray, target_optim_np: np.ndarray,
                           target_formulation: str = "amplitude"):
    """
    target_formulation controls the saved filename for the optimization target
    array:
        - "amplitude": writes "target_amplitude.npy" - UNCHANGED filename,
          preserves exact on-disk naming from the published pipeline.
        - "intensity": writes "target_intensity_optim.npy" - a new, distinct
          name (deliberately NOT "target_intensity.npy", which is already used
          below for the evaluation target and must never be overwritten by this
          function regardless of target_formulation).
    """
    np.save(os.path.join(out_dir, "target_intensity.npy"), target_eval_np)
    optim_filename = "target_amplitude.npy" if target_formulation == "amplitude" else "target_intensity_optim.npy"
    np.save(os.path.join(out_dir, optim_filename), target_optim_np)
    save_display_image(target_eval_np, os.path.join(out_dir, "target.png"))

def evaluate_gd_sweep(
    target_amplitude_np,
    target_eval_np,
    out_dir,
    device,
    binary_phase,
    iterations_list,
    gd_kwargs=None,
    target_formulation: str = "amplitude",
):
    """
    GD's run() needs to know target_formulation explicitly (see gd.py module
    docstring: GD historically always squared its target array, assuming
    amplitude; this is only correct when target_formulation=="amplitude").
    target_formulation is therefore both (a) forwarded into gd.run() so the
    correct intensity target is used internally, and (b) used to tag output
    filenames, so sibling-directory sweeps under different formulations cannot
    collide.

    NOTE: if gd_kwargs already contains a "target_formulation" key, that value
    takes precedence (gd_kwargs is expanded after the explicit kwarg below would
    be - see call below); this function's target_formulation parameter exists
    so callers that don't already build a full gd_kwargs dict have a simple way
    to set it.
    """
    ensure_dir(out_dir)
    gd_kwargs = dict(gd_kwargs or {})
    gd_kwargs.setdefault("target_formulation", target_formulation)
    # Extract the faithful-model settings from gd_kwargs (already splatted into
    # gd.run() below for GENERATION) so the EXACT SAME settings are reused for
    # SCORING (reconstruct_from_phase_np). Generation and scoring must not drift.
    pad_factor = gd_kwargs.get("pad_factor", 1)
    apply_sinc = gd_kwargs.get("apply_sinc", False)
    fill_factor = gd_kwargs.get("fill_factor", 1.0)
    fill_is_areal = gd_kwargs.get("fill_is_areal", True)
    target_eval_np_scaled = _target_eval_for_padding(target_eval_np, pad_factor)

    gd = GradientDescentHologram()
    results = []
    tag_suffix = "" if target_formulation == "amplitude" else f"_{target_formulation}"

    for n_iter in iterations_list:
        t0 = time.time()
        out = gd.run(
            target_amplitude_np,
            iterations=n_iter,
            binary_phase=binary_phase,
            device=device,
            **gd_kwargs,
        )
        elapsed = time.time() - t0

        recon_np = reconstruct_from_phase_np(
            out["phase_radians"], device=device,
            pad_factor=pad_factor, apply_sinc=apply_sinc,
            fill_factor=fill_factor, fill_is_areal=fill_is_areal,
        )
        metrics = compute_metrics(recon_np, target_eval_np_scaled, pad_factor)

        item = {
            "iterations": int(n_iter),
            "wall_clock_seconds": float(elapsed),
            "best_loss": float(out.get("best_loss", math.nan)),
            "target_formulation": target_formulation,
            **metrics,
        }
        results.append(item)

        tag = f"gd_iter_{n_iter}{tag_suffix}"
        np.save(os.path.join(out_dir, f"{tag}_phase.npy"), out["phase_radians"])
        Image.fromarray(out["phase_uint8"]).save(os.path.join(out_dir, f"{tag}_phase_8bit.png"))
        np.save(os.path.join(out_dir, f"{tag}_recon.npy"), recon_np)
        save_display_image(recon_np, os.path.join(out_dir, f"{tag}_recon.png"))

    return results


def evaluate_single_image_multilevel(
    image_path: str,
    out_dir: str,
    device="cuda",
    gd_iterations_list=(100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000),
    gd_kwargs=None,
    target_formulations=("amplitude",),
    pad_factor: int = 1,
    apply_sinc: bool = False,
    fill_factor: float = 1.0,
    fill_is_areal: bool = True,
):
    """
    Run the gradient-descent simulation sweep for one image and write a summary.

    The name is historical: upstream this function orchestrates several methods,
    and it is kept here so existing call sites do not have to change. In this
    deposit it runs gradient descent only.

    target_formulations: tuple of one or more of "amplitude" | "intensity".

    pad_factor / apply_sinc / fill_factor / fill_is_areal: the single top-level
    knob for the forward model (see physics.py). Defaults (1, False, 1.0, True)
    give the idealized model - a bare Fourier transform. The faithful model is
    (2, True, 0.91, True). These are injected into gd_kwargs by .setdefault(), so
    an explicit value already present in a caller-supplied gd_kwargs wins.

    Called with the default target_formulations=("amplitude",), output paths and
    summary keys are unchanged: out_dir/gd/gd_iter_{n}_*, target_amplitude.npy,
    and summary["regime1"]["gd_sweep"].

    Passing additional formulations, e.g. ("amplitude", "intensity"), runs the
    sweep again under each and writes to a namespaced sibling directory
    (out_dir/gd_intensity_target/) - never into the bare out_dir/gd/ used by the
    first, always-primary formulation, so the primary sweep's files cannot be
    touched however many additional formulations are requested. Results for the
    extras are reported separately under
    summary["regime1"]["additional_target_formulations"][<formulation>], so
    nothing is ambiguous about which corresponds to the published condition.

    Note that gd.run squares an amplitude target to recover its intensity
    training target, so the two formulations describe what is passed in rather
    than the domain the loss is formed in - which is intensity either way.
    """
    ensure_dir(out_dir)

    if not target_formulations:
        raise ValueError("target_formulations must contain at least one formulation.")
    for f in target_formulations:
        if f not in ("amplitude", "intensity"):
            raise ValueError(f"Unknown target_formulation: {f!r} in target_formulations={target_formulations!r}")
    if len(set(target_formulations)) != len(target_formulations):
        raise ValueError(f"target_formulations contains duplicates: {target_formulations!r}")

    primary_formulation = target_formulations[0]
    additional_formulations = target_formulations[1:]

    # Resolve gd_kwargs ONCE, injecting the top-level forward-model settings via
    # .setdefault() so an explicit value in a caller-supplied dict wins. Built
    # once and reused for the primary formulation and every additional one, so
    # every sweep in this call shares the SAME forward model.
    gd_kwargs_resolved = dict(gd_kwargs) if gd_kwargs else {
        "init_phase": "random",
        "lr": 0.03,
        "optimizer_name": "adam",
        "loss_type": "mse_sum",  # see gd.py: this is what the code has always actually done
        "tv_weight": 0.0,
        "lowpass": False,
        "lpf_sigma": 0.08,
        "add_carrier": False,
        "show": False,
    }
    gd_kwargs_resolved.setdefault("pad_factor", pad_factor)
    gd_kwargs_resolved.setdefault("apply_sinc", apply_sinc)
    gd_kwargs_resolved.setdefault("fill_factor", fill_factor)
    gd_kwargs_resolved.setdefault("fill_is_areal", fill_is_areal)

    # Primary formulation: writes to the original, unchanged paths.
    target = prepare_target_arrays(image_path=image_path, device=device, target_formulation=primary_formulation)
    target_optim_np = target["target_optim_np"]
    target_eval_np = target["target_eval_np"]
    H, W = target["H"], target["W"]

    save_target_artifacts(out_dir, target_eval_np, target_optim_np, target_formulation=primary_formulation)

    summary = {
        "image_path": image_path,
        "slm_shape": [H, W],
        "target_formulation": primary_formulation,
        "regime1": {},
    }

    gd_dir = os.path.join(out_dir, "gd")
    gd_results = evaluate_gd_sweep(
        target_amplitude_np=target_optim_np,
        target_eval_np=target_eval_np,
        out_dir=gd_dir,
        device=device,
        binary_phase=False,
        iterations_list=gd_iterations_list,
        gd_kwargs=gd_kwargs_resolved,
        target_formulation=primary_formulation,
    )
    summary["regime1"]["gd_sweep"] = gd_results

    # Additional formulations, never entered under the default call. Each
    # writes to a namespaced sibling directory that cannot collide with the
    # primary formulation's gd/ directory above, or with each other.
    if additional_formulations:
        summary["regime1"]["additional_target_formulations"] = {}

    for formulation in additional_formulations:
        formulation_target = prepare_target_arrays(
            image_path=image_path, device=device, target_formulation=formulation
        )
        formulation_optim_np = formulation_target["target_optim_np"]
        # target_eval_np is identical regardless of formulation (see
        # prepare_target_arrays); recomputed here for clarity rather than
        # relying on that invariant silently.
        formulation_eval_np = formulation_target["target_eval_np"]

        formulation_subdir = os.path.join(out_dir, f"_{formulation}_target")
        ensure_dir(formulation_subdir)
        save_target_artifacts(
            formulation_subdir, formulation_eval_np, formulation_optim_np, target_formulation=formulation
        )

        gd_results_f = evaluate_gd_sweep(
            target_amplitude_np=formulation_optim_np,
            target_eval_np=formulation_eval_np,
            out_dir=os.path.join(out_dir, f"gd_{formulation}_target"),
            device=device,
            binary_phase=False,
            iterations_list=gd_iterations_list,
            gd_kwargs=gd_kwargs_resolved,
            target_formulation=formulation,
        )

        summary["regime1"]["additional_target_formulations"][formulation] = {
            "gd_sweep": gd_results_f,
        }

    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


#---------------------------------------------------
# EXPERIMENTAL COMPARISON METHODS
#---------------------------------------------------
def build_dataset_manifest(target_dir, capture_root):
    manifest = []

    for fname in os.listdir(target_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_name = os.path.splitext(fname)[0]

        target_path = os.path.join(target_dir, fname)
        capture_dir = os.path.join(capture_root, image_name)

        capture_paths = {}

        for key in ["gd", "gd_citl", "citl"]:
            path = os.path.join(capture_dir, f"{key}.jpg")
            if os.path.exists(path):
                capture_paths[key] = path

        manifest.append({
            "image_name": image_name,
            "image_path": target_path,
            "capture_paths": capture_paths
        })

    return manifest

def evaluate_camera_capture_once(
    image_path,
    target_eval_np,
    out_dir,
    device,
    roi,
    out_hw,
    angle=0,
    eps_norm=1e-12,
    dc_radius=45,
    auto_center=True,
    dc_center=None,
    subtract_min=True,
    median_ksize=0,
    use_dc_exclusion=True,
    tag="camera_capture",
):
    ensure_dir(out_dir)
    print("Saving outputs to:", out_dir)  # temporary debug
    
    t0 = time.time()

    if use_dc_exclusion:
        I_cam, cam_resized_np, dc_mask_t, dc_center_used = load_camera_capture_for_citl(
            image_path=image_path,
            roi=roi,
            out_hw=out_hw,
            device=device,
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
        )
    else:
        I_cam, cam_resized_np = no_dc_load_camera_capture_for_citl(
            image_path=image_path,
            roi=roi,
            out_hw=out_hw,
            device=device,
            angle=angle,
            eps_norm=eps_norm,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
        )
        dc_mask_t = None
        dc_center_used = None

    elapsed = time.time() - t0

    recon_np = I_cam[0].detach().cpu().numpy().astype(np.float32)
    metrics = compute_metrics(recon_np, target_eval_np)

    result = {
        "image_path": image_path,
        # "wall_clock_seconds": float(elapsed),
        "use_dc_exclusion": bool(use_dc_exclusion),
        **metrics,
    }

    if dc_center_used is not None:
        result["dc_center_used"] = [int(dc_center_used[0]), int(dc_center_used[1])]
        result["dc_radius"] = int(dc_radius)

    np.save(os.path.join(out_dir, f"{tag}_recon.npy"), recon_np)
    save_display_image(recon_np, os.path.join(out_dir, f"{tag}_recon.png"))

    np.save(os.path.join(out_dir, f"{tag}_raw_resized.npy"), cam_resized_np)
    save_display_image(cam_resized_np, os.path.join(out_dir, f"{tag}_raw_resized.png"))

    if dc_mask_t is not None:
        dc_mask_np = (dc_mask_t.detach().cpu().numpy().astype(np.uint8) * 255)
        Image.fromarray(dc_mask_np).save(os.path.join(out_dir, f"{tag}_dc_mask.png"))

    with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result

def evaluate_single_target_experimental(
    target_image_path,
    image_name,
    capture_paths,
    out_dir,
    device,
    roi,
    angle=0,
    eps_norm=1e-12,
    dc_radius=45,
    auto_center=True,
    dc_center=None,
    subtract_min=True,
    median_ksize=0,
    use_dc_exclusion=True,
):
    ensure_dir(out_dir)

    target = prepare_target_arrays(image_path=target_image_path, device=device)
    target_eval_np = target["target_eval_np"]
    target_amplitude_np = target["target_amplitude_np"]
    H, W = target["H"], target["W"]

    save_target_artifacts(out_dir, target_eval_np, target_amplitude_np)

    summary = {
        "image_name": image_name,
        "image_path": target_image_path,
        "summary": {
            "image_path": target_image_path,
            "slm_shape": [H, W],
                        "regime1": {}
        }
    }

    # GD experimental (fixed chosen setting: 750 iters)
    if "gd" in capture_paths:
        gd_result = evaluate_camera_capture_once(
            image_path=capture_paths["gd"],
            target_eval_np=target_eval_np,
            out_dir=os.path.join(out_dir, "gd"),
            device=device,
            roi=roi,
            out_hw=(H, W),
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
            use_dc_exclusion=use_dc_exclusion,
            tag="gd_exp",
        )
        summary["summary"]["regime1"]["gd"] = {
            "iterations": 750,
            "psnr": gd_result["psnr"],
            "ssim": gd_result["ssim"],
            "mse": gd_result["mse"],
            "nmse": gd_result["nmse"],
            "diffraction_efficiency": gd_result["diffraction_efficiency"],
        }

    known_methods = {"gd"}

    for method_name, capture_path in capture_paths.items():
        if method_name in known_methods:
            continue

        result = evaluate_camera_capture_once(
            image_path=capture_path,
            target_eval_np=target_eval_np,
            out_dir=os.path.join(out_dir, method_name),
            device=device,
            roi=roi,
            out_hw=(H, W),
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
            use_dc_exclusion=use_dc_exclusion,
            tag=method_name,
        )

        summary["summary"]["regime1"][method_name] = {
            "psnr": result["psnr"],
            "ssim": result["ssim"],
            "mse": result["mse"],
            "nmse": result["nmse"],
            "diffraction_efficiency": result["diffraction_efficiency"],
        }

    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary

def evaluate_experimental_dataset(
    dataset_manifest,
    out_dir,
    device,
    roi,
    angle=0,
    eps_norm=1e-12,
    dc_radius=45,
    auto_center=True,
    dc_center=None,
    subtract_min=True,
    median_ksize=0,
    use_dc_exclusion=True,
):
    ensure_dir(out_dir)

    all_results = []

    for item in dataset_manifest:
        image_name = item["image_name"]
        target_image_path = item["image_path"]
        capture_paths = item["capture_paths"]

        print(f"\n=== Evaluating {image_name} ===")

        target_out_dir = os.path.join(out_dir, image_name)

        result = evaluate_single_target_experimental(
            target_image_path=target_image_path,
            image_name=image_name,
            capture_paths=capture_paths,
            out_dir=target_out_dir,
            device=device,
            roi=roi,
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
            use_dc_exclusion=use_dc_exclusion,
        )

        all_results.append(result)

    with open(os.path.join(out_dir, "comparison_experimental_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results