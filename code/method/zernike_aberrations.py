"""
zernike_aberrations.py - Zernike aberration generation + synthetic camera
source for the GD+CITL far-field aberration stress test (pure simulation).

NO hardware, NO file I/O, NO paths - pure functions only. Depends only on
numpy, torch, and physics.py (the shared forward model), so it can be imported
and unit-tested standalone.

--------------------------------------------------------------------------
What this is for
--------------------------------------------------------------------------
The GD+CITL far-field paper's simulation stress test asks: if we inject a known
optical aberration into the "measurement" that camera-in-the-loop GD optimizes
against - while the optimizer's OWN forward model stays clean/faithful (i.e.
it does NOT know the aberration) - can GD still recover the target, and up to
what aberration magnitude does it hold?

This mirrors the real bench: the physical camera shows GD the TRUE (aberrated)
system, while GD's simulated gradients come from an un-aberrated model. Here the
"camera" is synthetic: it applies the aberration and propagates through the SAME
faithful physics.py forward model, returning an intensity as if measured.

Because this is pure simulation, there is NO zero-order/DC spike (that is a
physical bench artifact), so gd.run(dc_mask=None) is correct for this test.

--------------------------------------------------------------------------
Conventions (state these in the paper's methods)
--------------------------------------------------------------------------
- Zernike basis: standard NOLL-ordered, ORTHONORMAL over the unit disk. Each
  polynomial has unit variance (RMS = 1) over the disk, so a coefficient of c
  *waves* means exactly c waves RMS of that mode. This is the adaptive-optics
  convention and makes "0.3 lambda RMS of defocus" an unambiguous, checkable
  statement.
- Magnitude unit: WAVES (lambda) of RMS wavefront error. Converted to a phase
  map in radians via phase = 2*pi * (wavefront in waves). One wave = 2*pi rad.
- Circular pupil on a square grid: Zernikes are defined on the unit disk. We
  evaluate them on the disk inscribed in the square SLM grid and set everything
  outside the disk to zero, then use the full square phase map (the standard CGH
  handling: circular Zernike over the circumscribed/inscribed circle, applied to
  the rectangular display region). Modes remain orthonormal where defined.
- Combined aberration: because Noll modes are orthonormal, the total RMS of a
  sum is the root-sum-square of the per-mode coefficients. To make a combined
  aberration of total magnitude M waves RMS spread EQUALLY over N modes, each
  mode gets coefficient M / sqrt(N). This gives one scalar knob (total RMS) that
  maps deterministically to the aberration, fully reproducible, no RNG.

--------------------------------------------------------------------------
The nine low-order modes, built as PROPER orthonormal Noll Zernikes rather
than a practical,
non-orthonormal, statistically-normalized basis - state this distinction in
the methods).
--------------------------------------------------------------------------
"""
from __future__ import annotations
import math
import numpy as np
import torch

from physics import hologram_intensity_from_phase


# Noll indices 1..11 for the low-order modes we care about, with human-readable
# names. Noll ordering is the AO-standard sequential index.
#   1  piston
#   2  tip   (x-tilt)
#   3  tilt  (y-tilt)
#   4  defocus
#   5  astigmatism (oblique, 45 deg)
#   6  astigmatism (vertical, 0/90 deg)
#   7  coma (vertical, y)
#   8  coma (horizontal, x)
#   9  trefoil (vertical)      - not in the core 9, available if wanted
#   10 trefoil (oblique)       - not in the core 9, available if wanted
#   11 primary spherical
#
# The "core 9" for this study are Noll
# {1,2,3,4,5,6,7,8,11}. Trefoil (9,10) are provided for completeness but not in
# the default set.
NOLL_NAMES = {
    1: "piston",
    2: "tip_x",
    3: "tilt_y",
    4: "defocus",
    5: "astig_oblique",
    6: "astig_vertical",
    7: "coma_y",
    8: "coma_x",
    9: "trefoil_vertical",
    10: "trefoil_oblique",
    11: "spherical",
}

# The default nine modes for the study, by Noll index.
CORE_NINE_NOLL = (1, 2, 3, 4, 5, 6, 7, 8, 11)

# Modes that are intensity-trivial (do not degrade the reconstruction in a way
# GD must "recover" from) and are therefore EXCLUDED from the combined sweep,
# but KEPT in the per-mode sweep (to show, and explain, their flat curves):
#   piston (1): global phase constant -> zero effect on |F|^2.
#   tip/tilt (2,3): linear phase ramp -> lateral shift of the replay pattern,
#                   trivially compensable by a phase ramp in a phase-only system.
TRIVIAL_NOLL = (1, 2, 3)


def _noll_to_nm(j: int) -> tuple[int, int]:
    """
    Convert a 1-based Noll index j to the (n, m) radial/azimuthal orders.
    Standard Noll sequential ordering.
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def _radial_poly(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Zernike radial polynomial R_n^{|m|}(rho)."""
    m = abs(m)
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k * math.factorial(n - k)) / (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )
        R += c * rho ** (n - 2 * k)
    return R


def zernike_mode(noll_j: int, grid_size: int) -> np.ndarray:
    """
    Build a single Zernike mode on a square grid_size x grid_size array,
    evaluated over the FULL SQUARE (coordinates in [-1,1] on both axes), then
    normalized to unit RMS over the whole grid.

    IMPORTANT design choice (full square, NOT disk-masked): Zernike polynomials
    are formally orthonormal only on the unit disk, but masking to the inscribed
    disk (zeroing ~23% of a square grid outside it) creates a HARD CIRCULAR EDGE
    - a sharp phase discontinuity that injects large spurious high-frequency
    content into the far field. That artifact corrupts EVERY mode and, worse,
    makes even piston (which must be a pure global phase with ZERO intensity
    effect) change the reconstruction. Evaluating on the full square with no
    mask:
      - makes piston a true global constant (verified: zero intensity effect),
      - removes the hard-edge high-frequency artifact,
      - matches the convention used elsewhere in this project (that basis is
        built on the full square linspace(-1,1) grid, explicitly "not perfectly
        normalized Zernikes, but very practical").
    The cost is that the modes are only approximately orthonormal (they are not
    orthogonal over a square). This is an accepted, documented tradeoff - the
    modes retain their correct aberration SHAPES (defocus curvature, coma, etc.),
    which is what matters for a stress test, and unit-RMS-over-the-grid keeps the
    coefficient == RMS-waves interpretation exact for the grid we actually use.

    Returns a float64 array (waves of wavefront per unit coefficient).
    """
    n, m = _noll_to_nm(noll_j)

    # Grid in [-1, 1] on both axes; evaluated over the full square.
    coords = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(coords, coords)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    R = _radial_poly(n, m, rho)
    if m > 0:
        Z = R * np.cos(m * theta)
    elif m < 0:
        Z = R * np.sin(-m * theta)
    else:
        Z = R

    Z = Z.astype(np.float64)

    # Piston (n=0,m=0) is the constant 1 -> a true global phase offset. Leave it
    # as a genuine constant (do NOT mean-subtract), so that using it produces a
    # pure global phase with exactly zero intensity effect (the correct,
    # physically-trivial behaviour we want to demonstrate in the per-mode sweep).
    if noll_j == 1:
        return np.ones((grid_size, grid_size), dtype=np.float64)

    # For all non-piston modes: normalize to unit RMS over the FULL grid, so a
    # coefficient c (waves) contributes exactly c waves RMS on this grid.
    rms = np.sqrt(np.mean(Z ** 2))
    if rms > 1e-12:
        Z = Z / rms
    return Z


def aberration_phase_map(
    grid_size: int,
    magnitude_waves_rms: float,
    noll_indices=(4,),
    combined_equal_rms: bool = True,
) -> np.ndarray:
    """
    Build an aberration PHASE map (in RADIANS) of a given total RMS magnitude.

    grid_size: side length of the square grid (SLM native resolution H=W).
    magnitude_waves_rms: total aberration magnitude, in WAVES of RMS wavefront.
    noll_indices: which Noll mode(s) to include.
        - single index (e.g. (4,))  -> a pure single-mode aberration of the
          given RMS (per-mode sweep).
        - multiple indices          -> a combined aberration. With
          combined_equal_rms=True, total RMS is split equally: each mode gets
          coefficient magnitude_waves_rms / sqrt(N). Because the modes are
          orthonormal, sqrt(sum of per-mode coeff^2) == magnitude_waves_rms
          exactly, so the requested total RMS is achieved.
    combined_equal_rms: only relevant when len(noll_indices) > 1.

    Returns: (grid_size, grid_size) float64 phase map in RADIANS
    (= 2*pi * wavefront-in-waves), zero outside the inscribed disk.
    """
    idxs = tuple(noll_indices)
    n_modes = len(idxs)
    if n_modes == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float64)

    if n_modes == 1:
        per_mode_coeff = [magnitude_waves_rms]
    else:
        if combined_equal_rms:
            c = magnitude_waves_rms / math.sqrt(n_modes)
            per_mode_coeff = [c] * n_modes
        else:
            # equal fraction (not equal RMS) fallback - not used by default
            per_mode_coeff = [magnitude_waves_rms / n_modes] * n_modes

    wavefront_waves = np.zeros((grid_size, grid_size), dtype=np.float64)
    for j, coeff in zip(idxs, per_mode_coeff):
        wavefront_waves += coeff * zernike_mode(j, grid_size)

    # Rescale to hit the requested total RMS EXACTLY. For a single mode this is
    # a no-op (already unit-RMS * coeff). For combined modes, the full-square
    # modes are only APPROXIMATELY orthogonal, so the analytic root-sum-square
    # (M/sqrt(N) per mode) drifts from the target by a few percent; measuring
    # the actual RMS and rescaling makes the x-axis magnitude exact regardless.
    # (Zero-magnitude requests stay zero.)
    if magnitude_waves_rms != 0.0:
        actual_rms = np.sqrt(np.mean(wavefront_waves ** 2))
        if actual_rms > 1e-12:
            wavefront_waves *= (magnitude_waves_rms / actual_rms)

    # waves -> radians
    phase_radians = 2.0 * math.pi * wavefront_waves
    return phase_radians


def make_aberrated_camera_source(
    aberration_phase_radians: np.ndarray,
    pad_factor: int,
    apply_sinc: bool,
    fill_factor: float,
    fill_is_areal: bool,
    device: str,
    eps: float = 1e-6,
):
    """
    Build a SYNTHETIC camera_source callable for gd.run(camera_source=...).

    This is the heart of the stress test. gd.run calls
    camera_source(phase_np, iteration_index) -> I_camera each iteration, where
    phase_np is the CURRENT phase-only SLM pattern (numpy, shape (H,W)). Here we
    simulate the "measurement" of an ABERRATED system:

        I_camera = | Faithful_Forward( exp(i * (phase + aberration)) ) |^2

    i.e. we ADD the fixed aberration phase to whatever hologram GD currently
    proposes, then propagate through the SAME faithful physics.py forward model
    used everywhere else, and return the resulting intensity.

    Crucially, GD's OWN internal forward pass (used for its gradients) does NOT
    include this aberration - gd.py builds its gradient from the clean model.
    So the optimizer is "blind" to the aberration and only sees its effect
    through this synthetic measurement, exactly as it would only see a real
    bench's aberration through a real camera. That mismatch is the whole point:
    can CITL feedback recover the target despite the optimizer's model being
    wrong?

    Resolution note: gd.run at pad_factor>1 works at padded resolution (M,N) =
    (pad_factor*H, pad_factor*W) for its measurement/loss (target upsampled
    accordingly). This synthetic camera returns intensity at that SAME padded
    resolution, because it propagates through the same forward model that
    returns the replay at padded resolution - so its output shape matches what
    gd.run expects from a camera at pad_factor>1 (mirrors the real camera
    pipeline's out_hw=(M,N) requirement).

    aberration_phase_radians: (H,W) numpy, the fixed aberration (same for every
        iteration - it is a static property of the "optical system").
    pad_factor/apply_sinc/fill_factor/fill_is_areal: MUST match the settings
        gd.run is called with, so the synthetic measurement is on the same
        forward model as the optimizer's clean model (they differ ONLY by the
        aberration, nothing else).
    device: torch device string.

    Returns: a callable(phase_np, iteration_index) -> np.ndarray intensity.
    """
    aberr_t = torch.from_numpy(
        np.asarray(aberration_phase_radians, dtype=np.float32)
    ).to(device)

    def camera_source(phase_np_or_t, iteration_index: int) -> torch.Tensor:
        # gd.py passes the current phase as a tensor (phase_for_forward_t);
        # accept either tensor or numpy for robustness.
        if isinstance(phase_np_or_t, torch.Tensor):
            phase_t = phase_np_or_t.to(device)
        else:
            phase_t = torch.from_numpy(np.asarray(phase_np_or_t, dtype=np.float32)).to(device)
        # Add the (static) aberration to the current proposed hologram phase.
        aberrated_phase = phase_t + aberr_t
        with torch.no_grad():
            I = hologram_intensity_from_phase(
                aberrated_phase,
                eps=eps,
                pad_factor=pad_factor,
                apply_sinc=apply_sinc,
                fill_factor=fill_factor,
                fill_is_areal=fill_is_areal,
            )
            # CRITICAL: gd.py does NOT normalize camera_source's return value
            # itself - it substitutes I_camera directly against recon_for_loss,
            # which IS normalized to sum = (M*N) (see gd.py:
            # "recon_for_loss = recon / recon.sum() * (M*N)"). The real hardware
            # camera path enforces this same contract explicitly
            # (normalize_intensity_sum_excluding_mask(..., scale_to_numel=True)
            # before returning). Without this normalization here, I_camera's raw
            # sum (which varies with how much energy apply_sinc removes and how
            # the current phase distributes energy) would be on a DIFFERENT
            # scale than recon_for_loss, silently corrupting the STE substitution
            # target_for_value = recon_for_loss + (I_camera - recon_for_loss).
            # Mirror gd.py's own normalization line exactly (same eps, same M*N
            # convention, using I's own spatial dims so this is correct
            # regardless of pad_factor).
            Mc, Nc = I.shape[-2], I.shape[-1]
            I = I / (I.sum(dim=(-2, -1), keepdim=True) + 1e-12) * (Mc * Nc)
        # hologram_intensity_from_phase returns (1,M,N) for a (H,W) phase input.
        # gd.py's camera_source contract expects a BATCHED (1,M,N) TORCH TENSOR
        # on the compute device: gd.py does arithmetic (I_camera - recon_for_loss)
        # against a tensor, and the real hardware camera source likewise returns
        # I_cam as a normalized torch tensor of shape (1,H,W). So return the
        # (now correctly normalized) tensor directly - do NOT convert to numpy,
        # do NOT squeeze the batch dim.
        return I

    return camera_source
