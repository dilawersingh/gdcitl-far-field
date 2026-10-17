r"""
perceptual.py - perceptual image-quality metrics, complementing PSNR and SSIM.

    lpips    LPIPS with the AlexNet backbone (Zhang et al., CVPR 2018).
             Lower is better; 0 means identical.
    haarpsi  HaarPSI, a Haar-wavelet-based perceptual similarity index
             (Reisenhofer et al., Signal Process. Image Commun. 2018).
             Higher is better; 1 means identical.

Both see exactly the images PSNR and SSIM see: reconstruction and target are
min-max normalized to [0, 1] over the full frame, as in
evaluate_methods.compute_metrics, at the same resolution (pad_factor applied
upstream). Grayscale is repeated to three channels for LPIPS and scaled to
[-1, 1], which is the input range the network expects.

These are kept out of compute_metrics on purpose: compute_metrics is called
inside the optimization loop, and LPIPS loads a network. Nothing here changes
any existing metric.

Requires: pip install lpips piq   (LPIPS downloads AlexNet weights on first use)
"""
from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-12
_LPIPS = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _lpips_model(device: str):
    if device not in _LPIPS:
        import lpips
        _LPIPS[device] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _LPIPS[device]


def _minmax(x) -> np.ndarray:
    """Identical to the normalization in compute_metrics."""
    x = np.asarray(x, dtype=np.float32)
    x = x - x.min()
    return x / (x.max() + _EPS)


def perceptual_metrics(recon, target, device: str | None = None) -> dict:
    import piq

    device = device or _device()
    r = torch.from_numpy(_minmax(recon))[None, None].to(device)
    t = torch.from_numpy(_minmax(target))[None, None].to(device)
    with torch.no_grad():
        lp = _lpips_model(device)(r.repeat(1, 3, 1, 1) * 2 - 1,
                                  t.repeat(1, 3, 1, 1) * 2 - 1)
        hp = piq.haarpsi(r.clamp(0, 1), t.clamp(0, 1), data_range=1.0,
                         reduction="none")
    return {"lpips": float(lp.item()), "haarpsi": float(hp.item())}
