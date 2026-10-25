"""Batched Grad-CAM for CNN backbones (hook-based, no external CAM library).

Notes / sanity properties:
  * Grad-CAM is computed w.r.t. an EXPLICIT target class (recorded by caller).
  * Temperature Scaling multiplies logits by 1/T (a positive constant). This
    scales the class score and therefore the gradients uniformly; after the
    per-map ReLU + [0,1] normalization the heatmap is IDENTICAL. Hence Grad-CAM
    grounding is invariant to scalar TS -- a key point for hypothesis H1. We do
    NOT recompute CAM per temperature; we reuse it and only the confidences c_i
    (and thus GCE) change.
  * Heatmaps are returned at the input spatial resolution, normalized to [0,1].
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def normalize_cam(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max normalize a (B,H,W) tensor to [0,1]."""
    b = cam.shape[0]
    flat = cam.view(b, -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    flat = (flat - mn) / (mx - mn + eps)
    return flat.view_as(cam)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module,
                 reshape_transform=None):
        """reshape_transform: optional fn mapping a hooked activation/gradient to
        (B, C, H, W). Used for transformers whose token activations are (B, N, C)
        or whose feature maps are channels-last (B, H, W, C)."""
        self.model = model
        self.target_layer = target_layer
        self.reshape = reshape_transform
        self._acts = None
        self._grads = None
        # forward hook captures the activation AND registers a tensor hook for its
        # gradient. This avoids full_backward_hook's "view modified inplace" error
        # on architectures with inplace ReLU (e.g. DenseNet) and is robust across
        # CNN and transformer backbones.
        self._h1 = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self._acts = out
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def remove(self):
        self._h1.remove()

    def __call__(self, images: torch.Tensor, target_classes=None,
                 out_size=None, temperature: float = 1.0, method: str = "gradcam"):
        """Return (cams[B,H,W] in [0,1], logits[B,C], target_classes[B]).

        target_classes: if None, uses the model's predicted (argmax) class.
        out_size: (H,W) to upsample to; defaults to input H,W.
        temperature: divide logits by T before backprop (uniform scaling; removed
            by the [0,1] normalization -> map invariant to T).
        method: attribution operator on the same hooked activation/gradient:
            'gradcam'   -- GAP the gradients, weight the activation channels;
            'hirescam'  -- elementwise grad*act, sum over channels (no grad GAP);
            'gradcampp' -- Grad-CAM++ first-gradient-power approximation;
            not exact higher derivatives for nonlinear gated heads.
        """
        self.model.zero_grad(set_to_none=True)
        images = images.requires_grad_(False)
        logits = self.model(images)
        if target_classes is None:
            target = logits.argmax(dim=1)
        else:
            target = torch.as_tensor(target_classes, device=logits.device)
        score = (logits / temperature).gather(1, target.view(-1, 1)).sum()
        score.backward()

        acts = self._acts                       # (B, K, h, w) or transformer shape
        grads = self._grads
        if self.reshape is not None:             # transformers -> (B, K, h, w)
            acts = self.reshape(acts)
            grads = self.reshape(grads)
        if method == "hirescam":
            cam = F.relu((grads * acts).sum(dim=1))          # elementwise, no GAP
        elif method == "gradcampp":
            g2, g3 = grads ** 2, grads ** 3
            # Eq. 19 keeps g^3 at (i,j) outside the activation spatial sum.
            # Powers of first derivatives assume a locally affine score head;
            # sigmoid gating violates that assumption (see regression tests).
            denom = 2 * g2 + acts.sum(dim=(2, 3), keepdim=True) * g3
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha = g2 / denom
            weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1))
        else:                                                # gradcam (default)
            weights = grads.mean(dim=(2, 3), keepdim=True)   # GAP over spatial dims
            cam = F.relu((weights * acts).sum(dim=1))        # (B, h, w)
        if out_size is None:
            out_size = images.shape[-2:]
        cam = F.interpolate(cam.unsqueeze(1), size=out_size,
                            mode="bilinear", align_corners=False).squeeze(1)
        cam = normalize_cam(cam)
        return (cam.detach().cpu().numpy(),
                logits.detach().cpu().numpy(),
                target.detach().cpu().numpy())


def resolve_target_layer(model, name: str):
    """Map a config name to a module (ResNet/DenseNet/EfficientNet friendly)."""
    if hasattr(model, name):
        return getattr(model, name)
    # dotted path fallback
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod
