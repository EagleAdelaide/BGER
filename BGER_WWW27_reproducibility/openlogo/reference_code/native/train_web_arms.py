#!/usr/bin/env python3
"""E1 Web model and selected-batch balanced BCE; CLI delegates to guarded runner.
Web keeps ImageNet V1, batch-level loss weighting and existing adaptive-average
mask-to-grid >.5 rule. Retrieval uses raw pooled embeddings and stabilized
cosine. CPU fixtures use weights=None, never a pretrained download.
"""
import argparse, json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv

def balanced_bce(logits, target):
    pos = target.sum().clamp_min(1.0); neg = (1 - target).sum().clamp_min(1.0)
    w = torch.where(target > 0.5, (pos + neg) / (2 * pos), (pos + neg) / (2 * neg))
    return F.binary_cross_entropy_with_logits(logits, target, weight=w)

class WebBGER(nn.Module):
    def __init__(self, n_ids, arm, *, weights=tv.models.ResNet50_Weights.IMAGENET1K_V1):
        super().__init__()
        if arm not in {"ce", "aux", "bger", "cam_bce"}:  # abn_box: see abn_box.ABNBox
            raise ValueError(f"Unknown arm: {arm}")
        m = tv.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(m.children())[:-2])   # -> (B,2048,h,w)
        self.gate = nn.Conv2d(2048, 1, 1)
        self.head = nn.Linear(2048, n_ids)
        self.arm = arm
    def features(self, x): return self.backbone(x)
    def pooled_features(self, H):
        """Spatial features whose GAP is the actual representation (B,C,h,w).

        A1 adopted by author: unnormalized gated GAP; shared with secondary.
        """
        if H.ndim != 4:
            raise ValueError("Expected features with shape (B,C,h,w)")
        return torch.sigmoid(self.gate(H)) * H if self.arm == "bger" else H

    def pool(self, H):
        return self.pooled_features(H).mean(dim=(2, 3))
    def forward(self, x):
        H = self.features(x)
        z = self.pool(H)
        return self.head(z), H, z

def build_model(n_ids, arm, *, weights=tv.models.ResNet50_Weights.IMAGENET1K_V1, init_seed=0):
    if arm == "abn_box":
        from abn_box import ABNBox
        return ABNBox(n_ids, weights=weights, init_seed=init_seed)
    return WebBGER(n_ids, arm, weights=weights)

def step_losses(model, x, y, box_mask, has_box, lam, arm):
    logits, H, z = model(x)
    loss = F.cross_entropy(logits, y); parts = {"ce": float(loss.detach())}
    if arm == "abn_box":
        # official ABN objective: CE(attention branch) + CE(perception branch)
        att_logit, att_cls = model._last["att_logit"][:, 0], model._last["att_cls"]
        att_ce = F.cross_entropy(att_cls, y); loss = loss + att_ce; parts["att_ce"] = float(att_ce.detach())
        if has_box.any():
            tgt = F.adaptive_avg_pool2d(box_mask[:, None], att_logit.shape[-2:])[:, 0]
            sup = balanced_bce(att_logit[has_box], (tgt[has_box] > 0.5).float())
            loss = loss + lam * sup; parts["att_box_bce"] = float(sup.detach())
        return loss, parts
    if arm in ("aux", "bger") and has_box.any():
        gate_logits = model.gate(H)[:, 0]
        tgt = F.adaptive_avg_pool2d(box_mask[:, None], gate_logits.shape[-2:])[:, 0]
        sup = balanced_bce(gate_logits[has_box], (tgt[has_box] > 0.5).float())
        loss = loss + lam * sup; parts["gate_bce"] = float(sup.detach())
    if arm == "cam_bce" and has_box.any():
        w_y = model.head.weight[y]                                  # (B,2048)
        cam_proxy = torch.einsum("bc,bchw->bhw", w_y, H)
        tgt = F.adaptive_avg_pool2d(box_mask[:, None], cam_proxy.shape[-2:])[:, 0]
        sup = balanced_bce(cam_proxy[has_box], (tgt[has_box] > 0.5).float())
        loss = loss + lam * sup; parts["cam_bce"] = float(sup.detach())
    return loss, parts

def main():
    # All executions, including fixture runs, go through the production guards.
    from e1_runner import main as run
    run()

if __name__ == "__main__": main()
