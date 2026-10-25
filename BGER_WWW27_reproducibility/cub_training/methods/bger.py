"""Box-Guided Evidence Routing (BGER) — ResNet-50.

Backbone feature map H -> learned gate M=sigmoid(Conv1x1(H)) -> gated features
H_e=M*H -> z=GAP(H_e) -> logits=linear(z). The gate is supervised to concentrate
inside the GT bounding box (downsampled to the feature grid) plus an area
constraint. No reliability head, no keep/remove losses in this pilot.

  L = CE(logits, y) + lambda_box * (1 - sum(M*B)/(sum(M)+eps)) + lambda_area*(mean(M)-rho)^2
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class BGERResNet50(nn.Module):
    def __init__(self, num_classes=200, pretrained=True):
        super().__init__()
        w = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = tvm.resnet50(weights=w)
        self.features = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3, base.layer4)
        C = base.fc.in_features
        self.gate = nn.Conv2d(C, 1, kernel_size=1)
        self.classifier = nn.Linear(C, num_classes)

    def forward(self, x, return_gate=False):
        H = self.features(x)
        M = torch.sigmoid(self.gate(H))      # (B,1,h,w)
        z = (M * H).mean(dim=(2, 3))
        logits = self.classifier(z)
        if return_gate:
            return logits, M
        return logits

    def cam_target(self):
        return self.features


def box_to_grid(bbox, g, S=224):
    """bbox (B,4) [x1,y1,x2,y2] in S-px coords -> (B,1,g,g) binary feature-grid mask."""
    B = bbox.shape[0]
    out = torch.zeros(B, 1, g, g, device=bbox.device)
    s = g / S
    bb = (bbox * s)
    for i in range(B):
        x1, y1, x2, y2 = bb[i].tolist()
        gx1, gy1 = int(np.floor(x1)), int(np.floor(y1))
        gx2, gy2 = max(gx1 + 1, int(np.ceil(x2))), max(gy1 + 1, int(np.ceil(y2)))
        gx1 = max(0, min(g - 1, gx1)); gy1 = max(0, min(g - 1, gy1))
        gx2 = max(gx1 + 1, min(g, gx2)); gy2 = max(gy1 + 1, min(g, gy2))
        out[i, 0, gy1:gy2, gx1:gx2] = 1.0
    return out


def bger_loss(logits, M, B, y, hp, eps=1e-6):
    inside = (M * B).sum(dim=(1, 2, 3))
    total = M.sum(dim=(1, 2, 3)) + eps
    L_box = (1 - inside / total).mean()
    L_area = (M.mean() - hp["rho"]) ** 2
    loss = F.cross_entropy(logits, y) + hp["lambda_box"] * L_box + hp["lambda_area"] * L_area
    return loss, dict(box=float(L_box.item()), area=float(L_area.item()), gate=float(M.mean().item()))
