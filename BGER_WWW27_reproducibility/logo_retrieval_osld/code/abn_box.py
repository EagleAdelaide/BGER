"""ABN-box: box-supervised adaptation of the Attention Branch Network.

Structure follows the official ImageNet ResNet-50 ABN (Fukui et al., CVPR 2019;
github.com/machine-perception-robotics-group/attention_branch_network, commit
5e418783, models/imagenet/resnet.py): shared stem+layer1-3; attention branch
att_layer4 (3 bottlenecks, stride 1) -> BN -> 1x1 conv to K -> BN -> ReLU;
attention M = sigmoid(BN(3x3 conv K->1)); attention-branch logits =
GAP(1x1 conv K->K); perception branch = layer4((1 + M) * x) -> GAP -> FC.
Training loss as in the official code: CE(attention logits) + CE(perception
logits).

Adaptation (not an official reproduction): the attention logit map (14x14) is
additionally supervised with the same batch-balanced box BCE used by the other
arms, against the same 7x7 any-overlap box target nearest-upsampled to 14x14
(identical supervision information for every spatially supervised arm),
weighted by lambda.
Shared tensors (stem, layer1-4, perception FC) are loaded from the common
per-seed template; the attention branch is new and initialized with the official
Kaiming/BN rule under a locked seed. Retrieval uses the perception-branch GAP
embedding; readouts differentiate that embedding's score with respect to the
perception layer4 output (the actual GAP input).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.models.resnet import Bottleneck

ABN_INIT_OFFSET = 777_000  # locked seed offset for the new attention branch


class ABNBox(nn.Module):
    arm = "abn_box"

    def __init__(self, n_ids, *, weights=tv.models.ResNet50_Weights.IMAGENET1K_V1, init_seed=0):
        super().__init__()
        m = tv.models.resnet50(weights=weights)
        # identical module names/shapes to WebBGER.backbone so the common template loads
        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.head = nn.Linear(2048, n_ids)          # perception-branch FC
        g = torch.random.fork_rng(devices=[])
        with g:
            torch.manual_seed(init_seed + ABN_INIT_OFFSET)
            down = nn.Sequential(nn.Conv2d(1024, 2048, 1, stride=1, bias=False), nn.BatchNorm2d(2048))
            self.att_layer4 = nn.Sequential(Bottleneck(1024, 512, 1, down), Bottleneck(2048, 512), Bottleneck(2048, 512))
            self.bn_att = nn.BatchNorm2d(2048)
            self.att_conv = nn.Conv2d(2048, n_ids, 1, bias=False)
            self.bn_att2 = nn.BatchNorm2d(n_ids)
            self.att_conv2 = nn.Conv2d(n_ids, n_ids, 1, bias=False)
            self.att_conv3 = nn.Conv2d(n_ids, 1, 3, padding=1, bias=False)
            self.bn_att3 = nn.BatchNorm2d(1)
            for mod in [self.att_layer4, self.bn_att, self.att_conv, self.bn_att2,
                        self.att_conv2, self.att_conv3, self.bn_att3]:
                for sub in mod.modules():
                    if isinstance(sub, nn.Conv2d):
                        nn.init.kaiming_normal_(sub.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(sub, nn.BatchNorm2d):
                        nn.init.constant_(sub.weight, 1); nn.init.constant_(sub.bias, 0)
        self._last = {}

    def _trunk(self, x):
        b = self.backbone
        x = b[3](b[2](b[1](b[0](x))))      # conv1, bn1, relu, maxpool
        return b[6](b[5](b[4](x)))          # layer1-3 -> (B,1024,14,14)

    def attention(self, fe):
        ax = self.bn_att(self.att_layer4(fe))
        ax = F.relu(self.bn_att2(self.att_conv(ax)))
        att_logit = self.bn_att3(self.att_conv3(ax))              # (B,1,14,14)
        att_cls = self.att_conv2(ax).mean(dim=(2, 3))             # (B,K)
        return att_logit, att_cls

    def features(self, x):
        """Perception-branch layer4 output, the actual GAP input (B,2048,7,7)."""
        fe = self._trunk(x)
        att_logit, att_cls = self.attention(fe)
        self._last = {"att_logit": att_logit, "att_cls": att_cls}
        return self.backbone[7](fe * torch.sigmoid(att_logit) + fe)

    def pooled_features(self, H):
        if H.ndim != 4:
            raise ValueError("Expected features with shape (B,C,h,w)")
        return H

    def pool(self, H):
        return self.pooled_features(H).mean(dim=(2, 3))

    def forward(self, x):
        H = self.features(x)
        z = self.pool(H)
        return self.head(z), H, z


def load_common_template(model: ABNBox, state: dict):
    """Load shared tensors from the common per-seed template; report what was
    shared and what is new. The template's BGER gate has no ABN counterpart."""
    own = model.state_dict()
    shared = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape
              and (k.startswith("backbone.") or k.startswith("head."))}
    missing = sorted(set(own) - set(shared))
    model.load_state_dict({**own, **shared})
    return {"shared_tensors": len(shared), "new_tensors": len(missing),
            "new_parameter_count": int(sum(own[k].numel() for k in missing if own[k].is_floating_point()))}
