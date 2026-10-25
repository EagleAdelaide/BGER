#!/usr/bin/env python3
"""Parameter accounting per arm (CPU, random input; no data, no weights download).

total: all trainable parameters; training_graph: parameters that receive a gradient
from the arm's training loss on a boxed batch; retrieval: parameters on the path to
the retrieval embedding z (gradient of z), i.e. what a deployed ranker evaluates.
Also checks that every arm loads identical shared tensors from one common template.
"""
import json, sys
from pathlib import Path
CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
import torch
from train_web_arms import build_model, step_losses, WebBGER
from abn_box import load_common_template
from e1_common import state_hash
from e1_common import ARMS

N_IDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3516
torch.manual_seed(0)
template = {k: v.clone() for k, v in WebBGER(N_IDS, 'ce', weights=None).state_dict().items()}
x = torch.randn(2, 3, 224, 224); y = torch.tensor([0, 1])
mask = torch.zeros(2, 224, 224); mask[:, 64:160, 64:160] = 1; has = torch.tensor([True, True])
rows = {}; shared = {}
for arm in ARMS:
    m = build_model(N_IDS, arm, weights=None, init_seed=0)
    if arm == 'abn_box': load_common_template(m, template)
    else: m.load_state_dict(template)
    sd = m.state_dict()
    shared[arm] = state_hash({k: sd[k] for k in sd if k.startswith(('backbone.', 'head.'))})
    params = dict(m.named_parameters()); total = sum(p.numel() for p in params.values())
    m.train(); m.zero_grad(); loss, _ = step_losses(m, x, y, mask, has, 2.0, arm); loss.backward()
    train_used = {k for k, p in params.items() if p.grad is not None and p.grad.abs().sum() > 0}
    m.eval(); m.zero_grad(); z = m.pool(m.features(x[:1])); z.sum().backward()
    retr = {k for k, p in params.items() if p.grad is not None and p.grad.abs().sum() > 0}
    rows[arm] = dict(total_parameters=total,
                     training_graph_parameters=sum(params[k].numel() for k in train_used),
                     retrieval_parameters=sum(params[k].numel() for k in retr),
                     not_in_retrieval=sorted({k.split('.')[0] + ('.' + k.split('.')[1] if k.startswith('backbone') else '') for k in params if k not in retr}))
out = dict(n_ids=N_IDS, arms=rows, shared_backbone_head_hash=shared,
           shared_tensors_identical=len(set(shared.values())) == 1,
           note='retrieval = parameters with nonzero gradient of the pooled embedding; BN running statistics are buffers, not counted; '
                'gate-family arms differ only in loss/pooling, ABN-box adds an attention branch (not an equal-parameter control)')
print(json.dumps(out, indent=1))
