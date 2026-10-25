"""CUB-200-2011 dataset with bounding-box grounding annotations.

Split policy (no test leakage):
  * Official CUB `train_test_split.txt` defines a train POOL and a TEST set.
  * The TEST set is used untouched for final evaluation.
  * The train POOL is re-split, stratified by class with a fixed seed, into
    train / val / calibration. Temperature Scaling is fit ONLY on calibration.

Geometry (sanity check A):
  * Eval/grounding transform resizes the image to a fixed (S, S) square, so the
    GT bounding box maps EXACTLY by scaling x by S/W_orig and y by S/H_orig.
    Returned bbox is in resized pixel coords, clamped to [0, S]; a binary mask
    built from it is therefore guaranteed to match the (S, S) heatmap frame.
  * Train transform uses RandomResizedCrop+HFlip (boxes not needed for training).
"""
from __future__ import annotations
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from models.classifier import IMAGENET_MEAN, IMAGENET_STD


def _read_table(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    return rows


def load_cub_index(root):
    """Parse CUB metadata. Returns dict id-> {path,label,is_train,bbox(orig)}."""
    img_paths = {int(i): p for i, p in _read_table(os.path.join(root, "images.txt"))}
    labels = {int(i): int(c) - 1 for i, c in _read_table(os.path.join(root, "image_class_labels.txt"))}
    splits = {int(i): int(s) for i, s in _read_table(os.path.join(root, "train_test_split.txt"))}
    bboxes = {}
    for i, x, y, w, h in _read_table(os.path.join(root, "bounding_boxes.txt")):
        bboxes[int(i)] = (float(x), float(y), float(w), float(h))
    index = {}
    for i in img_paths:
        index[i] = {
            "path": os.path.join(root, "images", img_paths[i]),
            "label": labels[i],
            "is_train": splits[i],
            "bbox_orig": bboxes[i],   # (x, y, w, h) in original pixels
        }
    return index


def make_cub_splits(index, val_fraction, calibration_fraction, seed):
    """Stratified split of the official train pool into train/val/calibration."""
    rng = np.random.default_rng(seed)
    train_pool = [i for i, v in index.items() if v["is_train"] == 1]
    test_ids = [i for i, v in index.items() if v["is_train"] == 0]
    by_class = {}
    for i in train_pool:
        by_class.setdefault(index[i]["label"], []).append(i)
    train_ids, val_ids, cal_ids = [], [], []
    for c, ids in by_class.items():
        ids = np.array(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(round(n * val_fraction)))
        n_cal = max(1, int(round(n * calibration_fraction)))
        val_ids += ids[:n_val].tolist()
        cal_ids += ids[n_val:n_val + n_cal].tolist()
        train_ids += ids[n_val + n_cal:].tolist()
    return {"train": sorted(train_ids), "val": sorted(val_ids),
            "calibration": sorted(cal_ids), "test": sorted(test_ids)}


class CUBDataset(Dataset):
    def __init__(self, index, ids, image_size=224, mode="eval"):
        assert mode in ("train", "eval")
        self.index = index
        self.ids = list(ids)
        self.S = image_size
        self.mode = mode
        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        if mode == "train":
            self.aug = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        sid = self.ids[k]
        rec = self.index[sid]
        img = Image.open(rec["path"]).convert("RGB")
        W, H = img.size
        label = rec["label"]
        if self.mode == "train":
            img = self.aug(img)
            x = self.normalize(T.functional.to_tensor(img))
            # bbox not meaningful under random crop; return sentinel
            bbox = torch.tensor([-1, -1, -1, -1], dtype=torch.float32)
        else:
            img = img.resize((self.S, self.S), Image.BILINEAR)
            x = self.normalize(T.functional.to_tensor(img))
            ox, oy, ow, oh = rec["bbox_orig"]
            sx, sy = self.S / W, self.S / H
            x1 = max(0.0, ox * sx); y1 = max(0.0, oy * sy)
            x2 = min(float(self.S), (ox + ow) * sx)
            y2 = min(float(self.S), (oy + oh) * sy)
            bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        return {"image": x, "label": label, "bbox": bbox, "sample_id": sid}


def bbox_to_mask(bbox, size):
    """Filled-rectangle binary mask (size,size) from [x1,y1,x2,y2] coords."""
    m = np.zeros((size, size), dtype=np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(size, x1)); x2 = max(0, min(size, x2))
    y1 = max(0, min(size, y1)); y2 = max(0, min(size, y2))
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    return m
