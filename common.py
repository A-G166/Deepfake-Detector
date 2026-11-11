"""Shared model/data code: EfficientNet-B0 detector, augmentation, dataset."""
import io
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]  # ImageNet statistics (the backbone was pretrained with these)
STD = [0.229, 0.224, 0.225]


class RandomJPEG:
    """Randomly re-compress an image as JPEG, so the model sees the kind of
    compression artifacts real-world videos have."""

    def __init__(self, p=0.3, quality=(30, 90)):
        self.p, self.quality = p, quality

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def train_transform(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
        RandomJPEG(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class FaceDataset(Dataset):
    """samples: list of (path, label, group_id). Returns (image, label, index)."""

    def __init__(self, samples, tf):
        self.samples, self.tf = samples, tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, _ = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), torch.tensor(label, dtype=torch.float32), i


class DeepfakeNet(nn.Module):
    """EfficientNet-B0 with a single-logit head (positive = fake)."""

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    @property
    def features(self):
        return self.backbone.features

    @property
    def head(self):
        return self.backbone.classifier

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # logits, shape (B,)


def load_model(ckpt_path, device="cpu"):
    """Load a checkpoint written by train.py. Returns (model, img_size, threshold)."""
    try:  # our own checkpoint: newer torch defaults to weights_only=True, which rejects numpy scalars
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:  # older torch has no weights_only argument
        ck = torch.load(ckpt_path, map_location=device)
    model = DeepfakeNet(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, int(ck.get("img_size", 224)), float(ck.get("threshold", 0.5))
