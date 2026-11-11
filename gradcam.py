"""Grad-CAM: which face regions pushed the model toward "fake".

Hooks the last convolutional block of the EfficientNet backbone, weights its
feature maps by the gradient of the fake logit, and overlays the result.
"""
import cv2
import numpy as np
import torch
from PIL import Image

from common import eval_transform


class GradCAM:
    def __init__(self, model, layer=None):
        self.model = model
        self.layer = layer if layer is not None else model.features[-1]

    def __call__(self, x):
        """x: (1,3,H,W) normalised tensor. Returns (cam in [0,1] of shape (h,w), fake_prob)."""
        store = {}

        def fwd(_module, _inp, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        handle = self.layer.register_forward_hook(fwd)
        try:
            self.model.eval()
            x = x.clone().requires_grad_(True)
            with torch.enable_grad():
                logit = self.model(x)[0]
                self.model.zero_grad(set_to_none=True)
                logit.backward()
        finally:
            handle.remove()

        act, grad = store["act"].detach(), store["grad"].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1))[0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, float(torch.sigmoid(logit.detach()))


def overlay_cam(rgb, cam, alpha=0.45):
    """Blend a heatmap onto an RGB uint8 image."""
    h, w = rgb.shape[:2]
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (w, h))), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb + alpha * heat)


def explain(model, crop_rgb, device="cpu", img_size=224):
    """Return (overlay_rgb_uint8, fake_prob) for one face crop."""
    x = eval_transform(img_size)(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    cam, prob = GradCAM(model)(x)
    base = cv2.resize(crop_rgb, (img_size, img_size))
    return overlay_cam(base, cam), prob
