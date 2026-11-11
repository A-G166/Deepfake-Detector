"""Frame-score -> video-level decision (no torch needed).

These methods work on the sparse frame sampling used by the pipeline
(e.g. 1 frame in 15). A recurrent model (LSTM/GRU) would need short runs of
consecutive frames instead; see README "Next steps".
"""
import math

import numpy as np


def aggregate_scores(probs, method="mean", topk_frac=0.3):
    """Combine per-frame fake probabilities into one video-level fake probability.

    mean   - average of all frames (smooth, default)
    median - robust to a few odd frames
    topk   - mean of the most suspicious `topk_frac` of frames (catches
             manipulation that only appears in part of the video)
    vote   - fraction of frames individually called fake
    """
    p = np.asarray(probs, dtype=float)
    if p.size == 0:
        raise ValueError("no frame scores to aggregate")
    if method == "mean":
        return float(p.mean())
    if method == "median":
        return float(np.median(p))
    if method == "topk":
        k = max(1, math.ceil(topk_frac * p.size))
        return float(np.sort(p)[-k:].mean())
    if method == "vote":
        return float((p >= 0.5).mean())
    raise ValueError(f"unknown aggregation method: {method}")


def verdict(fake_prob, threshold=0.5):
    """Return (label, confidence). Confidence is the model's score for the
    chosen label; it is NOT a calibrated probability."""
    label = "FAKE" if fake_prob >= threshold else "REAL"
    confidence = max(fake_prob, 1.0 - fake_prob)
    return label, float(confidence)
