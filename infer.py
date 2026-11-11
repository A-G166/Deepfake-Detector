"""Run the detector on a video or image.

    python infer.py --ckpt checkpoints/best.pt --video clip.mp4 --gradcam_dir out
    python infer.py --ckpt checkpoints/best.pt --image face.jpg
"""
import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

from aggregate import aggregate_scores, verdict
from common import eval_transform, load_model
from video_utils import FaceDetector, crop_face, read_image, sample_frames


class Predictor:
    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.img_size, self.threshold = load_model(ckpt_path, self.device)
        self.detector = FaceDetector(self.device)
        self.tf = eval_transform(self.img_size)

    @torch.no_grad()
    def score_crops(self, crops, batch=16):
        """Fake probability for each RGB uint8 face crop."""
        probs = []
        for i in range(0, len(crops), batch):
            x = torch.stack([self.tf(Image.fromarray(c)) for c in crops[i:i + batch]]).to(self.device)
            probs += torch.sigmoid(self.model(x)).cpu().tolist()
        return probs

    def _result(self, probs, method, **extra):
        p = aggregate_scores(probs, method)
        label, conf = verdict(p, self.threshold)
        return {"label": label, "fake_prob": p, "confidence": conf, "frame_probs": probs,
                "method": method, **extra}

    def predict_video(self, path, every_n=15, max_frames=32, method="mean"):
        frames = sample_frames(path, every_n, max_frames)
        crops, idxs = [], []
        for i, rgb in frames:
            box = self.detector.detect(rgb)
            if box is None:
                continue
            crop = crop_face(rgb, box, 0.2, self.img_size)
            if crop is not None:
                crops.append(crop)
                idxs.append(i)
        if not crops:
            return {"error": f"No face found in any of the {len(frames)} sampled frames."}
        return self._result(self.score_crops(crops), method, crops=crops, frame_indices=idxs,
                            n_sampled=len(frames))

    def predict_image(self, path):
        rgb = read_image(path)
        box = self.detector.detect(rgb)
        crop = crop_face(rgb, box, 0.2, self.img_size) if box else cv2.resize(rgb, (self.img_size, self.img_size))
        return self._result(self.score_crops([crop]), "mean", crops=[crop], frame_indices=[0],
                            face_found=box is not None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--image")
    p.add_argument("--every_n", type=int, default=15)
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--method", default="mean", choices=["mean", "median", "topk", "vote"])
    p.add_argument("--gradcam_dir", help="save Grad-CAM overlays of the 3 most suspicious frames here")
    args = p.parse_args()

    pred = Predictor(args.ckpt)
    res = pred.predict_video(args.video, args.every_n, args.max_frames, args.method) if args.video \
        else pred.predict_image(args.image)
    if "error" in res:
        print(res["error"])
        return
    print(f"Verdict: {res['label']}  (fake score {res['fake_prob']:.3f}, confidence {res['confidence']:.2f}, "
          f"aggregation={res['method']}, {len(res['frame_probs'])} faces scored)")
    print("Per-frame fake scores:", [round(v, 2) for v in res["frame_probs"]])

    if args.gradcam_dir:
        from gradcam import explain
        out = Path(args.gradcam_dir)
        out.mkdir(parents=True, exist_ok=True)
        top = sorted(range(len(res["frame_probs"])), key=lambda i: -res["frame_probs"][i])[:3]
        for rank, i in enumerate(top, 1):
            overlay, prob = explain(pred.model, res["crops"][i], pred.device, pred.img_size)
            f = out / f"gradcam_{rank}_frame{res['frame_indices'][i]}_p{prob:.2f}.jpg"
            cv2.imwrite(str(f), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print("saved", f)


if __name__ == "__main__":
    main()
