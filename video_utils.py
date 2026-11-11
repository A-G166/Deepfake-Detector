"""Video/face helpers built on OpenCV only (no torch needed).

Face detection uses MTCNN (facenet-pytorch) when it is installed and falls
back to OpenCV's Haar cascade otherwise, so the project still runs if
facenet-pytorch is hard to install.
"""
import cv2
import numpy as np


def frame_indices(total, every_n=15, max_frames=32):
    """Indices of frames to sample: every `every_n`-th, capped at `max_frames`
    (spread evenly over the video if there would be more)."""
    if total <= 0:
        return None
    idx = list(range(0, total, every_n))
    if len(idx) > max_frames:
        idx = sorted({int(i) for i in np.linspace(0, total - 1, max_frames)})
    return idx


def sample_frames(path, every_n=15, max_frames=32):
    """Return [(frame_index, rgb_array), ...] sampled from a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = frame_indices(total, every_n, max_frames)
    out = []
    if idx is None:  # container doesn't report a length: read sequentially
        i = 0
        while len(out) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every_n == 0:
                out.append((i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            i += 1
    else:
        wanted = set(idx)
        for i in range(idx[-1] + 1):
            if not cap.grab():
                break
            if i in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    out.append((i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


class FaceDetector:
    def __init__(self, device="cpu"):
        self.mtcnn = None
        self.haar = None
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(keep_all=True, device=device)
            self.backend = "mtcnn"
        except Exception:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar = cv2.CascadeClassifier(cascade)
            self.backend = "haar"
            if self.haar.empty():
                raise RuntimeError(
                    "No face detector available. Install MTCNN with "
                    "'pip install facenet-pytorch --no-deps', or use a standard "
                    "opencv-python build that ships haarcascade_frontalface_default.xml.")

    def detect(self, rgb):
        """Return the largest face as (x1, y1, x2, y2), or None."""
        if self.mtcnn is not None:
            from PIL import Image
            boxes, _ = self.mtcnn.detect(Image.fromarray(rgb))
            if boxes is None or len(boxes) == 0:
                return None
            b = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            return tuple(float(v) for v in b)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (float(x), float(y), float(x + w), float(y + h))


def _detect_with_landmarks(self, rgb):
    """Largest face as (box, points) where points is a (5,2) array in the order
    left eye, right eye, nose, left mouth corner, right mouth corner. MTCNN only."""
    if self.mtcnn is None:
        raise RuntimeError("Landmarks need MTCNN: pip install facenet-pytorch --no-deps")
    from PIL import Image
    boxes, _, points = self.mtcnn.detect(Image.fromarray(rgb), landmarks=True)
    if boxes is None or len(boxes) == 0 or points is None:
        return None
    i = max(range(len(boxes)), key=lambda k: (boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]))
    return tuple(float(v) for v in boxes[i]), np.asarray(points[i], dtype=float)


FaceDetector.detect_with_landmarks = _detect_with_landmarks


def crop_face(rgb, box, margin=0.2, size=224):
    """Crop a face with a margin (keeps blending edges visible) and resize."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = rgb.shape[:2]
    x1, x2 = max(int(x1 - margin * w), 0), min(int(x2 + margin * w), W)
    y1, y2 = max(int(y1 - margin * h), 0), min(int(y2 + margin * h), H)
    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (size, size))


def read_image(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
