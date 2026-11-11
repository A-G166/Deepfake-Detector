# Deepfake Detector (demo app)

Streamlit app: upload a video or face photo, get a verdict, per-frame scores and Grad-CAM.
EfficientNet-B0 trained on FaceForensics++ (c23). Screening aid only, not proof.

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`.
Needs `checkpoints/best.pt` (and optionally `checkpoints/results.json`).
Face detection uses MTCNN (vendored from facenet-pytorch, MIT licence, see facenet_pytorch/LICENSE.md).
