# Deepfake Detector

A Streamlit web app that screens a video or face photo for signs of manipulation. It returns a **REAL / FAKE** verdict, a frame-by-frame score chart, and **Grad-CAM** heatmaps showing which facial regions pushed the model towards "fake".

Built by students of the B.Tech Artificial Intelligence programme at SRM Institute of Science and Technology, Chennai.

> **This is a screening aid, not proof.** It is a research prototype trained on a small subset of one dataset. Do not use it to make decisions about real people. See [Limitations](#limitations).

<!-- Add a screenshot: upload an image to the repo (e.g. docs/screenshot.png) and uncomment the line below -->
<!-- ![App screenshot](docs/screenshot.png) -->

**Live demo:** https://deepfake-detector-vytd3mb5kaxg4yunpij3as.streamlit.app/

---

## Features

- Upload a **video** (frames are sampled automatically) or a **single face image**
- Face detection with **MTCNN**, with an automatic OpenCV Haar-cascade fallback
- **EfficientNet-B0** classifier fine-tuned on FaceForensics++
- Four ways to combine frame scores into one video verdict: `mean`, `median`, `topk`, `vote`
- **Grad-CAM** explanations for the most suspicious frames
- Adjustable settings: sampling rate, maximum frames, aggregation method, number of Grad-CAM frames
- A command-line tool for batch or scripted use

## How it works

```
video / image
   -> sample frames (every 15th, up to 32)
   -> detect the largest face (MTCNN)
   -> crop with a 20% margin, resize to 160x160
   -> EfficientNet-B0 -> fake probability per frame
   -> aggregate frame scores -> verdict (threshold 0.5)
   -> Grad-CAM on the most suspicious frames
```

**Model:** EfficientNet-B0 pretrained on ImageNet, with the classifier replaced by `Dropout(0.3)` and a single-logit linear head (positive = fake).

**Training:** two phases. First the head only (2 epochs, lr 1e-3), then the whole network fine-tuned (8 epochs, lr 1e-4). Batch size 32, early stopping on validation AUC. Augmentations: random resized crop, horizontal flip, colour jitter, Gaussian blur, and random JPEG re-compression.

**Data:** FaceForensics++ (c23) face crops, split **by video** to avoid frame leakage: 420 train, 90 validation, 90 test videos.

## Results

Evaluated on 90 held-out test videos (45 real, 45 fake), 20 frames per video.

| Metric | Frame level | Video level |
|---|---|---|
| Accuracy | 65.9% | 67.8% |
| AUC | 0.729 | 0.730 |
| Precision | 0.736 | 0.750 |
| Recall | 0.498 | 0.533 |
| F1 | 0.594 | 0.623 |

Video-level confusion matrix: 37 real correctly identified, 8 real wrongly flagged as fake, 21 fakes missed, 24 fakes caught.

The full training history is in [`checkpoints/results.json`](checkpoints/results.json).

## Installation

Requires **Python 3.10 - 3.12**.

```bash
git clone https://github.com/A-G166/Deepfake-Detector.git
cd Deepfake-Detector

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

If the PyTorch CPU index line at the top of `requirements.txt` causes an error, delete that line and install again.

## Usage

### Web app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a video or image, adjust the settings if you like, and click **Run analysis**.

### Command line

```bash
python infer.py --video clip.mp4
python infer.py --video clip.mp4 --method topk --gradcam_dir out
python infer.py --image face.jpg
```

Options: `--every_n` (frame sampling step), `--max_frames`, `--method {mean,median,topk,vote}`, `--gradcam_dir` (saves Grad-CAM overlays of the 3 most suspicious frames).

## Project structure

```
app.py              Streamlit interface
infer.py            Predictor class + command-line tool
common.py           Model, preprocessing and augmentation
video_utils.py      Frame sampling, face detection, cropping
aggregate.py        Frame scores -> video verdict
gradcam.py          Grad-CAM explanations
facenet_pytorch/    Vendored MTCNN (MIT licence)
checkpoints/        best.pt (trained weights) and results.json
```

## Limitations

- **Small training set.** Only 420 training videos; the 90-video test set means every metric has wide uncertainty.
- **Misses many fakes.** Video-level recall is about 53%, and training loss (~0.08) is far below validation loss (> 1.0), a sign of overfitting.
- **One dataset.** Trained only on FaceForensics++ (c23). It has **not** been tested on other datasets such as Celeb-DF, so generalisation is unknown.
- **Not for AI-generated images.** It targets face-swap and reenactment video manipulations. Images from diffusion models or GANs, edited photos, and deepfakes from other tools will often be classified as real.
- **Low resolution.** Faces are analysed at 160x160, and Grad-CAM comes from a coarse 5x5 feature map, so heatmaps are approximate.
- **Uncalibrated confidence.** The confidence value is the model's score, not a true probability.
- **Scope.** Only the largest face per frame is analysed; audio is ignored.

## Future work

- Train on more data: the full FaceForensics++ with all manipulation types, balanced classes
- Cross-dataset evaluation on Celeb-DF (and DFDC)
- Higher input resolution (224 px or more) and stronger regularisation
- Choose the decision threshold from the validation ROC curve; calibrate the confidence score
- Revisit temporal models only if new evidence shows a usable temporal signal

## Responsible use

Deepfake detection is an arms race, and no detector is reliable on its own. Treat any output as one weak signal among many, and never as evidence about a person.

## Team

Nashita, Amritha, Arpitha, Manotharani, B.Tech Artificial Intelligence, SRM Institute of Science and Technology.

## Acknowledgements

- **FaceForensics++:** Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019. The dataset is not included in this repository and is subject to its own terms of use.
- **EfficientNet:** Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
- **MTCNN:** Zhang et al., *Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks*, 2016. Implementation vendored from [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MIT licence).
- **Grad-CAM:** Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017.

## Licence

Released under the MIT licence. See the `LICENSE` file.
