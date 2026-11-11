"""Streamlit demo: upload a video or image, get a verdict, confidence and Grad-CAM.

    streamlit run app.py
"""
import html
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from common import eval_transform
from gradcam import GradCAM
from infer import Predictor

st.set_page_config(page_title="Deepfake Detector", page_icon="◐", layout="wide")

PAPER, CARD, INK, MUTE, RULE = "#F3EFE6", "#FAF8F2", "#16140F", "#6F6A5D", "#D9D3C4"
FAKE, REAL = "#C73E1D", "#2F6B4F"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {{ color-scheme: light; }}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: {PAPER} !important; color: {INK}; }}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}
.block-container {{ max-width: 1080px; padding: 3.2rem 1.5rem 3rem; }}
html, body, .stApp, p, li, label, span, div {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
.stApp h1.hero, .stApp h1.hero span, .stApp h1.hero em, .stApp .section h3, .stApp .section h3 span, .stApp .steps b, .stApp .word
  {{ font-family: 'Fraunces', Georgia, serif !important; }}
.stApp p, .stApp label, .stApp li, .stApp span, .stApp div, [data-testid="stMarkdownContainer"] {{ color: {INK}; }}

.stApp .mono, .stApp p.mono {{ font-family: 'IBM Plex Mono', ui-monospace, monospace !important; font-size: 11.5px !important; letter-spacing: .09em; text-transform: uppercase; color: {MUTE} !important; margin: 0; }}
.mast {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid {INK}; padding-bottom: 10px; }}
.stApp h1.hero {{ font-family: 'Fraunces', Georgia, serif !important; font-weight: 400 !important; font-size: clamp(40px, 6.4vw, 76px) !important;
  line-height: 1.02 !important; letter-spacing: -.02em !important; margin: 34px 0 14px !important; color: {INK} !important; padding: 0 !important; }}
.stApp h1.hero em {{ font-style: italic; color: {FAKE} !important; font-family: 'Fraunces', Georgia, serif !important; }}
.lede {{ max-width: 560px; font-size: 17px; line-height: 1.55; color: {MUTE} !important; margin: 0 0 26px; }}

[data-testid="stFileUploaderDropzone"] {{ background: {CARD}; border: 1.5px dashed {INK}; border-radius: 2px; padding: 26px; }}
[data-testid="stFileUploaderDropzone"] * {{ color: {INK} !important; }}
[data-testid="stFileUploaderDropzone"] button {{ background: transparent; border: 1px solid {INK}; border-radius: 2px; color: {INK} !important; }}
[data-testid="stFileUploaderFile"] * {{ color: {INK} !important; }}
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] div, [data-testid="stFileUploaderFileData"] {{ background: transparent !important; }}
[data-testid="stFileUploaderFile"] svg {{ fill: {INK} !important; color: {INK} !important; }}

.stApp .stButton > button, .stApp .stButton > button:focus {{ background: {INK} !important; color: {PAPER} !important; border: 1px solid {INK} !important; border-radius: 2px !important;
  font-family: 'IBM Plex Mono', monospace; letter-spacing: .08em; text-transform: uppercase; font-size: 13px; padding: .65rem 1.6rem; }}
.stApp .stButton > button:hover {{ background: {FAKE} !important; border-color: {FAKE} !important; }}
.stApp .stButton > button p {{ color: {PAPER} !important; font-family: 'IBM Plex Mono', monospace; }}

[data-testid="stExpander"] {{ background: transparent; border: none; border-top: 1px solid {RULE}; border-radius: 0; }}
[data-testid="stExpander"] summary {{ padding-left: 0; }}
[data-testid="stExpander"] summary p {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: .09em; text-transform: uppercase; color: {MUTE} !important; }}
[data-testid="stExpander"] details {{ border: none; }}
[data-testid="stTextInput"] input, [data-baseweb="select"] > div {{ background: {CARD} !important; border-radius: 2px; color: {INK} !important; border-color: {RULE} !important; }}
[data-testid="stSlider"] [role="slider"] {{ background: {INK}; }}
[data-testid="stSlider"] div[data-testid="stTickBarMin"], [data-testid="stSlider"] div[data-testid="stTickBarMax"] {{ color: {MUTE}; }}
[data-testid="stImage"] img {{ width: 100%; border-radius: 1px; }}
[data-testid="stAlert"] {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 2px; }}

.steps {{ display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid {INK}; margin-top: 38px; }}
.steps div {{ padding: 16px 18px 6px 0; }}
.steps div + div {{ border-left: 1px solid {RULE}; padding-left: 18px; }}
.steps b {{ display: block; font-family: 'Fraunces', Georgia, serif; font-weight: 400; font-size: 22px; margin: 6px 0 4px; color: {INK}; }}
.steps span.t {{ font-size: 14px; line-height: 1.5; color: {MUTE} !important; }}

.sheet {{ border-top: 1px solid {INK}; margin-top: 34px; padding-top: 6px; }}
.verdict-grid {{ display: grid; grid-template-columns: 1.15fr 1fr; gap: 0; margin-top: 14px; }}
.verdict {{ padding: 6px 28px 26px 0; }}
.word {{ font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: clamp(64px, 10vw, 124px); line-height: .95; letter-spacing: -.03em; margin: 10px 0 12px; }}
.qual {{ font-size: 15px; margin: 0 0 22px; color: {INK}; }}
.gauge .track {{ position: relative; height: 10px; background: {RULE}; border-radius: 1px; }}
.gauge .fill {{ position: absolute; left: 0; top: 0; bottom: 0; }}
.gauge .tick {{ position: absolute; top: -6px; bottom: -6px; width: 0; border-left: 1.5px solid {INK}; }}
.gauge .scale {{ display: flex; justify-content: space-between; margin-top: 9px; }}
.facts {{ border-left: 1px solid {RULE}; padding: 6px 0 0 28px; }}
.facts .row {{ display: flex; justify-content: space-between; align-items: baseline; padding: 11px 0; border-bottom: 1px solid {RULE}; }}
.facts .row .v {{ font-family: 'IBM Plex Mono', monospace; font-size: 17px; color: {INK}; }}
.section {{ margin-top: 34px; border-top: 1px solid {RULE}; padding-top: 12px; }}
.section h3 {{ font-family: 'Fraunces', Georgia, serif; font-weight: 400; font-size: 26px; margin: 4px 0 4px; color: {INK}; padding: 0; }}
.section p.sub {{ font-size: 14px; color: {MUTE} !important; margin: 0 0 14px; max-width: 640px; }}
.strip {{ position: relative; display: flex; align-items: flex-end; gap: 3px; height: 120px; border-bottom: 1px solid {INK}; }}
.strip .b {{ flex: 1 1 0; min-width: 2px; }}
.strip .line {{ position: absolute; left: 0; right: 0; border-top: 1px dashed {INK}; opacity: .55; }}
.cap {{ margin: 6px 0 22px; }}
.note {{ border-top: 1px solid {RULE}; margin-top: 34px; padding-top: 12px; font-size: 13.5px; line-height: 1.55; color: {MUTE} !important; max-width: 760px; }}
.foot {{ margin-top: 44px; padding-top: 12px; border-top: 1px solid {INK}; display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap; }}
@media (max-width: 760px) {{ .verdict-grid {{ grid-template-columns: 1fr; }} .facts {{ border-left: none; padding-left: 0; }} .steps {{ grid-template-columns: 1fr; }}
  .steps div + div {{ border-left: none; border-top: 1px solid {RULE}; padding-left: 0; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def h(*parts):
    """Join HTML pieces without newlines/indentation (markdown would turn indented lines into code)."""
    return "".join(parts)


st.markdown(h(
    '<div class="mast"><span class="mono">Deepfake detection &middot; FaceForensics++</span>',
    '<span class="mono">EfficientNet-B0 &middot; Grad-CAM</span></div>',
    '<h1 class="hero">Is this face real,<br>or <em>made?</em></h1>',
    '<p class="lede">Upload a short video or a face photo. The model samples frames, finds the face, scores each one, '
    'and shows which parts of the face drove its decision.</p>'), unsafe_allow_html=True)

ckpt, every_n, max_frames, method, n_cam = "checkpoints/best.pt", 15, 32, "mean", 3


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(path):
    return Predictor(path)


def paper_overlay(rgb, cam):
    """Tint only the hot regions in one colour (no rainbow colormap)."""
    hh, ww = rgb.shape[:2]
    a = cv2.resize(cam.astype(np.float32), (ww, hh), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    a = (0.82 * a ** 1.4)[..., None]
    tint = np.array([199, 62, 29], dtype=np.float32)
    return np.uint8(rgb * (1 - a) + tint * a)


def exhibit(pred, crop):
    """(face | attention) side by side, and the fake probability."""
    x = eval_transform(pred.img_size)(Image.fromarray(crop)).unsqueeze(0).to(pred.device)
    cam, prob = GradCAM(pred.model)(x)
    base = cv2.resize(crop, (pred.img_size, pred.img_size))
    gap = np.full((base.shape[0], 8, 3), (243, 239, 230), dtype=np.uint8)
    return np.hstack([base, gap, paper_overlay(base, cam)]), prob


def test_summary(ckpt_path):
    """One-line held-out performance of this checkpoint, if train.py saved it."""
    try:
        r = json.loads((Path(ckpt_path).parent / "results.json").read_text())["test_video_level"]
        return (f"On held-out test videos this checkpoint reached {r['accuracy']:.0%} accuracy "
                f"(AUC {r['auc']:.2f}) but caught only {r['recall']:.0%} of fakes.")
    except Exception:
        return None


with st.expander("Instrument settings"):
    ckpt = st.text_input("Checkpoint", ckpt)
    c1, c2 = st.columns(2)
    every_n = c1.slider("Sample 1 frame in every N", 5, 60, every_n)
    max_frames = c2.slider("Max frames to score", 4, 64, max_frames)
    c3, c4 = st.columns(2)
    method = c3.selectbox("Aggregation", ["mean", "median", "topk", "vote"])
    n_cam = c4.slider("Grad-CAM frames to show", 1, 6, n_cam)

if not Path(ckpt).is_file():
    st.warning(f"No checkpoint at `{ckpt}`. Train one first, or point to a checkpoint such as `checkpoints300/best.pt`.")
    st.stop()

pred = get_predictor(ckpt)
up = st.file_uploader("Upload a video or a face image", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"],
                      label_visibility="collapsed")

if up is None:
    st.markdown(h(
        '<div class="steps">',
        '<div><span class="mono">01</span><b>Sample</b><span class="t">One frame in every 15, so a short clip gives a few dozen looks at the face.</span></div>',
        '<div><span class="mono">02</span><b>Score</b><span class="t">Each face crop gets a fake score; the scores are averaged into one verdict.</span></div>',
        '<div><span class="mono">03</span><b>Explain</b><span class="t">Grad-CAM marks the facial regions that pushed the score toward fake.</span></div>',
        '</div>'), unsafe_allow_html=True)
elif st.button("Run analysis", type="primary"):
    suffix = Path(up.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(up.getbuffer())
        path = tmp.name
    try:
        with st.spinner("Detecting faces and scoring frames..."):
            if suffix in (".jpg", ".jpeg", ".png"):
                res = pred.predict_image(path)
            else:
                res = pred.predict_video(path, every_n, max_frames, method)
    finally:
        os.remove(path)

    if "error" in res:
        st.error(res["error"])
        st.stop()
    if res.get("face_found") is False:
        st.info("No face was detected, so the whole image was scored. Results are less reliable.")

    p, conf, thr = res["fake_prob"], res["confidence"], getattr(pred, "threshold", 0.5)
    is_fake = res["label"] == "FAKE"
    color = FAKE if is_fake else REAL
    word = "Fake" if is_fake else "Real"
    qual = ("High confidence." if conf >= 0.8 else "Moderate confidence." if conf >= 0.65
            else "Low confidence: treat this as unsure.")
    n_faces = len(res["frame_probs"])
    sampled = res.get("n_sampled", n_faces)
    rows = [("Fake score", f"{p:.2f}"), ("Confidence", f"{conf:.0%}"),
            ("Faces scored", f"{n_faces} of {sampled}"), ("Aggregation", res["method"]),
            ("Decision line", f"{thr:.2f}")]
    facts = "".join(f'<div class="row"><span class="mono">{k}</span><span class="v">{html.escape(v)}</span></div>' for k, v in rows)
    st.markdown(h(
        '<div class="sheet"><span class="mono">Result &middot; ', html.escape(up.name), '</span>',
        '<div class="verdict-grid"><div class="verdict">',
        f'<span class="mono">Verdict</span><div class="word" style="color:{color}">{word}</div>',
        f'<p class="qual">{qual}</p>',
        f'<div class="gauge"><div class="track"><div class="fill" style="width:{p * 100:.1f}%;background:{color}"></div>',
        f'<div class="tick" style="left:{thr * 100:.1f}%"></div></div>',
        '<div class="scale"><span class="mono">0 real</span><span class="mono">decision line</span><span class="mono">1 fake</span></div></div>',
        f'</div><div class="facts">{facts}</div></div></div>'), unsafe_allow_html=True)

    if n_faces > 1:
        bars = "".join(
            f'<div class="b" title="frame {i}: {v:.2f}" style="height:{max(v, 0.02) * 100:.0f}%;background:{FAKE if v >= thr else REAL}"></div>'
            for i, v in zip(res["frame_indices"], res["frame_probs"]))
        st.markdown(h(
            '<div class="section"><h3>Frame by frame</h3>',
            '<p class="sub">One bar per sampled frame. Red bars lean fake, green lean real; the dashed line is the decision line.</p>',
            f'<div class="strip">{bars}<div class="line" style="bottom:{thr * 100:.0f}%"></div></div>',
            f'<div style="display:flex;justify-content:space-between;margin-top:6px"><span class="mono">frame {res["frame_indices"][0]}</span>',
            f'<span class="mono">frame {res["frame_indices"][-1]}</span></div></div>'), unsafe_allow_html=True)

    st.markdown(h(
        '<div class="section"><h3>Where the model looked</h3>',
        '<p class="sub">Face on the left, attention on the right. Red marks regions that pushed the score toward fake. '
        'Shown for the most suspicious frames.</p></div>'), unsafe_allow_html=True)
    order = sorted(range(n_faces), key=lambda i: -res["frame_probs"][i])[:n_cam]
    for start in range(0, len(order), 3):
        chunk = order[start:start + 3]
        cols = st.columns(3)
        for col, i in zip(cols, chunk):
            img, prob = exhibit(pred, res["crops"][i])
            col.image(img)
            col.markdown(f'<p class="mono cap">Frame {res["frame_indices"][i]} &middot; score {prob:.2f}</p>',
                         unsafe_allow_html=True)

    summary = test_summary(ckpt)
    st.markdown(h(
        '<div class="note"><b>How far to trust this.</b> The score is the model\'s own output, not a calibrated probability, '
        'and a REAL verdict is not proof that a video is genuine. ',
        html.escape(summary or ""), ' Use it as a screening aid, not as evidence.</div>'), unsafe_allow_html=True)

st.markdown(h(
    '<div class="foot"><span class="mono">SRM Institute of Science and Technology &middot; B.Tech AI</span>',
    '<span class="mono">Nashita &middot; Amritha &middot; Arpitha &middot; Manotharani</span></div>'), unsafe_allow_html=True)
