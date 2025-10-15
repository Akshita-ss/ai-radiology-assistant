# ai_radiology_app.py
# Futuristic Streamlit prototype for "AI Radiology Assistant" (SIMULATED)
# Not for clinical or diagnostic use.

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import base64

# --- Page config ---
st.set_page_config(page_title="AI Radiology Assistant — Prototype",
                   layout="wide", initial_sidebar_state="expanded")

# --- CSS / Dark Theme ---
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 10% 10%, #0f1724 0%, #041021 40%, #00070a 100%);
    color: #cfe8ff;
}
.block-container { padding-top: 1rem; }
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #021021, #001018);
    color: #cfe8ff;
}
h1 { font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; color: #a8e0ff; }
.neon { color: #7fffd4; text-shadow: 0 0 10px rgba(127,255,212,0.12); }
.card { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px; }
.small { font-size: 12px; color: #9fb8c8; }
a { color: #80e6ff; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.markdown("## ⚙ Controls")
model_choice = st.sidebar.selectbox("Model (prototype)", ["Mock-CNN (Simulated)", "Edge-Heatmap (Faster)"])
threshold = st.sidebar.slider("Sensitivity threshold", 0.05, 0.6, 0.18, 0.01)
overlay_alpha = st.sidebar.slider("Overlay opacity", 0.15, 0.9, 0.45, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("💡 *This is a simulated prototype using filters and heatmaps — replace with a trained CNN + Grad-CAM for production.*")

# --- Main layout ---
col1, col2 = st.columns([1, 1.15])
with col1:
    st.markdown("# <span class='neon'>AI Radiology Assistant</span>", unsafe_allow_html=True)
    st.markdown("*Prototype — Futuristic dark UI*  \nUpload an X-ray image (PNG/JPG) to simulate detection.")
    uploaded = st.file_uploader("Upload X-ray image", type=['png', 'jpg', 'jpeg'])
    st.markdown("---")
    run_button = st.button("🔎 Run Analysis")

with col2:
    st.markdown("### Preview / Output")
    output_container = st.empty()

# --- Utility functions ---
def read_image_to_rgb(file) -> np.ndarray:
    image = Image.open(file).convert('RGB')
    return np.array(image)

def resize_keep_aspect(img_rgb: np.ndarray, target_size=512):
    h, w = img_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_x = target_size - new_w
    pad_y = target_size - new_h
    top, bottom = pad_y // 2, pad_y - pad_y // 2
    left, right = pad_x // 2, pad_x - pad_x // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

def preprocess_gray(img_rgb: np.ndarray, size=(512,512)):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    resized = cv2.resize(clahe, size, interpolation=cv2.INTER_AREA)
    norm = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

def simulate_heatmap_pretrained_like(img_gray: np.ndarray):
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    edges = np.hypot(sobelx, sobely)
    edges = np.uint8(255 * edges / edges.max()) if edges.max() > 0 else np.zeros_like(img_gray, np.uint8)
    blur = cv2.GaussianBlur(edges, (31, 31), 0)
    combined = cv2.normalize(img_gray.astype(np.float32)*0.6 + blur.astype(np.float32)*1.4, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(combined)

def make_overlay_rgb(orig_rgb: np.ndarray, heatmap: np.ndarray, alpha=0.45):
    color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig_rgb, 1 - alpha, color_map, alpha, 0)

def extract_regions(heatmap: np.ndarray, thresh_val=0.18, min_area=100):
    hm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(hm, int(thresh_val * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, w, h, area))
    return sorted(rects, key=lambda x: x[4], reverse=True)

def score_from_heatmap(heatmap: np.ndarray):
    s = float(np.sum(heatmap) / (heatmap.size * 255.0))
    return min(1.0, max(0.0, (s - 0.02) * 1.8))

def make_download_link(text, filename="report.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download report</a>'

# --- Main Analysis ---
if uploaded is not None and run_button:
    try:
        raw_rgb = read_image_to_rgb(uploaded)
        orig_rgb = resize_keep_aspect(raw_rgb, 512)
        pre_gray = preprocess_gray(orig_rgb)

        if model_choice == "Mock-CNN (Simulated)":
            heat = simulate_heatmap_pretrained_like(pre_gray)
        else:
            lap = cv2.Laplacian(pre_gray, cv2.CV_64F)
            lap_u8 = np.uint8(np.absolute(lap))
            heat = cv2.normalize(pre_gray.astype(np.float32) + lap_u8.astype(np.float32)*1.2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        overlay = make_overlay_rgb(orig_rgb, heat, alpha=overlay_alpha)
        score = score_from_heatmap(heat)
        rects = extract_regions(heat, thresh_val=threshold, min_area=90)

        with output_container.container():
            c1, c2 = st.columns([0.6, 0.4])
            with c1:
                st.image(overlay, caption="AI Overlay — Suspicion Score", use_container_width=True)
                st.image(heat, clamp=True, channels="GRAY", use_container_width=True)
            with c2:
                st.markdown("### 🔍 Findings")
                if len(rects) == 0:
                    st.success("No significant suspicious regions detected (prototype).")
                else:
                    st.warning(f"{len(rects)} suspicious region(s) flagged (prototype).")

                st.markdown("---")
                for i, (x, y, w, h, a) in enumerate(rects[:6]):
                    st.markdown(f"- Region {i+1}: bbox({x},{y},{w},{h}), area={int(a)}")

                st.markdown("---")
                if rects:
                    boxed = overlay.copy()
                    for (x, y, w, h, _) in rects[:6]:
                        cv2.rectangle(boxed, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    st.image(boxed, caption="Overlay with bounding boxes", use_container_width=True)

                report = (
                    f"AI Radiology Assistant — Prototype Report\n\n"
                    f"Score: {score:.2f}\n"
                    f"Model: {model_choice}\n"
                    f"Regions detected: {len(rects)}\n"
                    f"Threshold: {threshold}\n\n"
                    "⚠ Note: This is a simulated prototype — not for clinical use."
                )
                st.markdown(make_download_link(report), unsafe_allow_html=True)

            st.markdown("---")
            st.image(overlay, caption="Final annotated overlay (for review)", use_container_width=True)
    except Exception as e:
        st.error(f"Error during processing: {e}")
else:
    with output_container.container():
        st.image("https://images.unsplash.com/photo-1605902711622-cfb43c44367e?auto=format&fit=crop&w=1200&q=60",
                 caption="Futuristic radiology concept (placeholder)", use_container_width=True)
        st.markdown("<div class='small'>Upload an X-ray and click 'Run Analysis' to simulate detection.</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Prototype — *AI Radiology Assistant*  •  Not for clinical use • Built with ❤️ in Python (Streamlit + OpenCV).")
