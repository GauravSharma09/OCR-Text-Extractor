"""
OCR Text Extractor - Main Application
======================================
A production-grade OCR app built with Streamlit, EasyOCR, and OpenCV.
Supports images and PDFs with advanced preprocessing.
"""

import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import time
import json
from pathlib import Path
import tempfile
import os

# Import our utility functions
from utils import (
    preprocess_image,
    extract_text_from_image,
    extract_text_from_pdf,
    calculate_confidence,
    save_text_as_txt,
    save_text_as_pdf,
    get_supported_languages,
    validate_file,
)

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Text Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark SaaS Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

  /* ── Root Variables ── */
  :root {
    --bg-primary:    #0a0a0f;
    --bg-secondary:  #111118;
    --bg-card:       #16161f;
    --bg-hover:      #1e1e2a;
    --accent:        #7c6aff;
    --accent-soft:   #a89bff;
    --accent-glow:   rgba(124,106,255,0.15);
    --green:         #00d68f;
    --red:           #ff4f6a;
    --text-primary:  #eeeef5;
    --text-muted:    #8888a8;
    --border:        rgba(255,255,255,0.07);
    --border-accent: rgba(124,106,255,0.4);
  }

  /* ── Global Reset ── */
  html, body, [class*="css"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
  }

  /* ── Streamlit Overrides ── */
  .stApp { background: var(--bg-primary) !important; }
  .block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }
  section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
  }

  /* ── Hero Header ── */
  .hero-header {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
  }
  .hero-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-soft) 60%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    line-height: 1.1 !important;
  }
  .hero-sub {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .hero-badge {
    display: inline-block;
    background: var(--accent-glow);
    border: 1px solid var(--border-accent);
    color: var(--accent-soft);
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 1rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  /* ── Cards ── */
  .ocr-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    transition: border-color 0.2s;
  }
  .ocr-card:hover { border-color: var(--border-accent); }

  /* ── Section Labels ── */
  .section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent-soft);
    margin-bottom: 0.75rem;
  }

  /* ── Upload Zone ── */
  [data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-accent) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: all 0.2s;
  }
  [data-testid="stFileUploader"]:hover {
    background: var(--bg-hover) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 24px var(--accent-glow) !important;
  }
  [data-testid="stFileUploaderDropzone"] { background: transparent !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #9c7fff 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.25rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px var(--accent-glow) !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  /* ── Secondary button ── */
  .btn-secondary > button {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
  }
  .btn-secondary > button:hover {
    border-color: var(--border-accent) !important;
    background: var(--bg-card) !important;
    box-shadow: none !important;
  }

  /* ── Text Area (output) ── */
  .stTextArea textarea {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
    resize: vertical !important;
  }
  .stTextArea textarea:focus { border-color: var(--border-accent) !important; }

  /* ── Selectbox / Slider ── */
  .stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
  }
  .stSlider .st-bj { background: var(--accent) !important; }
  .stCheckbox label { color: var(--text-muted) !important; font-size: 0.85rem !important; }

  /* ── Metrics ── */
  [data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
  }
  [data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
  }

  /* ── Confidence badge ── */
  .conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
  }
  .conf-high  { background: rgba(0,214,143,0.12); color: #00d68f; border: 1px solid rgba(0,214,143,0.3); }
  .conf-med   { background: rgba(255,180,0,0.12);  color: #ffb400; border: 1px solid rgba(255,180,0,0.3); }
  .conf-low   { background: rgba(255,79,106,0.12); color: #ff4f6a; border: 1px solid rgba(255,79,106,0.3); }

  /* ── Alerts ── */
  .stSuccess { background: rgba(0,214,143,0.08) !important; border-color: var(--green) !important; border-radius: 10px !important; }
  .stError   { background: rgba(255,79,106,0.08) !important; border-color: var(--red)   !important; border-radius: 10px !important; }
  .stWarning { background: rgba(255,180,0,0.08)  !important; border-radius: 10px !important; }
  .stInfo    { background: var(--accent-glow)     !important; border-color: var(--accent) !important; border-radius: 10px !important; }

  /* ── Progress ── */
  .stProgress > div > div { background: var(--accent) !important; border-radius: 999px !important; }

  /* ── Sidebar ── */
  .sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-section {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent-soft);
    font-weight: 600;
    margin: 1.25rem 0 0.5rem;
  }

  /* ── Divider ── */
  hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-secondary); }
  ::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 999px; }

  /* ── Hide Streamlit Branding ── */
  #MainMenu, footer, header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0.0
if "word_count" not in st.session_state:
    st.session_state.word_count = 0
if "reader" not in st.session_state:
    st.session_state.reader = None
if "last_lang" not in st.session_state:
    st.session_state.last_lang = []


# ─────────────────────────────────────────────
# SIDEBAR — Settings & Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ Settings</div>', unsafe_allow_html=True)

    # ── Language Selection ──
    st.markdown('<div class="sidebar-section">OCR Language</div>', unsafe_allow_html=True)
    lang_options = get_supported_languages()
    selected_langs = st.multiselect(
        "Languages",
        options=list(lang_options.keys()),
        default=["English"],
        format_func=lambda x: f"{lang_options[x]['flag']} {x}",
        label_visibility="collapsed",
    )
    lang_codes = [lang_options[l]["code"] for l in selected_langs] if selected_langs else ["en"]

    # ── Preprocessing Options ──
    st.markdown('<div class="sidebar-section">Preprocessing</div>', unsafe_allow_html=True)
    use_grayscale   = st.checkbox("Grayscale Conversion",  value=True)
    use_denoise     = st.checkbox("Noise Removal",         value=True)
    use_threshold   = st.checkbox("Adaptive Thresholding", value=True)
    use_deskew      = st.checkbox("Deskew / Straighten",   value=False)
    scale_factor    = st.slider("Upscale Factor", 1.0, 4.0, 2.0, 0.5,
                                 help="Higher = better OCR on small text, slower")

    # ── GPU toggle ──
    st.markdown('<div class="sidebar-section">Performance</div>', unsafe_allow_html=True)
    use_gpu = st.checkbox("Use GPU (if available)", value=False)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:var(--text-muted);line-height:1.7">'
        '🔍 <b>OCR Text Extractor</b><br>'
        'Powered by EasyOCR + OpenCV<br>'
        'No Tesseract required</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-badge">✦ AI-Powered OCR Engine</div>
  <h1>OCR Text Extractor</h1>
  <p class="hero-sub">Upload an image or PDF — extract text instantly</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD / CACHE EASYOCR READER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_reader(lang_codes: tuple, gpu: bool):
    """Load EasyOCR reader — cached so it only loads once per language set."""
    return easyocr.Reader(list(lang_codes), gpu=gpu)


# ─────────────────────────────────────────────
# MAIN LAYOUT — Two Columns
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════
# LEFT COLUMN — Upload + Preview
# ══════════════════════════════════════════════
with col_left:
    st.markdown('<div class="section-label">📁 Upload File</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "pdf"],
        label_visibility="collapsed",
        help="Supports JPG, PNG, BMP, TIFF, WEBP, PDF",
    )

    if uploaded_file:
        # ── Validate file ──
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            st.markdown('<div class="section-label" style="margin-top:1.5rem">🖼 Image Preview</div>',
                        unsafe_allow_html=True)

            file_type = uploaded_file.type

            # ── Show preview ──
            if "pdf" in file_type:
                st.info("📄 PDF detected — all pages will be processed.")
                st.markdown(f"""
                <div class="ocr-card" style="text-align:center;padding:2rem">
                  <div style="font-size:3rem">📑</div>
                  <div style="margin-top:0.75rem;font-family:'Syne',sans-serif;font-weight:700">
                    {uploaded_file.name}
                  </div>
                  <div style="color:var(--text-muted);font-size:0.8rem;margin-top:0.25rem">
                    {uploaded_file.size/1024:.1f} KB
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)
                st.markdown(
                    f'<div style="color:var(--text-muted);font-size:0.75rem;text-align:right">'
                    f'{image.width}×{image.height}px · {uploaded_file.size/1024:.1f} KB</div>',
                    unsafe_allow_html=True,
                )

            # ── EXTRACT button ──
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍  Extract Text", key="extract_btn"):
                if not selected_langs:
                    st.warning("⚠ Please select at least one language.")
                else:
                    # ── Load reader ──
                    with st.spinner("Loading OCR engine…"):
                        reader = load_reader(tuple(lang_codes), use_gpu)

                    preprocessing_cfg = {
                        "grayscale":   use_grayscale,
                        "denoise":     use_denoise,
                        "threshold":   use_threshold,
                        "deskew":      use_deskew,
                        "scale":       scale_factor,
                    }

                    progress = st.progress(0, text="Starting OCR…")
                    start_time = time.time()

                    try:
                        uploaded_file.seek(0)

                        if "pdf" in file_type:
                            # ── PDF extraction ──
                            progress.progress(20, text="Reading PDF…")
                            raw_bytes = uploaded_file.read()
                            progress.progress(50, text="Running OCR on pages…")
                            text, conf = extract_text_from_pdf(raw_bytes, reader, preprocessing_cfg)
                        else:
                            # ── Image extraction ──
                            progress.progress(20, text="Preprocessing image…")
                            img_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                            cv_img    = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                            processed = preprocess_image(cv_img, preprocessing_cfg)
                            progress.progress(60, text="Running OCR…")
                            text, conf = extract_text_from_image(processed, reader)

                        progress.progress(90, text="Finalizing…")

                        # ── Store results ──
                        st.session_state.extracted_text = text.strip()
                        st.session_state.confidence     = conf
                        st.session_state.processing_time = round(time.time() - start_time, 2)
                        st.session_state.word_count     = len(text.split()) if text.strip() else 0

                        progress.progress(100, text="Done ✓")
                        time.sleep(0.3)
                        progress.empty()

                        if text.strip():
                            st.success(f"✅ Extracted {st.session_state.word_count} words in {st.session_state.processing_time}s")
                        else:
                            st.warning("⚠ No text found. Try adjusting preprocessing settings.")

                    except Exception as e:
                        progress.empty()
                        st.error(f"❌ OCR failed: {str(e)}")
    else:
        # ── Placeholder when no file uploaded ──
        st.markdown("""
        <div class="ocr-card" style="text-align:center;padding:3rem 2rem;margin-top:0.5rem">
          <div style="font-size:3.5rem;margin-bottom:1rem">🖼</div>
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem">
            No file selected
          </div>
          <div style="color:var(--text-muted);font-size:0.8rem;margin-top:0.5rem">
            Supports JPG · PNG · BMP · TIFF · WEBP · PDF
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# RIGHT COLUMN — Results
# ══════════════════════════════════════════════
with col_right:
    st.markdown('<div class="section-label">📋 Extracted Text</div>', unsafe_allow_html=True)

    # ── Metrics row ──
    m1, m2, m3 = st.columns(3)
    m1.metric("Words",    st.session_state.word_count)
    m2.metric("Time (s)", st.session_state.processing_time)

    conf_val = st.session_state.confidence
    conf_pct = f"{conf_val:.0%}" if conf_val else "—"
    m3.metric("Confidence", conf_pct)

    # ── Confidence badge ──
    if conf_val:
        if conf_val >= 0.75:
            badge_cls, badge_txt = "conf-high", f"✓ High Confidence ({conf_pct})"
        elif conf_val >= 0.45:
            badge_cls, badge_txt = "conf-med",  f"~ Medium Confidence ({conf_pct})"
        else:
            badge_cls, badge_txt = "conf-low",  f"✗ Low Confidence ({conf_pct})"
        st.markdown(
            f'<div style="margin:0.75rem 0"><span class="conf-badge {badge_cls}">{badge_txt}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Text output ──
    output_text = st.text_area(
        "output",
        value=st.session_state.extracted_text,
        height=340,
        placeholder="Extracted text will appear here after processing…",
        label_visibility="collapsed",
        key="output_area",
    )

    # ── Action buttons ──
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("📋  Copy Text", key="copy_btn"):
            if st.session_state.extracted_text:
                st.code(st.session_state.extracted_text, language=None)
                st.success("Text displayed above — select all & copy!")
            else:
                st.warning("Nothing to copy.")

    with b2:
        if st.session_state.extracted_text:
            txt_bytes = save_text_as_txt(st.session_state.extracted_text)
            st.download_button(
                label="💾  Save as TXT",
                data=txt_bytes,
                file_name="extracted_text.txt",
                mime="text/plain",
                key="dl_txt",
            )
        else:
            st.button("💾  Save as TXT", disabled=True, key="dl_txt_disabled")

    with b3:
        if st.session_state.extracted_text:
            pdf_bytes = save_text_as_pdf(st.session_state.extracted_text)
            if pdf_bytes:
                st.download_button(
                    label="📄  Export PDF",
                    data=pdf_bytes,
                    file_name="extracted_text.pdf",
                    mime="application/pdf",
                    key="dl_pdf",
                )
            else:
                st.button("📄  Export PDF", disabled=True, key="dl_pdf_na",
                          help="Install fpdf2 to enable PDF export")
        else:
            st.button("📄  Export PDF", disabled=True, key="dl_pdf_disabled")

    # ── Clear button ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
    if st.button("🗑  Clear All", key="clear_btn"):
        st.session_state.extracted_text = ""
        st.session_state.confidence     = 0.0
        st.session_state.processing_time = 0.0
        st.session_state.word_count     = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HOW IT WORKS — Info Section
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label" style="text-align:center">⚡ Preprocessing Pipeline</div>',
            unsafe_allow_html=True)

pipe_cols = st.columns(5)
steps = [
    ("🎨", "Grayscale", "Remove colour noise, focus on intensity"),
    ("📐", "Upscale",   "Increase resolution for small text"),
    ("🔇", "Denoise",   "Remove JPEG artifacts and grain"),
    ("⬛", "Threshold", "Binarise image for clear text"),
    ("📏", "Deskew",    "Correct skewed / rotated scans"),
]
for col, (icon, title, desc) in zip(pipe_cols, steps):
    col.markdown(f"""
    <div class="ocr-card" style="text-align:center;padding:1.25rem 1rem">
      <div style="font-size:1.75rem">{icon}</div>
      <div style="font-family:'Syne',sans-serif;font-weight:700;margin:0.4rem 0;font-size:0.9rem">{title}</div>
      <div style="color:var(--text-muted);font-size:0.72rem;line-height:1.5">{desc}</div>
    </div>""", unsafe_allow_html=True)
