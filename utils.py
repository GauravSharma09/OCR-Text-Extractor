"""
utils.py — OCR Text Extractor Utilities
=========================================
All helper functions: preprocessing, OCR, export, validation.
Kept separate from app.py for clean, modular code.
"""

import cv2
import numpy as np
from PIL import Image
import io
import math
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# SUPPORTED LANGUAGES
# ─────────────────────────────────────────────
def get_supported_languages() -> dict:
    """
    Returns a dict of display name → {code, flag} for EasyOCR languages.
    Add more from: https://www.jaided.ai/easyocr/
    """
    return {
        "English":            {"code": "en",  "flag": "🇬🇧"},
        "Hindi":              {"code": "hi",  "flag": "🇮🇳"},
        "French":             {"code": "fr",  "flag": "🇫🇷"},
        "German":             {"code": "de",  "flag": "🇩🇪"},
        "Spanish":            {"code": "es",  "flag": "🇪🇸"},
        "Italian":            {"code": "it",  "flag": "🇮🇹"},
        "Portuguese":         {"code": "pt",  "flag": "🇵🇹"},
        "Chinese (Simplified)": {"code": "ch_sim", "flag": "🇨🇳"},
        "Japanese":           {"code": "ja",  "flag": "🇯🇵"},
        "Korean":             {"code": "ko",  "flag": "🇰🇷"},
        "Arabic":             {"code": "ar",  "flag": "🇸🇦"},
        "Russian":            {"code": "ru",  "flag": "🇷🇺"},
    }


# ─────────────────────────────────────────────
# FILE VALIDATION
# ─────────────────────────────────────────────
ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/tiff", "image/webp", "application/pdf"
}
MAX_SIZE_MB = 25


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validates an uploaded Streamlit file object.

    Returns:
        (True, "") if valid
        (False, error_message) if invalid
    """
    if uploaded_file is None:
        return False, "No file uploaded."

    # ── Check MIME type ──
    if uploaded_file.type not in ALLOWED_TYPES:
        return False, (
            f"Unsupported file type '{uploaded_file.type}'. "
            "Please upload JPG, PNG, BMP, TIFF, WEBP, or PDF."
        )

    # ── Check file size ──
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_SIZE_MB} MB."

    return True, ""


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Apply a series of preprocessing steps to improve OCR accuracy.

    Steps (all optional, controlled by cfg):
        1. Upscale — makes small text readable
        2. Grayscale — removes colour noise
        3. Denoise — removes JPEG/sensor grain
        4. Adaptive threshold — binarises the image
        5. Deskew — corrects rotation

    Args:
        img: OpenCV BGR image (numpy array)
        cfg: dict with keys: grayscale, denoise, threshold, deskew, scale

    Returns:
        Preprocessed image as numpy array
    """
    # ── Step 1: Upscale ──
    scale = cfg.get("scale", 2.0)
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LANCZOS4)

    # ── Step 2: Grayscale conversion ──
    if cfg.get("grayscale", True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 3: Noise removal ──
    if cfg.get("denoise", True):
        if len(img.shape) == 2:
            # Grayscale: fast Non-Local Means
            img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
        else:
            img = cv2.fastNlMeansDenoisingColored(img, h=10)

    # ── Step 4: Adaptive thresholding ──
    if cfg.get("threshold", True):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

    # ── Step 5: Deskew ──
    if cfg.get("deskew", False):
        img = deskew_image(img)

    return img


def deskew_image(img: np.ndarray) -> np.ndarray:
    """
    Detect and correct the skew/rotation of a document image.

    Uses Hough Line Transform to find the dominant angle,
    then rotates the image to straighten it.
    """
    # Work on a grayscale copy for detection
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return img  # No lines found; return as-is

    # ── Compute median angle ──
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))

    # ── Rotate image to correct skew ──
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


# ─────────────────────────────────────────────
# OCR EXTRACTION — IMAGE
# ─────────────────────────────────────────────
def extract_text_from_image(
    img: np.ndarray,
    reader,
) -> Tuple[str, float]:
    """
    Run EasyOCR on a preprocessed image.

    Args:
        img: Preprocessed numpy image
        reader: EasyOCR Reader instance

    Returns:
        (extracted_text, average_confidence)
    """
    # EasyOCR returns a list of (bbox, text, confidence)
    results = reader.readtext(img, detail=1, paragraph=False)

    if not results:
        return "", 0.0

    # ── Assemble text preserving rough line order ──
    lines   = [item[1] for item in results]
    confs   = [item[2] for item in results]
    text    = "\n".join(lines)
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return text, avg_conf


# ─────────────────────────────────────────────
# OCR EXTRACTION — PDF
# ─────────────────────────────────────────────
def extract_text_from_pdf(
    pdf_bytes: bytes,
    reader,
    preprocessing_cfg: dict,
) -> Tuple[str, float]:
    """
    Convert each PDF page to an image and run OCR on it.

    Requires: pdf2image  +  poppler (Windows: poppler in PATH)
    Falls back gracefully if pdf2image is not installed.

    Args:
        pdf_bytes: Raw PDF file bytes
        reader: EasyOCR Reader instance
        preprocessing_cfg: Preprocessing config dict

    Returns:
        (full_text, average_confidence)
    """
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        return (
            "[ERROR] pdf2image is not installed.\n"
            "Run:  pip install pdf2image\n"
            "Also install Poppler for Windows from:\n"
            "https://github.com/oschwartz10612/poppler-windows/releases",
            0.0,
        )

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as e:
        return f"[ERROR] Could not read PDF: {e}", 0.0

    all_text = []
    all_conf = []

    for i, page in enumerate(pages):
        # ── Convert PIL → OpenCV ──
        cv_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        processed = preprocess_image(cv_img, preprocessing_cfg)
        text, conf = extract_text_from_image(processed, reader)

        if text.strip():
            all_text.append(f"── Page {i + 1} ──\n{text}")
            all_conf.append(conf)

    full_text = "\n\n".join(all_text)
    avg_conf  = float(np.mean(all_conf)) if all_conf else 0.0
    return full_text, avg_conf


# ─────────────────────────────────────────────
# CONFIDENCE CALCULATION (helper)
# ─────────────────────────────────────────────
def calculate_confidence(results: list) -> float:
    """
    Average confidence from EasyOCR result list.

    Args:
        results: List of (bbox, text, confidence) tuples

    Returns:
        Average confidence as a float [0.0, 1.0]
    """
    if not results:
        return 0.0
    confs = [r[2] for r in results]
    return float(np.mean(confs))


# ─────────────────────────────────────────────
# EXPORT — TXT
# ─────────────────────────────────────────────
def save_text_as_txt(text: str) -> bytes:
    """
    Encode extracted text as UTF-8 bytes for download.

    Args:
        text: The extracted text string

    Returns:
        UTF-8 encoded bytes
    """
    return text.encode("utf-8")


# ─────────────────────────────────────────────
# EXPORT — PDF
# ─────────────────────────────────────────────
def save_text_as_pdf(text: str) -> Optional[bytes]:
    """
    Convert extracted text to a downloadable PDF file.

    Requires: fpdf2  (pip install fpdf2)

    Args:
        text: The extracted text string

    Returns:
        PDF bytes, or None if fpdf2 is unavailable
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Title ──
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Extracted Text", ln=True, align="C")
    pdf.ln(5)

    # ── Body ──
    pdf.set_font("Helvetica", size=11)
    for line in text.split("\n"):
        # Handle long lines gracefully
        pdf.multi_cell(0, 7, line)

    return pdf.output(dest="S").encode("latin-1")
