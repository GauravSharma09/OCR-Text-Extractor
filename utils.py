"""
utils.py — OCR Text Extractor Utilities
=========================================
All helper functions: preprocessing, OCR, export, validation.
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
    """Returns display name → {code, flag} for EasyOCR language codes."""
    return {
        "English":              {"code": "en",     "flag": "🇬🇧"},
        "Hindi":                {"code": "hi",     "flag": "🇮🇳"},
        "French":               {"code": "fr",     "flag": "🇫🇷"},
        "German":               {"code": "de",     "flag": "🇩🇪"},
        "Spanish":              {"code": "es",     "flag": "🇪🇸"},
        "Italian":              {"code": "it",     "flag": "🇮🇹"},
        "Portuguese":           {"code": "pt",     "flag": "🇵🇹"},
        "Chinese (Simplified)": {"code": "ch_sim", "flag": "🇨🇳"},
        "Japanese":             {"code": "ja",     "flag": "🇯🇵"},
        "Korean":               {"code": "ko",     "flag": "🇰🇷"},
        "Arabic":               {"code": "ar",     "flag": "🇸🇦"},
        "Russian":              {"code": "ru",     "flag": "🇷🇺"},
    }


# ─────────────────────────────────────────────
# FILE VALIDATION
# ─────────────────────────────────────────────
ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/tiff", "image/webp", "application/pdf",
}
MAX_SIZE_MB = 25


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate a Streamlit uploaded file object.
    Returns (True, "") on success or (False, error_message) on failure.
    """
    if uploaded_file is None:
        return False, "No file uploaded."
    if uploaded_file.type not in ALLOWED_TYPES:
        return False, (
            f"Unsupported file type '{uploaded_file.type}'. "
            "Please upload JPG, PNG, BMP, TIFF, WEBP, or PDF."
        )
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        return False, (
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum allowed: {MAX_SIZE_MB} MB."
        )
    return True, ""


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def _looks_like_screenshot(img: np.ndarray) -> bool:
    """
    Heuristic check: is this image a UI screenshot rather than a scanned doc?

    Screenshots tend to be:
      • Wide-and-short (aspect ratio > 2.5), OR
      • Small overall (< 400 px tall)

    For screenshots we skip adaptive thresholding because it destroys
    coloured UI elements (blue links, icons, badges).
    """
    h, w = img.shape[:2]
    aspect = w / max(h, 1)
    return aspect > 2.5 or h < 400


def preprocess_image(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Preprocessing pipeline — all steps are optional via cfg flags.

    Step 1 : Upscale          — enlarges small text for better detection
    Step 2 : Grayscale        — reduces colour noise
    Step 3 : CLAHE            — adaptive contrast boost (always with grayscale)
    Step 4 : Denoise          — removes JPEG grain (gentle h=7)
    Step 5 : Adaptive thresh  — binarises the image (skipped for screenshots)
    Step 6 : Deskew           — corrects rotated scans

    Args:
        img : OpenCV BGR numpy array
        cfg : dict with keys: scale, grayscale, denoise, threshold, deskew
    """
    is_screenshot = _looks_like_screenshot(img)

    # ── Step 1: Upscale ──────────────────────────────────────────────────
    scale = cfg.get("scale", 2.0)
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )

    # ── Step 2: Grayscale ────────────────────────────────────────────────
    if cfg.get("grayscale", True):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Step 3: CLAHE (adaptive contrast) ───────────────────────────
        # Boosts local contrast so faint text becomes clearly visible
        # before denoising/thresholding.  Always applied with grayscale.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    # ── Step 4: Denoise ──────────────────────────────────────────────────
    # h=7 (was h=10) — gentler.  h=10 blurs thin strokes and drops confidence.
    if cfg.get("denoise", True):
        if len(img.shape) == 2:
            img = cv2.fastNlMeansDenoising(
                img, h=7, templateWindowSize=7, searchWindowSize=21
            )
        else:
            img = cv2.fastNlMeansDenoisingColored(img, h=7)

    # ── Step 5: Adaptive Threshold ───────────────────────────────────────
    # Skipped for screenshots: adaptive threshold inverts coloured UI text
    # (blue links, orange icons) and makes OCR output garbage.
    # blockSize=15, C=4 (was 11/2) — wider block suits printed documents.
    if cfg.get("threshold", True) and not is_screenshot:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=4,
        )

    # ── Step 6: Deskew ───────────────────────────────────────────────────
    if cfg.get("deskew", False):
        img = _deskew(img)

    return img


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Detect and correct image skew using Hough Line Transform.
    Works on both grayscale and colour images.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return img

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180.0 / np.pi) - 90.0
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ─────────────────────────────────────────────
# OCR — IMAGE
# ─────────────────────────────────────────────
def extract_text_from_image(
    img: np.ndarray,
    reader,
) -> Tuple[str, float]:
    """
    Run EasyOCR on a preprocessed image and return (text, avg_confidence).

    Key improvements:
    • width_ths=0.5  — prevents adjacent words being merged into one long
                        unreadable token (was 0.9 which was far too aggressive).
    • contrast_ths / adjust_contrast — catches low-contrast text regions.
    • Results sorted top→bottom, left→right for correct reading order,
      which matters for multi-column layouts like resumes.
    """
    results = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        width_ths=0.5,        # merge only truly adjacent boxes (was 0.9 → word merging bug)
        ycenter_ths=0.5,      # vertical grouping tolerance
        height_ths=0.7,       # merge boxes of similar height
        slope_ths=0.1,        # ignore heavily slanted detections
        contrast_ths=0.1,     # detect low-contrast text
        adjust_contrast=0.5,  # internal EasyOCR contrast boost
    )

    if not results:
        return "", 0.0

    # Sort top-to-bottom, then left-to-right, using the top-left bbox corner
    results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))

    lines    = [item[1] for item in results]
    confs    = [item[2] for item in results]
    text     = "\n".join(lines)
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return text, avg_conf


# ─────────────────────────────────────────────
# OCR — PDF
# ─────────────────────────────────────────────
def extract_text_from_pdf(
    pdf_bytes: bytes,
    reader,
    preprocessing_cfg: dict,
) -> Tuple[str, float]:
    """
    Convert each PDF page to an image and run OCR on it.
    Requires: pdf2image + poppler.
    """
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        return (
            "[ERROR] pdf2image is not installed.\n"
            "Run: pip install pdf2image\n"
            "Also install Poppler for Windows:\n"
            "https://github.com/oschwartz10612/poppler-windows/releases",
            0.0,
        )

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as exc:
        return f"[ERROR] Could not read PDF: {exc}", 0.0

    all_text: list[str] = []
    all_conf: list[float] = []

    for i, page in enumerate(pages):
        cv_img    = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        processed = preprocess_image(cv_img, preprocessing_cfg)
        text, conf = extract_text_from_image(processed, reader)
        if text.strip():
            all_text.append(f"── Page {i + 1} ──\n{text}")
            all_conf.append(conf)

    return "\n\n".join(all_text), float(np.mean(all_conf)) if all_conf else 0.0


# ─────────────────────────────────────────────
# CONFIDENCE HELPER
# ─────────────────────────────────────────────
def calculate_confidence(results: list) -> float:
    """Average confidence from an EasyOCR result list."""
    if not results:
        return 0.0
    return float(np.mean([r[2] for r in results]))


# ─────────────────────────────────────────────
# EXPORT — TXT
# ─────────────────────────────────────────────
def save_text_as_txt(text: str) -> bytes:
    """Return extracted text as UTF-8 bytes ready for st.download_button."""
    return text.encode("utf-8")


# ─────────────────────────────────────────────
# EXPORT — PDF
# ─────────────────────────────────────────────
def _sanitize_for_pdf(line: str, max_token: int = 60) -> str:
    """
    Make a single line safe for fpdf2's default latin-1 Helvetica font.

    Two operations:
    1. latin-1 encode — replaces every unencodable unicode char with '?'.
       This is the root cause of:
         FPDFException: Not enough horizontal space to render a single character
       Any char outside latin-1 makes fpdf2's line-break algorithm fail.

    2. Long-token break — inserts spaces every max_token characters inside
       any unbroken token (e.g. a URL or an OCR-merged word like
       "theassignmentswwhichihavecompletedimy").  A single token wider than
       the text cell also triggers the same FPDFException.
    """
    line = line.replace("\t", "    ")
    # Step 1: latin-1 safety
    line = line.encode("latin-1", errors="replace").decode("latin-1")
    # Step 2: break monster tokens
    words, result = line.split(" "), []
    for word in words:
        chunks = [word[i : i + max_token] for i in range(0, len(word), max_token)]
        result.extend(chunks)
    return " ".join(result)


def save_text_as_pdf(text: str) -> Optional[bytes]:
    """
    Convert extracted text to a downloadable PDF.

    Fixes applied:
    • _sanitize_for_pdf() handles latin-1 encoding AND overlong tokens.
    • Explicit page_width = pdf.w - margins  (not 0 which is ambiguous
      in fpdf2 and can resolve to 0 usable pixels in some versions).
    • Per-line try/except so one bad line never crashes the whole export;
      falls back to ASCII-only, then silently skips if still failing.
    • Empty lines → small vertical gap instead of blank multi_cell.

    Requires: fpdf2  (pip install fpdf2)
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    L, R, T, B = 15, 15, 15, 15          # margins in mm

    pdf = FPDF()
    pdf.set_margins(left=L, top=T, right=R)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=B)

    # Calculate actual usable width (avoids the width=0 ambiguity bug)
    usable_w = pdf.w - L - R

    # ── Title ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(usable_w, 10, "Extracted Text", ln=True, align="C")
    pdf.ln(5)

    # ── Body ─────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", size=11)
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            pdf.ln(4)        # blank line → small vertical gap
            continue

        safe = _sanitize_for_pdf(raw_line)
        try:
            pdf.multi_cell(usable_w, 7, safe)
        except Exception:
            # Fallback 1: strip to ASCII only
            try:
                pdf.multi_cell(
                    usable_w, 7,
                    safe.encode("ascii", errors="replace").decode("ascii"),
                )
            except Exception:
                pass          # Fallback 2: skip this line entirely

    # fpdf2 v2+ returns bytearray from output(), older versions returned a
    # latin-1 string.  Handle both so the code works on any installed version.
    raw = pdf.output(dest="S")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return raw.encode("latin-1")