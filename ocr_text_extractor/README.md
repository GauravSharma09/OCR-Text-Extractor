# 🔍 OCR Text Extractor

> A production-grade OCR application built with **EasyOCR**, **OpenCV**, and **Streamlit**.  
> **No Tesseract required** — works entirely with Python packages.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 OCR Engine | EasyOCR (neural network, no Tesseract) |
| 🖼 Input Formats | JPG, PNG, BMP, TIFF, WEBP, PDF |
| 🌍 Languages | 12+ languages (English, Hindi, French, German…) |
| ⚙ Preprocessing | Grayscale, Denoise, Threshold, Deskew, Upscale |
| 📊 Confidence Score | Per-extraction accuracy indicator |
| 💾 Export | Copy text, Save as TXT, Export as PDF |
| 🎨 UI | Dark modern SaaS-style Streamlit interface |
| ⚡ Performance | GPU acceleration supported (optional) |

---

## 📁 Project Structure

```
ocr_text_extractor/
├── app.py            ← Streamlit UI — main entry point
├── utils.py          ← All helper functions (OCR, preprocessing, export)
├── requirements.txt  ← Python dependencies
├── README.md         ← This file
└── assets/           ← Put sample images here for testing
```

---

## 🚀 Installation & Setup (Windows)

### Step 1 — Clone / Download the project

```bash
git clone https://github.com/your-repo/ocr_text_extractor
cd ocr_text_extractor
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install PyTorch (CPU version — smallest download)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> ℹ️ For GPU (NVIDIA CUDA 11.8):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 4 — Install all other dependencies

```bash
pip install -r requirements.txt
```

> ⏳ EasyOCR will download language model files (~100 MB) on first run. This is normal.

### Step 5 — (Optional) Install Poppler for PDF support

PDF input requires Poppler:

1. Download the latest Windows build from:  
   👉 https://github.com/oschwartz10612/poppler-windows/releases
2. Extract the ZIP (e.g. to `C:\poppler\`)
3. Add `C:\poppler\Library\bin` to your Windows **PATH** environment variable

---

## ▶ Running the App

### In VS Code

1. Open the `ocr_text_extractor/` folder in VS Code
2. Open the terminal (`Ctrl + ~`)
3. Make sure your venv is active: `venv\Scripts\activate`
4. Run:

```bash
streamlit run app.py
```

5. The app opens automatically at **http://localhost:8501**

### Stop the app

Press `Ctrl + C` in the terminal.

---

## 🔧 How to Use

1. **Upload** a JPG, PNG, or PDF file using the upload box
2. **Configure** preprocessing in the left sidebar (grayscale, denoise, threshold, scale)
3. **Select** OCR language(s) from the sidebar
4. Click **🔍 Extract Text**
5. View the result in the output box with confidence score
6. **Copy**, **Save as TXT**, or **Export as PDF**

---

## ⚙ Preprocessing — What Each Step Does

### Why preprocessing matters

Raw images often have noise, low resolution, or poor contrast. Preprocessing converts them into clean black-and-white images that OCR engines can read much more accurately.

| Step | What it does | When to use |
|---|---|---|
| **Grayscale** | Strips colour information, reduces complexity | Almost always |
| **Upscale** | Enlarges image so small text becomes readable | Small text, thumbnails |
| **Denoise** | Removes JPEG compression artifacts and grain | Scanned photos, screenshots |
| **Adaptive Threshold** | Binarises image (black text on white bg) | Documents, invoices, receipts |
| **Deskew** | Corrects skewed/rotated documents | Scanned pages, phone photos |

### Accuracy tips

- For **printed documents**: enable all preprocessing steps
- For **screenshots**: grayscale + threshold is usually enough
- For **handwriting**: disable threshold (it can break strokes)
- For **small text**: increase Upscale Factor to 3.0–4.0
- For **PDF scans**: enable all steps + set DPI to 200+

---

## ❌ Common Errors & Fixes

### `ModuleNotFoundError: No module named 'easyocr'`
```bash
pip install easyocr
```

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'pdf2image'`
```bash
pip install pdf2image
# Also install Poppler (see Step 5 above)
```

### PDF extraction fails with "Unable to get page count"
- Poppler is not installed or not in PATH
- Download from the link in Step 5 and add to Windows PATH

### EasyOCR downloads models every run
- Models are cached in `~/.EasyOCR/model/` after first download
- Subsequent runs are instant

### App is very slow on first OCR run
- EasyOCR loads a neural network (~100 MB) into memory
- This takes 5–15 seconds on first run per session
- Subsequent extractions in the same session are fast

### `StreamlitAPIException` / page config error
- Make sure `st.set_page_config()` is the very first Streamlit call in `app.py`
- Don't call any Streamlit function before it

### Low confidence / wrong text
- Increase the Upscale Factor in the sidebar
- Enable all preprocessing options
- Make sure the correct language is selected
- For very low-quality images, try a higher-resolution scan

---

## 🌍 Supported Languages

| Language | Code |
|---|---|
| English | en |
| Hindi | hi |
| French | fr |
| German | de |
| Spanish | es |
| Italian | it |
| Portuguese | pt |
| Chinese (Simplified) | ch_sim |
| Japanese | ja |
| Korean | ko |
| Arabic | ar |
| Russian | ru |

Add more by editing `get_supported_languages()` in `utils.py`.  
Full list: https://www.jaided.ai/easyocr/

---

## 🛠 Tech Stack

- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — Deep learning OCR, no Tesseract needed
- **[OpenCV](https://opencv.org/)** — Image preprocessing pipeline
- **[Streamlit](https://streamlit.io/)** — Web UI framework
- **[Pillow](https://python-pillow.org/)** — Image I/O
- **[pdf2image](https://github.com/Belval/pdf2image)** — PDF → image conversion
- **[fpdf2](https://pyfpdf.github.io/fpdf2/)** — PDF export

---

## 📄 License

MIT — free for personal and commercial use.
