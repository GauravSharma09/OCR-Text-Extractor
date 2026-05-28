
# OCR Text Extractor

A simple and efficient OCR (Optical Character Recognition) application built with Python that extracts text from images and scanned documents. This project provides an easy-to-use interface for uploading files and retrieving editable text output.

## Features

- Extract text from images using OCR
- Supports JPG, PNG, and JPEG formats
- Clean and user-friendly interface
- Fast text processing
- Copy extracted text easily
- Works with scanned documents and screenshots
- Error handling for unsupported files
- Real-time extracted text display

## Tech Stack

- Python
- Streamlit
- OpenCV
- Pillow (PIL)
- Tesseract OCR
- NumPy

## Project Structure

```bash
ocr_text_extractor/
│── app.py
│── requirements.txt
│── README.md
│── assets/
│── uploads/
│── extracted_text/
```

## Installation

### Clone Repository
```bash
git clone https://github.com/yourusername/ocr_text_extractor.git
cd ocr_text_extractor
```

### Create Virtual Environment
```bash
python -m venv venv
```

### Activate Virtual Environment (Windows)
```bash
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

## How It Works

1. Upload an image or scanned document  
2. Preprocess image for better OCR accuracy  
3. Extract text using OCR engine  
4. Display extracted text in real-time  
5. Copy or save the extracted text  

## Future Improvements

- PDF text extraction
- Multi-language OCR
- Handwriting recognition
- Export text to TXT/PDF
- Batch image processing
- Better UI enhancements

## Use Cases

- Extract text from screenshots
- Digitize printed documents
- Scan notes
- Convert image text into editable format
- OCR for academic/work purposes

## Requirements

- Python 3.9+
- Streamlit
- OpenCV
- Pillow
- NumPy
- Tesseract OCR

## Author

Prachi Golia | Gaurav Sharma

## License

This project is licensed under the MIT License.
