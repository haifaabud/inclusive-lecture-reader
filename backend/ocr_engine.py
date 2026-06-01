# OCR logic (EasyOCR/PaddleOCR)

import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image_path: str) -> str:
    results = reader.readtext(image_path, detail=0)
    return " ".join(results)