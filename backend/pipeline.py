# End-to-end orchestrator

from image_preprocessor import preprocess_for_ocr
from ocr_engine import extract_text
from ai_processor import explain_text
from tts_engine import text_to_speech
import cv2
import tempfile

def run_pipeline(image_path: str) -> dict:
    # 1. Preprocess
    processed = preprocess_for_ocr(image_path)
    temp_path = tempfile.mktemp(suffix=".png")
    cv2.imwrite(temp_path, processed)
    
    # 2. OCR
    raw_text = extract_text(temp_path)
    
    # 3. AI Explanation
    explained = explain_text(raw_text)
    
    # 4. TTS
    audio_path = text_to_speech(explained)
    
    return {
        "raw_text": raw_text,
        "explained_text": explained,
        "audio_path": audio_path
    }