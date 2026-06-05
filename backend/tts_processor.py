# Text-to-speech output

import pyttsx3
import tempfile
import os

def text_to_speech(text: str, output_path: str = "output.mp3"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Slower for accessibility
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path