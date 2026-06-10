from __future__ import annotations

import cv2
import numpy as np


def preprocess_for_ocr(frame: np.ndarray) -> np.ndarray:
    """
    Prepare a camera frame for OCR.

    The preprocessing is intentionally lightweight:
    - enlarge smaller frames
    - grayscale
    - denoise
    - contrast enhancement
    - adaptive thresholding
    - a small blur to remove speckle
    """
    if frame is None:
        raise ValueError("frame is None")

    image = frame.copy()
    height, width = image.shape[:2]

    # Make small frames easier for OCR to read.
    if max(height, width) < 1200:
        scale = 1.5
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Suppress camera noise while keeping text edges.
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Improve contrast for both chalkboard and whiteboard scenes.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Sharpen slightly to help text edges.
    sharpen_kernel = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float32,
    )
    sharpened = cv2.filter2D(gray, ddepth=-1, kernel=sharpen_kernel)

    mean_brightness = float(np.mean(sharpened))
    threshold_type = cv2.THRESH_BINARY_INV if mean_brightness < 110 else cv2.THRESH_BINARY

    processed = cv2.adaptiveThreshold(
        sharpened,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=threshold_type,
        blockSize=31,
        C=11,
    )

    processed = cv2.medianBlur(processed, 3)
    return processed
