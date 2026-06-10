from __future__ import annotations

import base64
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .cleanup import clean_ocr_lines, join_display_lines, is_duplicate
from .ocr import OCRRunner
from .preprocessing import preprocess_for_ocr
from .state import AssistantState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LectureAssistant:
    """
    Background worker that runs:
    camera frame -> preprocessing -> OCR -> cleanup/dedup -> state update
    """

    def __init__(
        self,
        camera_index: int = 0,
        capture_interval: float = 1.2,
        recent_window_size: int = 20,
        history_limit: int = 25,
        gpu: bool = False,
    ):
        self.camera_index = camera_index
        self.capture_interval = capture_interval
        self.recent_window_size = recent_window_size
        self.history_limit = history_limit

        self.ocr = OCRRunner(languages=["en"], gpu=gpu)
        self.state = AssistantState()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._recent_lines = deque(maxlen=recent_window_size)
        self._display_lines = deque(maxlen=20)

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                self.state.running = True
                self.state.status = "Already running"
                return

            self._stop_event.clear()
            self.state.running = True
            self.state.latest_error = ""
            self.state.status = "Starting camera worker..."

            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self.state.running = False
            self.state.camera_active = False
            self.state.ocr_running = False
            self.state.status = "Stopped"

    def set_audio_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.state.audio_enabled = bool(enabled)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            data = self.state.to_dict()
            data["display_text"] = join_display_lines(self._display_lines)
            return data

    def _update_state_after_frame(
        self,
        raw_lines: List[str],
        cleaned_lines: List[str],
        new_lines: List[str],
        source: str,
    ) -> None:
        with self._lock:
            self.state.camera_active = True
            self.state.ocr_running = True
            self.state.latest_raw_text = join_display_lines(raw_lines)
            self.state.latest_text = join_display_lines(self._display_lines)

            if new_lines:
                chunk = " ".join(new_lines).strip()
                self.state.last_new_chunk = chunk
                self.state.new_text_id += 1
                self.state.updated_at = utc_now_iso()
                self.state.status = f"New text detected from {source}"

                self.state.history.append(
                    {
                        "timestamp": self.state.updated_at,
                        "text": chunk,
                        "source": source,
                    }
                )
                if len(self.state.history) > self.history_limit:
                    self.state.history = self.state.history[-self.history_limit :]
            else:
                self.state.updated_at = utc_now_iso()
                if self._display_lines:
                    self.state.status = f"Running: {source} frames are being read"
                else:
                    self.state.status = f"Running: waiting for readable text from {source}"

            self.state.latest_error = ""

    def process_frame(self, frame: np.ndarray, source: str = "camera") -> Dict[str, Any]:
        """
        Process a single frame and update assistant state.

        This method is used by both the live camera worker and the optional
        upload fallback route.
        """
        if frame is None:
            raise ValueError("frame is None")

        processed = preprocess_for_ocr(frame)
        raw_lines = self.ocr.extract_text(processed)
        cleaned_lines = clean_ocr_lines(raw_lines)

        new_lines: List[str] = []

        with self._lock:
            for line in cleaned_lines:
                if not line:
                    continue

                if is_duplicate(line, list(self._recent_lines)):
                    continue

                self._recent_lines.append(line)
                self._display_lines.append(line)
                new_lines.append(line)

        self._update_state_after_frame(raw_lines, cleaned_lines, new_lines, source=source)

        return {
            "ok": True,
            "raw_lines": raw_lines,
            "cleaned_lines": cleaned_lines,
            "new_lines": new_lines,
            "display_text": join_display_lines(self._display_lines),
            "new_text_id": self.state.new_text_id,
            "last_new_chunk": self.state.last_new_chunk,
        }

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return None

        # Keep the frames reasonably detailed without making OCR too slow.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _worker_loop(self) -> None:
        cap: Optional[cv2.VideoCapture] = None

        try:
            while not self._stop_event.is_set():
                if cap is None:
                    cap = self._open_camera()
                    if cap is None:
                        with self._lock:
                            self.state.camera_active = False
                            self.state.ocr_running = False
                            self.state.latest_error = (
                                f"Could not open camera index {self.camera_index}. "
                                "Check permissions or try a different index."
                            )
                            self.state.status = "Camera unavailable"
                        time.sleep(2.0)
                        continue

                    with self._lock:
                        self.state.camera_active = True
                        self.state.ocr_running = True
                        self.state.latest_error = ""
                        self.state.status = "Camera connected"

                ok, frame = cap.read()
                if not ok or frame is None:
                    with self._lock:
                        self.state.latest_error = "Camera frame read failed. Reconnecting..."
                        self.state.status = "Reconnecting camera..."
                        self.state.camera_active = False
                        self.state.ocr_running = False
                    cap.release()
                    cap = None
                    time.sleep(1.0)
                    continue

                try:
                    self.process_frame(frame, source="camera")
                except Exception as exc:
                    with self._lock:
                        self.state.latest_error = f"OCR error: {exc}"
                        self.state.status = "OCR error"
                        self.state.ocr_running = False

                time.sleep(self.capture_interval)

        finally:
            if cap is not None:
                cap.release()
            with self._lock:
                self.state.running = False
                self.state.camera_active = False
                self.state.ocr_running = False
                if not self.state.latest_error:
                    self.state.status = "Stopped"

    @staticmethod
    def decode_upload_image(file_bytes: bytes) -> Optional[np.ndarray]:
        if not file_bytes:
            return None
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def decode_base64_image(data_url_or_base64: str) -> Optional[np.ndarray]:
        if not data_url_or_base64:
            return None

        payload = data_url_or_base64
        if "," in payload and payload.strip().lower().startswith("data:"):
            payload = payload.split(",", 1)[1]

        try:
            raw = base64.b64decode(payload)
        except Exception:
            return None

        return LectureAssistant.decode_upload_image(raw)
