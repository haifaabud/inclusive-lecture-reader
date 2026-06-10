from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AssistantState:
    running: bool = False
    camera_active: bool = False
    ocr_running: bool = False
    audio_enabled: bool = True
    status: str = "Idle"
    latest_text: str = ""
    latest_raw_text: str = ""
    last_new_chunk: str = ""
    latest_error: str = ""
    new_text_id: int = 0
    updated_at: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "camera_active": self.camera_active,
            "ocr_running": self.ocr_running,
            "audio_enabled": self.audio_enabled,
            "status": self.status,
            "latest_text": self.latest_text,
            "latest_raw_text": self.latest_raw_text,
            "last_new_chunk": self.last_new_chunk,
            "latest_error": self.latest_error,
            "new_text_id": self.new_text_id,
            "updated_at": self.updated_at,
            "history": list(self.history),
        }
