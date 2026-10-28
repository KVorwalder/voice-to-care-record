"""Local speech-to-text with faster-whisper. Runs fully offline once the model is downloaded."""

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

WHISPER_MODEL = os.getenv("CARE_WHISPER_MODEL", "large-v3-turbo")


@dataclass
class Transcription:
    text: str
    latency_s: float


@lru_cache(maxsize=1)
def _load(model_size: str):
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception:  # no CUDA available -> CPU
        return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(path: Path, model_size: str = WHISPER_MODEL) -> Transcription:
    model = _load(model_size)
    start = time.perf_counter()
    segments, _info = model.transcribe(str(path), language="de", vad_filter=True, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()  # lazy: consumed inside the timer
    return Transcription(text=text, latency_s=time.perf_counter() - start)
