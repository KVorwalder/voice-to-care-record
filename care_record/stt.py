"""Local speech-to-text with faster-whisper. Runs fully offline once the model is downloaded."""

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_WHISPER_MODEL = os.getenv("CARE_WHISPER_MODEL", "large-v3-turbo")


@dataclass
class Transcription:
    text: str
    latency_s: float
    avg_logprob: float | None  # rough confidence, useful to flag bad audio


@lru_cache(maxsize=2)
def _load(model_size: str):
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception:  # no CUDA available -> CPU
        return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(path: Path, model_size: str = DEFAULT_WHISPER_MODEL) -> Transcription:
    model = _load(model_size)
    start = time.perf_counter()
    segments, _info = model.transcribe(str(path), language="de", vad_filter=True, beam_size=5)
    segments = list(segments)  # transcription is lazy; consume it inside the timer
    latency = time.perf_counter() - start
    text = " ".join(s.text.strip() for s in segments).strip()
    logprob = sum(s.avg_logprob for s in segments) / len(segments) if segments else None
    return Transcription(text=text, latency_s=latency, avg_logprob=logprob)
