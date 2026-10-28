"""Local speech-to-text with faster-whisper. Runs fully offline once the model is downloaded."""

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Whisper model size; "large-v3-turbo" is accurate for German and still fast on a GPU
WHISPER_MODEL = os.getenv("CARE_WHISPER_MODEL", "large-v3-turbo")


@dataclass
class Transcription:
    """Result of speech-to-text. Fields: text (the transcript), latency_s (duration in seconds)."""

    text: str
    latency_s: float


@lru_cache(maxsize=1)
def _load(model_size: str):
    """Load the Whisper model once and reuse it (lru_cache keeps it in memory).

    Input:  model_size - Whisper model name, e.g. "large-v3-turbo"
    Output: a faster_whisper.WhisperModel, on the GPU if possible, otherwise on the CPU
    """
    import numpy as np
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        # Loading also works without the CUDA libraries (cuBLAS, cuDNN); only inference fails.
        # So transcribe one second of silence to find out whether the GPU really works.
        list(model.transcribe(np.zeros(16000, dtype=np.float32), language="de")[0])
        return model
    except Exception:  # no usable CUDA -> CPU
        return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(path: Path, model_size: str = WHISPER_MODEL) -> Transcription:
    """Transcribe a German audio file.

    Input:  path       - audio file (.wav, .mp3, ...)
            model_size - Whisper model name
    Output: Transcription with the full text and the time it took
    """
    model = _load(model_size)
    start = time.perf_counter()
    # vad_filter skips silence; beam_size 5 trades a little speed for accuracy
    segments, _info = model.transcribe(str(path), language="de", vad_filter=True, beam_size=5)
    # segments is lazy: the actual transcription happens while joining, so this is inside the timer
    text = " ".join(s.text.strip() for s in segments).strip()
    return Transcription(text=text, latency_s=time.perf_counter() - start)
