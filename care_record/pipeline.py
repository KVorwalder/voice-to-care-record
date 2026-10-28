"""End-to-end: audio or text -> validated CareRecord."""

from pathlib import Path

from .extraction import extract
from .llm import LLM
from .schema import CareRecord, ExtractedNote, ReviewFlag
from .validation import review

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def process_transcript(transcript: str, llm: LLM) -> CareRecord:
    result = extract(transcript, llm)
    meta = {
        "model": llm.model,
        "attempts": result.attempts,
        "valid_first_try": result.valid_first_try,
        "valid_after_retry": result.valid,
        "errors": result.errors,
        "latency_s": round(result.latency_s, 3),
    }

    if result.note is None:
        return CareRecord(
            transcript=transcript,
            note=ExtractedNote(),
            review_flags=[ReviewFlag(path="note", reason="extraction_failed",
                                     detail="no valid JSON after retry; document manually")],
            status="failed",
            meta=meta,
        )

    flags = [ReviewFlag(path=path, reason="invalid", detail=f"item dropped: {error}")
             for path, error in result.dropped]
    flags += review(result.note, transcript)
    return CareRecord(
        transcript=transcript,
        note=result.note,
        review_flags=flags,
        status="needs_review" if flags else "ok",
        meta=meta,
    )


def process_file(path: Path, llm: LLM) -> CareRecord:
    path = Path(path)
    if path.suffix.lower() in AUDIO_SUFFIXES:
        from .stt import transcribe

        stt = transcribe(path)
        record = process_transcript(stt.text, llm)
        record.meta["stt_latency_s"] = round(stt.latency_s, 3)
        return record
    return process_transcript(path.read_text(encoding="utf-8").strip(), llm)
