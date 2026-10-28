"""End-to-end: audio or text -> validated CareRecord."""

from pathlib import Path

from .extraction import extract
from .llm import LLM
from .schema import CareRecord, ExtractedNote, ReviewFlag
from .validation import review

# File endings that are treated as audio and sent through speech-to-text first
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def process_transcript(transcript: str, llm: LLM) -> CareRecord:
    """Turn a transcript into a final care record with review flags.

    Input:  transcript - the dictated note as text
            llm        - the model to use for extraction
    Output: CareRecord with status "ok", "needs_review" or "failed"
    """
    # 1. Extraction: LLM call(s) with validation, retry and salvage
    result = extract(transcript, llm)
    meta = {
        "model": llm.model,
        "attempts": result.attempts,
        "valid_first_try": result.valid_first_try,
        "valid_after_retry": result.valid,
        "errors": result.errors,
        "latency_s": round(result.latency_s, 3),
    }

    # 2a. Nothing usable came back: return an empty record that tells a human to document manually
    if result.note is None:
        return CareRecord(
            transcript=transcript,
            note=ExtractedNote(),
            review_flags=[ReviewFlag(path="note", reason="extraction_failed",
                                     detail="no valid JSON after retry; document manually")],
            status="failed",
            meta=meta,
        )

    # 2b. Items that salvage() had to drop are reported, so nobody assumes they were not said
    flags = [ReviewFlag(path=path, reason="invalid", detail=f"item dropped: {error}")
             for path, error in result.dropped]
    # 3. Validation checks: grounding, number-in-quote, plausibility, uncertainty
    flags += review(result.note, transcript)
    # 4. Any flag means a human has to look at the record
    return CareRecord(
        transcript=transcript,
        note=result.note,
        review_flags=flags,
        status="needs_review" if flags else "ok",
        meta=meta,
    )


def process_file(path: Path, llm: LLM) -> CareRecord:
    """Process a note stored in a file: audio is transcribed first, text is used directly.

    Input:  path - audio file (.wav, .mp3, ...) or UTF-8 text file
            llm  - the model to use for extraction
    Output: CareRecord (for audio, meta also contains the speech-to-text duration)
    """
    path = Path(path)
    if path.suffix.lower() in AUDIO_SUFFIXES:
        # Imported only when needed: faster-whisper is an optional, heavy dependency
        from .stt import transcribe

        stt = transcribe(path)
        record = process_transcript(stt.text, llm)
        record.meta["stt_latency_s"] = round(stt.latency_s, 3)
        return record
    return process_transcript(path.read_text(encoding="utf-8").strip(), llm)
