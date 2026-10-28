"""Minimal web app for a live test: dictate a note with the microphone and see the care record.

    pip install -e ".[stt,ui]"
    streamlit run app.py

Everything runs locally: Whisper for speech-to-text, Ollama for the extraction.
"""

import tempfile
from pathlib import Path

import streamlit as st

from care_record.llm import LiteLLM
from care_record.pipeline import process_file
from care_record.schema import SECTIONS

# Colour of the status box: green, yellow or red
STATUS_BOX = {"ok": st.success, "needs_review": st.warning, "failed": st.error}

st.set_page_config(page_title="Voice-to-Care-Record", page_icon="🩺")
st.title("Voice-to-Care-Record")
st.caption("Demo with synthetic notes only. Do not dictate real patient data.")

audio = st.audio_input("Dictate a care note (German)")
if audio is None:
    st.stop()  # nothing recorded yet

# 1. process_file() works on files, so store the recording as a temporary WAV file
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    f.write(audio.getvalue())
try:
    with st.spinner("Transcribing and extracting ..."):
        record = process_file(Path(f.name), LiteLLM())
finally:
    Path(f.name).unlink()

# 2. Overall status and what Whisper heard
STATUS_BOX[record.status](f"Status: {record.status} ({len(record.review_flags)} review flag(s))")
st.subheader("Transcript")
st.write(record.transcript)

# 3. Review flags first: these are the items a human must check
for flag in record.review_flags:
    st.warning(f"**{flag.path}** · {flag.reason}: {flag.detail}")

# 4. One table per section; the "review" column says why an item is flagged
for section in SECTIONS:
    items = getattr(record.note, section)
    if not items:
        continue
    st.subheader(section.replace("_", " ").capitalize())
    rows = []
    for i, item in enumerate(items):
        row = item.model_dump(mode="json", exclude_none=True)
        row["review"] = ", ".join(f.reason for f in record.flags_for(f"{section}[{i}]"))
        rows.append(row)
    st.dataframe(rows, hide_index=True)

st.subheader("Summary")
st.write(record.note.summary)

# 5. Timings and the raw JSON, for debugging
meta = record.meta
st.caption(f"Speech-to-text {meta['stt_latency_s']:.1f}s · extraction {meta['latency_s']:.1f}s"
           f" · {meta['attempts']} attempt(s) · {meta['model']}")
with st.expander("Raw JSON"):
    st.json(record.model_dump(mode="json"))
