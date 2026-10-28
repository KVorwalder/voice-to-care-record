# Voice-to-Care-Record Assistant

Caregivers in elderly care dictate a short voice note after a shift. This project turns that
note into a **validated, structured care record (JSON)** and **flags anything uncertain for
human review** instead of guessing.

```
Audio ─► [1 STT: faster-whisper, local] ─► Transcript ─► [2 LLM extraction] ─► JSON
                                                                                │
             CareRecord ◄── [3 validation: schema · retry · grounding · plausibility]
                                                                                │
                                              [4 evaluation harness vs. synthetic ground truth]
```

## What it does

| Step | Implementation | File |
|---|---|---|
| Speech-to-text | `faster-whisper`, German, runs offline | `care_record/stt.py` |
| Extraction | One LLM call via `litellm`; same code for API and local (Ollama) models | `care_record/llm.py`, `extraction.py` |
| Schema | Pydantic models; observations follow the 6 SIS topic areas, plus vital signs, medications (given/refused), incidents (fall/pain/wound/other), summary | `care_record/schema.py` |
| Validation & fallback | Invalid JSON → retry once with the validation error fed back → if still invalid, keep the items that validate and flag the rest | `care_record/extraction.py` |
| Review flags | `uncertain` (model hedged), `ungrounded` (quote not in transcript, or number not in its own quote), `implausible` (out of physiological range), `invalid`, `extraction_failed` | `care_record/validation.py` |
| Synthetic data | Ground truth first, then spoken-style German notes rendered from it (45 cases, 9 categories) | `care_record/synth.py` |
| Evaluation | Field-level P/R/F1, hallucination rate, wrong values, review recall, schema validity, latency, cost | `care_record/metrics.py`, `evaluate.py` |

Every extracted item carries an `evidence` field with the exact transcript quote it is based on.

### SIS topic areas
The observation areas follow the six *Themenfelder* of the SIS (Strukturierte
Informationssammlung, Strukturmodell): 1 kognitive und kommunikative Fähigkeiten,
2 Mobilität und Beweglichkeit, 3 krankheitsbezogene Anforderungen und Belastungen,
4 Selbstversorgung, 5 Leben in sozialen Beziehungen, 6 Wohnen/Häuslichkeit (stationär)
bzw. Haushaltsführung (ambulant).

### Why no LLM framework
The pipeline is a single extraction call plus validation and one retry. Plain Python keeps
every step visible and debuggable; a framework like LangChain would add abstraction without
removing any code we actually need.

## How to run

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"           # add ,stt,ui for Whisper and Streamlit
ollama pull gemma3:12b            # free local default model (needs Ollama)
```

```bash
care generate                                         # regenerate data/synthetic/cases.jsonl
care process note.txt                                 # text note -> care record
care process note.wav                                 # audio: local Whisper first
care evaluate data/synthetic/cases.jsonl              # writes results/runs/<run>/ and results/summary.csv
care evaluate data/synthetic/cases_test.jsonl --limit 10
pytest
streamlit run app.py                                  # live test: dictate a note with the microphone
```

Whisper runs on the GPU if the CUDA libraries (cuBLAS, cuDNN) are installed, otherwise on the CPU
(about 4 s for a 25 s note with `large-v3-turbo`).

Models are plain litellm strings, so an API model works too: `--model anthropic/claude-haiku-5-5`
(needs the provider key, e.g. `ANTHROPIC_API_KEY`). Defaults can be set with `CARE_MODEL` and
`CARE_WHISPER_MODEL`.

## Synthetic data

No real patient data is used anywhere. `care_record/synth.py` samples a ground-truth record,
then renders each fact as a spoken German phrase (filler words, "RR 130 zu 80",
"hundertdreißig", "Panto 40", …). Each phrase is inserted verbatim, so it is also the
ground-truth evidence. Generation is seeded and reproducible.

| Category | n | What it tests |
|---|---|---|
| standard | 16 | normal notes, mixed digits/words |
| nothing_noteworthy | 4 | empty record must stay empty |
| missing_value | 4 | "Blutdruck gemessen, Wert trag ich nach" → must **not** produce a value |
| self_correction | 4 | "Puls 72, nee, Quatsch, 76" → 76 |
| spoken_numbers | 4 | all numbers as German words |
| refused_meds | 4 | medication status `refused` |
| hedged | 3 | "so um die 38, glaub ich" → extract + flag |
| contradiction | 3 | conflicting statements → extract + flag |
| implausible | 3 | "Temperatur drei komma acht" → extract + flag |

## Evaluation method

Items are matched per section by entity (vital kind, drug name, incident type, SIS area):

- **correct** – entity and value match
- **wrong value** – entity found, value wrong (e.g. pulse 76 instead of 72)
- **hallucination** – predicted entity that is not in the note
- **missed** – truth entity not predicted

Hallucinations are reported separately and weighted highest
(`safety_error_score = (3·hallucinated + 2·wrong + 1·missed) / truth items`), because in care
documentation an invented entry such as "Ramipril given" is worse than a missing one.
For items the ground truth marks as needing review (hedged, contradictory, implausible), the
system only has to find the entity and flag it; `review_recall` measures how often it does.
Observations (free text in an SIS area) are scored separately and kept out of the
hallucination rate, since the area assignment is partly a judgement call.

## Results

_To be filled after the first model runs (API vs. local, text vs. audio)._

## Limitations

- Template-generated notes are cleaner and less varied than real dictation.
- Observation scoring only checks the SIS area, not the content of the text.
- Plausibility ranges are generic adult ranges, not per-resident thresholds.

## Privacy

- STT runs locally; with a local model the whole pipeline is offline.
- With an API model the transcript leaves the machine; real deployment would need a DPA,
  EU hosting and pseudonymisation of names before the call.
- The dataset is fully synthetic.

## Development workflow

I built this project together with [Claude Code](https://claude.com/claude-code).

## Debugging examples

_To be filled with 2–3 concrete failure cases from the evaluation runs._
