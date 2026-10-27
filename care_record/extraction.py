"""Transcript -> ExtractedNote with one validation-feedback retry.

Flow:
  1. Ask the LLM for JSON matching the ExtractedNote schema.
  2. Parse + validate with pydantic.
  3. On failure, send the validation error back once and ask for a corrected answer.
  4. If it is still invalid, salvage the items that do validate and report the rest,
     so one bad item does not throw away the whole note.
"""

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from .llm import LLM, LLMResponse
from .schema import SECTIONS, ExtractedNote, Incident, Medication, Observation, VitalSign

ITEM_MODELS = {
    "observations": Observation,
    "vital_signs": VitalSign,
    "medications": Medication,
    "incidents": Incident,
}

SYSTEM_PROMPT = """You turn a caregiver's dictated shift note (German, spoken style, from \
elderly care) into a structured care record.

Return ONLY a JSON object that matches this JSON schema:
{schema}

Observation areas follow the SIS ("Strukturierte Informationssammlung") topic areas:
- kognition_kommunikation: orientation, memory, communication
- mobilitaet_beweglichkeit: walking, transfers, repositioning
- krankheitsbezogene_anforderungen: disease-related needs (oedema, compression stockings, breathing)
- selbstversorgung: eating, drinking, washing, continence
- soziale_beziehungen: visitors, activities, withdrawal
- wohnen_haeuslichkeit: room, living environment, personal wishes about it

Rules:
- Extract only what the note states. Never infer, complete or invent values.
- `evidence` must be copied verbatim from the transcript (exact words, no paraphrase).
- If a measurement is mentioned without a concrete number (e.g. "war normal", "trag ich nach"), \
do NOT create a vital sign for it.
- Write numbers as digits, decimals with a dot (37,5 -> 37.5; "achtunddreißig" -> 38).
- blood_pressure: `value` is systolic, `diastolic` is required. Only blood_pressure has a diastolic.
- If the speaker corrects themselves ("72, nee, 76"), use the corrected value.
- Set `uncertain: true` if the statement is hedged ("so um die", "glaub ich"), contradicts \
itself, or the value seems physiologically impossible. Still extract the item.
- Contradictions: if the note first states one thing and then adds a detail that puts it in \
doubt (a fall is denied but a hint follows that it happened; a drug is reported as given but \
later found untaken), extract the item with `uncertain: true`. A possible fall is a `fall` \
incident. A human must check it; never resolve the contradiction yourself.
- Medication status: "given" if administered/taken, "refused" if refused, spat out or not taken.
- Each statement goes into exactly one section. Do not file the same quote twice.
- `incidents` are: falls (also found on the floor), pain, wounds and skin damage (pressure \
ulcer/Dekubitus, skin tear, reddened skin, bruise), and acute events (`other`, e.g. vomiting, \
restless night). A wound that is being treated is still a `wound` incident. Pain without a \
stated score has pain_score null.
- Ongoing symptoms (shortness of breath, oedema), needing help with transfers or walking, \
eating, washing and social activities are `observations`, not incidents.
- Use canonical drug names where the spoken name is a known brand or short form \
(Novalgin -> Novaminsulfon, Panto -> Pantoprazol, Simva -> Simvastatin).
- `summary`: two or three neutral German sentences. Empty lists are fine if nothing applies."""


def build_messages(transcript: str) -> list[dict]:
    schema = json.dumps(ExtractedNote.model_json_schema(), ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)},
        {"role": "user", "content": f"Transcript:\n{transcript}"},
    ]


def parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating markdown fences and leading/trailing prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data


def salvage(data: dict) -> tuple[ExtractedNote, list[tuple[str, str]]]:
    """Keep every item that validates on its own; return (note, [(path, error)])."""
    kept: dict = {name: [] for name in SECTIONS}
    dropped = []
    for name in SECTIONS:
        items = data.get(name) or []
        if not isinstance(items, list):
            dropped.append((name, "expected a list"))
            continue
        for i, raw in enumerate(items):
            try:
                kept[name].append(ITEM_MODELS[name].model_validate(raw))
            except ValidationError as e:
                dropped.append((f"{name}[{i}]", _short_error(e)))
    summary = data.get("summary") if isinstance(data.get("summary"), str) else ""
    return ExtractedNote(**kept, summary=summary), dropped


def _short_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()
        )
    return str(e)


@dataclass
class ExtractionResult:
    note: ExtractedNote | None
    attempts: int
    valid_first_try: bool
    valid: bool = False  # fully schema-valid on some attempt (no salvage needed)
    errors: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # salvaged-away items
    responses: list[LLMResponse] = field(default_factory=list)

    @property
    def latency_s(self) -> float:
        return sum(r.latency_s for r in self.responses)

    @property
    def cost_usd(self) -> float | None:
        costs = [r.cost_usd for r in self.responses]
        return None if any(c is None for c in costs) else sum(costs)


def extract(transcript: str, llm: LLM, max_attempts: int = 2) -> ExtractionResult:
    messages = build_messages(transcript)
    result = ExtractionResult(note=None, attempts=0, valid_first_try=False)
    last_data: dict | None = None

    for attempt in range(1, max_attempts + 1):
        response = llm.complete(messages)
        result.responses.append(response)
        result.attempts = attempt
        try:
            last_data = parse_json(response.text)
            result.note = ExtractedNote.model_validate(last_data)
            result.valid = True
            result.valid_first_try = attempt == 1
            return result
        except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
            error = _short_error(e)
            result.errors.append(error)
            messages = messages + [
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": (
                    "Your answer was not valid against the schema:\n"
                    f"{error}\n"
                    "Return the complete corrected JSON object only."
                )},
            ]

    if last_data is not None:
        result.note, result.dropped = salvage(last_data)
    return result
