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

# Which pydantic model validates the items of each section (used by salvage())
ITEM_MODELS = {
    "observations": Observation,
    "vital_signs": VitalSign,
    "medications": Medication,
    "incidents": Incident,
}

# Instructions for the LLM. {schema} is replaced by the JSON schema of ExtractedNote.
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
    """Build the chat for the first extraction attempt.

    Input:  transcript - the dictated note as text
    Output: [system message with rules + JSON schema, user message with the transcript]
    """
    # pydantic generates the JSON schema from the classes in schema.py, so prompt and
    # validation can never disagree about the expected format
    schema = json.dumps(ExtractedNote.model_json_schema(), ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)},
        {"role": "user", "content": f"Transcript:\n{transcript}"},
    ]


def parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating markdown fences and leading/trailing prose.

    Input:  text - raw LLM answer, e.g. 'Sure! ```json {"vital_signs": []} ```'
    Output: the parsed JSON object as a dict
    Raises: ValueError if there is no valid JSON object in the text
    """
    text = text.strip()
    # Models often wrap JSON in ```json ... ``` - keep only the part inside the fence
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    # If there is prose around the JSON, cut from the first "{" to the last "}"
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    # json.loads raises JSONDecodeError (a ValueError) for anything that is not valid JSON
    data = json.loads(text)
    # A list or a plain string is valid JSON, but not a note
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data


def salvage(data: dict) -> tuple[ExtractedNote, list[tuple[str, str]]]:
    """Fallback when the whole answer is invalid: keep every item that validates on its own.

    One broken item (e.g. a blood pressure without diastolic) should not throw away the
    other correct items of the note.

    Input:  data - the parsed (but schema-invalid) JSON answer of the LLM
    Output: (note with only the valid items, [(path, error) for every dropped item])
    """
    kept: dict = {name: [] for name in SECTIONS}
    dropped = []
    for name in SECTIONS:
        items = data.get(name) or []
        # A section that is not a list at all cannot be used
        if not isinstance(items, list):
            dropped.append((name, "expected a list"))
            continue
        # Validate each item separately with its own model; keep it or record why it failed
        for i, raw in enumerate(items):
            try:
                kept[name].append(ITEM_MODELS[name].model_validate(raw))
            except ValidationError as e:
                dropped.append((f"{name}[{i}]", _short_error(e)))
    # Keep the summary only if it is a string
    summary = data.get("summary") if isinstance(data.get("summary"), str) else ""
    return ExtractedNote(**kept, summary=summary), dropped


def _short_error(e: Exception) -> str:
    """Turn an exception into a short, one-line message for the retry prompt and the logs.

    Input:  e - a pydantic ValidationError or any other exception (e.g. a JSON error)
    Output: e.g. "vital_signs.0: Value error, blood_pressure requires a diastolic value"
    """
    if isinstance(e, ValidationError):
        # One "location: message" pair per problem, e.g. "medications.0.status: Input should be ..."
        return "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()
        )
    return str(e)


@dataclass
class ExtractionResult:
    """Everything extract() found out about one transcript.

    Fields:
        note            - the extracted note, or None if no attempt produced parseable JSON
        attempts        - number of LLM calls made (1 or 2)
        valid_first_try - the first answer was fully schema-valid
        valid           - some answer was fully schema-valid (no salvage needed)
        errors          - error message of every failed attempt
        dropped         - items removed by salvage(), as (path, error)
        responses       - raw LLM responses, kept for latency and debugging
    """

    note: ExtractedNote | None
    attempts: int
    valid_first_try: bool
    valid: bool = False  # fully schema-valid on some attempt (no salvage needed)
    errors: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # salvaged-away items
    responses: list[LLMResponse] = field(default_factory=list)

    @property
    def latency_s(self) -> float:
        """Total time spent in LLM calls, in seconds."""
        return sum(r.latency_s for r in self.responses)


def extract(transcript: str, llm: LLM, max_attempts: int = 2) -> ExtractionResult:
    """Turn a transcript into a schema-valid ExtractedNote, with one retry and a salvage fallback.

    Input:  transcript   - the dictated note as text
            llm          - any object with a complete(messages) method (see llm.py)
            max_attempts - how many LLM calls are allowed (1 first try + 1 retry)
    Output: ExtractionResult (note may be None if the model never returned parseable JSON)
    """
    messages = build_messages(transcript)
    result = ExtractionResult(note=None, attempts=0, valid_first_try=False)
    last_data: dict | None = None  # last answer that was valid JSON, used by salvage()

    for attempt in range(1, max_attempts + 1):
        # Ask the model and remember its raw response
        response = llm.complete(messages)
        result.responses.append(response)
        result.attempts = attempt
        try:
            # Step 1: text -> dict (fails on broken JSON)
            last_data = parse_json(response.text)
            # Step 2: dict -> ExtractedNote (fails if the schema rules are not met)
            result.note = ExtractedNote.model_validate(last_data)
            # Both steps passed: done
            result.valid = True
            result.valid_first_try = attempt == 1
            return result
        except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
            error = _short_error(e)
            result.errors.append(error)
            # Retry: show the model its own answer plus the exact error and ask for a fix
            messages = messages + [
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": (
                    "Your answer was not valid against the schema:\n"
                    f"{error}\n"
                    "Return the complete corrected JSON object only."
                )},
            ]

    # All attempts failed. If the last answer was at least valid JSON, keep its valid items;
    # otherwise result.note stays None and the pipeline marks the note as failed.
    if last_data is not None:
        result.note, result.dropped = salvage(last_data)
    return result
