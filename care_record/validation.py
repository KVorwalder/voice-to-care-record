"""Post-extraction checks that decide what a human must review.

Nothing here changes extracted values: checks only add review flags. Guessing a
"corrected" value would be exactly the kind of silent invention we want to avoid.
"""

import re

from .numbers_de import numbers_in
from .schema import SECTIONS, ExtractedNote, ReviewFlag, VitalKind, VitalSign

# Physiologically plausible ranges for an adult in elderly care. Values outside are not
# necessarily wrong (could be an emergency) but are far more often dictation/STT errors,
# so they always go to a human.
PLAUSIBLE_RANGES: dict[VitalKind, tuple[float, float]] = {
    VitalKind.BLOOD_PRESSURE: (60, 260),  # systolic, mmHg
    VitalKind.PULSE: (30, 200),  # bpm
    VitalKind.TEMPERATURE: (34.0, 42.5),  # °C
    VitalKind.RESPIRATORY_RATE: (6, 40),  # /min
    VitalKind.OXYGEN_SATURATION: (70, 100),  # %
    VitalKind.BLOOD_GLUCOSE: (30, 600),  # mg/dl
    VitalKind.WEIGHT: (25, 250),  # kg
}
DIASTOLIC_RANGE = (30, 150)


def normalize(text: str) -> str:
    """Bring text into a canonical form so that comparisons ignore case and punctuation.

    Input:  any text, e.g. "Puls lag bei 72."
    Output: lower-case words separated by single spaces, e.g. "puls lag bei 72"
    """
    # Lower-case; unify the typographic apostrophe with the plain one
    text = text.lower().replace("’", "'")
    # Replace every run of punctuation/whitespace by one space (letters, digits, umlauts stay)
    text = re.sub(r"[^\w']+", " ", text)
    # Collapse multiple spaces and trim the ends
    return " ".join(text.split())


def is_grounded(evidence: str, transcript: str) -> bool:
    """Check that the quote the LLM gave as proof really occurs in the transcript.

    Paraphrased quotes fail on purpose: a human then checks them.

    Input:  evidence   - quote copied by the LLM for one extracted item
            transcript - full text of the dictated note
    Output: True if the quote occurs verbatim (ignoring case and punctuation)
    """
    # Normalize both texts the same way, so "Puls 72." matches "puls 72"
    ev = normalize(evidence)
    # An empty quote proves nothing; otherwise a plain substring check
    return bool(ev) and ev in normalize(transcript)


def value_in_evidence(vital: VitalSign) -> bool:
    """Check that the number in the record also appears in its own evidence quote.

    Catches misread numbers: e.g. value 206 with the (real) quote "BZ zweihundertsechsundvierzig".

    Input:  vital - one extracted vital sign
    Output: True if every recorded number (value and, for blood pressure, diastolic)
            is mentioned in the quote, as digits or as German words
    """
    # All numbers mentioned in the quote, e.g. [246.0] for "BZ zweihundertsechsundvierzig"
    mentioned = numbers_in(vital.evidence)
    # The numbers the record claims: the value, plus the diastolic value for blood pressure
    wanted = [vital.value] + ([vital.diastolic] if vital.diastolic is not None else [])
    # Every wanted number must match one mentioned number (small tolerance for 37.5 vs 37.50)
    return all(any(abs(w - m) < 0.05 for m in mentioned) for w in wanted)


def check_plausibility(vital: VitalSign) -> str | None:
    """Check a vital sign against the plausible range for its kind.

    Input:  vital - one extracted vital sign
    Output: None if plausible, otherwise a short explanation of the problem
    """
    # Main value (systolic for blood pressure) must be inside the range for its kind
    low, high = PLAUSIBLE_RANGES[vital.kind]
    if not low <= vital.value <= high:
        return f"{vital.kind.value} {vital.value:g} outside plausible range {low:g}-{high:g}"
    # Blood pressure has two extra rules for the diastolic value
    if vital.kind == VitalKind.BLOOD_PRESSURE:
        dlow, dhigh = DIASTOLIC_RANGE
        if not dlow <= vital.diastolic <= dhigh:
            return f"diastolic {vital.diastolic:g} outside plausible range {dlow:g}-{dhigh:g}"
        # Diastolic is always lower than systolic; otherwise the numbers were swapped or misheard
        if vital.diastolic >= vital.value:
            return f"diastolic {vital.diastolic:g} not below systolic {vital.value:g}"
    return None


def review(note: ExtractedNote, transcript: str) -> list[ReviewFlag]:
    """Run all checks on every extracted item and collect review flags.

    Input:  note       - the schema-valid extracted note
            transcript - the text it was extracted from
    Output: list of ReviewFlag (empty if nothing needs a human)
    """
    flags: list[ReviewFlag] = []
    # Go through every item of every section, e.g. "vital_signs[0]", "medications[1]", ...
    for section in SECTIONS:
        for i, item in enumerate(getattr(note, section)):
            path = f"{section}[{i}]"
            # 1. The model itself said it is unsure (hedged or contradictory statement)
            if item.uncertain:
                flags.append(ReviewFlag(path=path, reason="uncertain",
                                        detail="model marked this item as uncertain"))
            # 2. The quote is not in the transcript: possible hallucination
            if not is_grounded(item.evidence, transcript):
                flags.append(ReviewFlag(path=path, reason="ungrounded",
                                        detail="evidence not found in transcript"))
            # Extra checks only for numbers
            if isinstance(item, VitalSign):
                # 3. The quote is real, but the number does not match it: misread number
                if not value_in_evidence(item):
                    flags.append(ReviewFlag(path=path, reason="ungrounded",
                                            detail="value does not appear in its evidence quote"))
                # 4. The number is physiologically implausible
                problem = check_plausibility(item)
                if problem:
                    flags.append(ReviewFlag(path=path, reason="implausible", detail=problem))
    return flags
