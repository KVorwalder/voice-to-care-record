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
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w']+", " ", text)  # drop punctuation, keep letters/digits/umlauts
    return " ".join(text.split())


def is_grounded(evidence: str, transcript: str) -> bool:
    """True if the evidence occurs verbatim in the transcript (ignoring case and punctuation).
    Paraphrased quotes fail on purpose: a human then checks them."""
    ev = normalize(evidence)
    return bool(ev) and ev in normalize(transcript)


def value_in_evidence(vital: VitalSign) -> bool:
    """The number in the record must also appear in its own evidence quote."""
    mentioned = numbers_in(vital.evidence)
    wanted = [vital.value] + ([vital.diastolic] if vital.diastolic is not None else [])
    return all(any(abs(w - m) < 0.05 for m in mentioned) for w in wanted)


def check_plausibility(vital: VitalSign) -> str | None:
    low, high = PLAUSIBLE_RANGES[vital.kind]
    if not low <= vital.value <= high:
        return f"{vital.kind.value} {vital.value:g} outside plausible range {low:g}-{high:g}"
    if vital.kind == VitalKind.BLOOD_PRESSURE:
        dlow, dhigh = DIASTOLIC_RANGE
        if not dlow <= vital.diastolic <= dhigh:
            return f"diastolic {vital.diastolic:g} outside plausible range {dlow:g}-{dhigh:g}"
        if vital.diastolic >= vital.value:
            return f"diastolic {vital.diastolic:g} not below systolic {vital.value:g}"
    return None


def review(note: ExtractedNote, transcript: str) -> list[ReviewFlag]:
    flags: list[ReviewFlag] = []
    for section in SECTIONS:
        for i, item in enumerate(getattr(note, section)):
            path = f"{section}[{i}]"
            if item.uncertain:
                flags.append(ReviewFlag(path=path, reason="uncertain",
                                        detail="model marked this item as uncertain"))
            if not is_grounded(item.evidence, transcript):
                flags.append(ReviewFlag(path=path, reason="ungrounded",
                                        detail="evidence not found in transcript"))
            if isinstance(item, VitalSign):
                if not value_in_evidence(item):
                    flags.append(ReviewFlag(path=path, reason="ungrounded",
                                            detail="value does not appear in its evidence quote"))
                problem = check_plausibility(item)
                if problem:
                    flags.append(ReviewFlag(path=path, reason="implausible", detail=problem))
    return flags
