"""Post-extraction checks that decide what a human must review.

Nothing here changes extracted values: checks only add review flags. Guessing a
"corrected" value would be exactly the kind of silent invention we want to avoid.
"""

import re
from difflib import SequenceMatcher

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

FUZZY_THRESHOLD = 0.85


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w']+", " ", text)  # drop punctuation, keep letters/digits/umlauts
    return " ".join(text.split())


def grounding_score(evidence: str, transcript: str) -> float:
    """1.0 if the evidence occurs verbatim (ignoring case/punctuation), else the best fuzzy
    match against any transcript window of about the same length in words (±1 word, so a
    single added or dropped word still matches)."""
    ev, tr = normalize(evidence), normalize(transcript)
    if not ev:
        return 0.0
    if ev in tr:
        return 1.0
    ev_words, tr_words = ev.split(), tr.split()
    best = 0.0
    for n in {max(1, len(ev_words) - 1), len(ev_words), len(ev_words) + 1}:
        for i in range(max(1, len(tr_words) - n + 1)):
            window = " ".join(tr_words[i : i + n])
            best = max(best, SequenceMatcher(None, ev, window).ratio())
    return best


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
            score = grounding_score(item.evidence, transcript)
            if score < FUZZY_THRESHOLD:
                flags.append(ReviewFlag(path=path, reason="ungrounded",
                                        detail=f"evidence not found in transcript (match {score:.2f})"))
            if isinstance(item, VitalSign):
                if not value_in_evidence(item):
                    flags.append(ReviewFlag(path=path, reason="ungrounded",
                                            detail="value does not appear in its evidence quote"))
                problem = check_plausibility(item)
                if problem:
                    flags.append(ReviewFlag(path=path, reason="implausible", detail=problem))
    return flags
