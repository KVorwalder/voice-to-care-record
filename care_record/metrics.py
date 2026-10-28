"""Field-level scoring of an extracted note against ground truth.

Items are matched per section by entity (vital kind, drug name, incident type, SIS area).
A matched pair is then either correct or a *wrong value*. Unmatched truth items are *missed*.
An unmatched prediction is *misplaced* if its quote is the evidence of a truth item in another
section (a real statement filed in the wrong place, e.g. "Luftnot" as an incident instead of
an observation), otherwise a *hallucination* (an entry the note does not support).

    precision = correct / predicted      recall = correct / truth

Hallucinations are reported separately and weighted highest: in care documentation an
invented entry ("Ramipril given") is worse than a missing one, because people act on it.

Observations are free text assigned to an SIS area, which is partly a judgement call, so
they are scored on their own and kept out of the hallucination rate. The "critical"
sections are vital signs, medications and incidents.
"""

import re
from dataclasses import dataclass, field

from .synth import MED_ALIASES
from .validation import normalize

CRITICAL_SECTIONS = ("vital_signs", "medications", "incidents")
ALL_SECTIONS = ("observations",) + CRITICAL_SECTIONS

# Weights for the safety-weighted error score.
WEIGHT_HALLUCINATION, WEIGHT_WRONG_VALUE, WEIGHT_MISSED, WEIGHT_MISPLACED = 3, 2, 1, 1

# Shorter quotes ("ja", "x") would match almost any evidence by containment.
MIN_QUOTE_CHARS = 5

# Brand/short names in the same form norm_med() produces, e.g. "novalgin" -> "Novaminsulfon"
_ALIASES = {re.sub(r"[^a-z]", "", k.lower()): v for k, v in MED_ALIASES.items()}


def norm_med(name: str) -> str:
    """Normalize a drug name so that spelling variants and brand names compare equal.

    Input:  drug name as written by the model or the truth, e.g. "Novalgin", "L-Thyroxin"
    Output: lower-case letters only, brand names mapped to the active ingredient,
            e.g. "novaminsulfon", "lthyroxin"
    """
    # Keep only letters ("L-Thyroxin 50" -> "lthyroxin")
    key = re.sub(r"[^a-zäöüß]", "", name.lower())
    # Map known brand/short names to the canonical name, normalized the same way
    canonical = _ALIASES.get(key)
    return re.sub(r"[^a-zäöüß]", "", canonical.lower()) if canonical else key


def entity_key(section: str, item: dict) -> str:
    """The 'what is it about' of an item, used to pair prediction and truth.

    Input:  section name and one item as dict
    Output: vital kind, normalized drug name, incident type or SIS area
    """
    if section == "vital_signs":
        return item["kind"]
    if section == "medications":
        return norm_med(item["name"])
    if section == "incidents":
        return item["type"]
    return item["area"]


def same_entity(section: str, pred: dict, truth: dict) -> bool:
    """Decide whether a predicted item and a truth item are about the same thing.

    Input:  section name, predicted item, truth item (dicts)
    Output: True if they describe the same entity (values may still differ)
    """
    a, b = entity_key(section, pred), entity_key(section, truth)
    if section == "medications":  # tolerate "L-Thyroxin" vs "Thyroxin", "Insulin" vs "Insulin glargin"
        return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a) or a in b or b in a))
    return a == b


def value_matches(section: str, pred: dict, truth: dict) -> bool:
    """Decide whether the value of a matched pair is correct.

    Input:  section name, predicted item, truth item (already known to be the same entity)
    Output: True if the value that matters for this section is equal
    """
    # Vital signs: the number(s) must match (small tolerance for float rounding)
    if section == "vital_signs":
        same = abs(float(pred["value"]) - float(truth["value"])) < 0.05
        if truth.get("diastolic") is not None:
            same = same and pred.get("diastolic") is not None and abs(
                float(pred["diastolic"]) - float(truth["diastolic"])) < 0.05
        return same
    # Medications: given vs. refused is what matters
    if section == "medications":
        return pred["status"] == truth["status"]
    # Incidents: the pain score (None if not stated)
    if section == "incidents":
        return pred.get("pain_score") == truth.get("pain_score")
    return True  # observations: area match is the whole check


def evidence_in_other_section(item: dict, section: str, truth: dict) -> bool:
    """Check whether an unmatched prediction is a real statement filed in the wrong section.

    Containment works both ways, since a model often copies a filler word ("Und ...") along.

    Input:  item    - the unmatched predicted item
            section - the section it was predicted in
            truth   - the whole ground-truth note
    Output: True if its quote overlaps the evidence of a truth item in another section
    """
    ev = normalize(item.get("evidence", ""))
    for other in ALL_SECTIONS:
        if other == section:
            continue
        for t in truth.get(other, []):
            t_ev = normalize(t["evidence"])
            # Very short quotes are ignored, they would match almost anything
            shorter = min(ev, t_ev, key=len)
            if len(shorter) >= MIN_QUOTE_CHARS and (t_ev in ev or ev in t_ev):
                return True
    return False


@dataclass
class SectionCounts:
    """Error counts for one section (or a sum of sections), plus precision/recall/F1.

    Fields: correct, wrong_value, hallucinated, misplaced, missed,
            n_pred (number of predicted items), n_truth (number of truth items)
    """

    correct: int = 0
    wrong_value: int = 0
    hallucinated: int = 0
    misplaced: int = 0
    missed: int = 0
    n_pred: int = 0
    n_truth: int = 0

    def add(self, other: "SectionCounts") -> None:
        """Add the counts of another SectionCounts to this one (field by field)."""
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def precision(self) -> float | None:
        """Share of predicted items that are correct (None if nothing was predicted)."""
        return self.correct / self.n_pred if self.n_pred else None

    @property
    def recall(self) -> float | None:
        """Share of truth items that were found correctly (None if there was nothing to find)."""
        return self.correct / self.n_truth if self.n_truth else None

    @property
    def f1(self) -> float | None:
        """Harmonic mean of precision and recall (None if one of them is undefined)."""
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class CaseScore:
    """Score of one note.

    Fields:
        sections                - SectionCounts per section name
        hallucinated_paths      - paths of hallucinated items in critical sections (for debugging)
        expected_review         - number of truth items a human must review
        expected_review_flagged - how many of those were found and flagged
    """

    sections: dict[str, SectionCounts] = field(default_factory=dict)
    hallucinated_paths: list[str] = field(default_factory=list)
    expected_review: int = 0  # truth items that should be flagged...
    expected_review_flagged: int = 0  # ...and were matched + flagged

    def critical(self) -> SectionCounts:
        """Sum of the counts of the critical sections (vital signs, medications, incidents)."""
        total = SectionCounts()
        for s in CRITICAL_SECTIONS:
            total.add(self.sections[s])
        return total


def score_case(pred: dict, truth: dict, expect_review: list[str], flagged_paths: set[str]) -> CaseScore:
    """Compare one predicted note with its ground truth.

    Input:  pred          - predicted ExtractedNote as dict
            truth         - ground-truth ExtractedNote as dict
            expect_review - paths of truth items that must be flagged, e.g. ["vital_signs[0]"]
            flagged_paths - paths of predicted items that received a review flag
    Output: CaseScore with counts per section and review statistics
    """
    score = CaseScore()
    expect = set(expect_review)
    score.expected_review = len(expect)

    for section in ALL_SECTIONS:
        preds, truths = pred.get(section, []), truth.get(section, [])
        counts = SectionCounts(n_pred=len(preds), n_truth=len(truths))
        unused = set(range(len(preds)))  # predictions not yet paired with a truth item

        # Pair every truth item with an unused prediction about the same entity
        for t_idx, t in enumerate(truths):
            t_path = f"{section}[{t_idx}]"
            candidates = [i for i in sorted(unused) if same_entity(section, preds[i], t)]
            if not candidates:
                counts.missed += 1  # nothing predicted for this truth item
                continue
            # prefer a candidate whose value also matches
            exact = [i for i in candidates if value_matches(section, preds[i], t)]
            p_idx = (exact or candidates)[0]
            unused.discard(p_idx)

            if t_path in expect:
                # For items humans must review anyway, finding + flagging is what counts.
                counts.correct += 1
                score.expected_review_flagged += f"{section}[{p_idx}]" in flagged_paths
            elif exact:
                counts.correct += 1
            else:
                counts.wrong_value += 1  # right entity, wrong value (e.g. pulse 76 instead of 72)

        # Predictions left over have no truth partner: misplaced or hallucinated
        for p_idx in sorted(unused):
            if evidence_in_other_section(preds[p_idx], section, truth):
                counts.misplaced += 1
                continue
            counts.hallucinated += 1
            if section in CRITICAL_SECTIONS:
                score.hallucinated_paths.append(f"{section}[{p_idx}]")

        score.sections[section] = counts
    return score


def safety_error_score(counts: SectionCounts) -> float | None:
    """Weighted errors per ground-truth item (lower is better).

    Input:  counts - usually the critical-section counts of a whole run
    Output: (3*hallucinated + 2*wrong + 1*missed + 1*misplaced) / truth items, or None
    """
    if not counts.n_truth:
        return None
    weighted = (WEIGHT_HALLUCINATION * counts.hallucinated + WEIGHT_WRONG_VALUE * counts.wrong_value
                + WEIGHT_MISSED * counts.missed + WEIGHT_MISPLACED * counts.misplaced)
    return weighted / counts.n_truth
