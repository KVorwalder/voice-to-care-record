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

_ALIASES = {re.sub(r"[^a-z]", "", k.lower()): v for k, v in MED_ALIASES.items()}


def norm_med(name: str) -> str:
    key = re.sub(r"[^a-zäöüß]", "", name.lower())
    canonical = _ALIASES.get(key)
    return re.sub(r"[^a-zäöüß]", "", canonical.lower()) if canonical else key


def entity_key(section: str, item: dict) -> str:
    if section == "vital_signs":
        return item["kind"]
    if section == "medications":
        return norm_med(item["name"])
    if section == "incidents":
        return item["type"]
    return item["area"]


def same_entity(section: str, pred: dict, truth: dict) -> bool:
    a, b = entity_key(section, pred), entity_key(section, truth)
    if section == "medications":  # tolerate "L-Thyroxin" vs "Thyroxin", "Insulin" vs "Insulin glargin"
        return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a) or a in b or b in a))
    return a == b


def value_matches(section: str, pred: dict, truth: dict) -> bool:
    if section == "vital_signs":
        same = abs(float(pred["value"]) - float(truth["value"])) < 0.05
        if truth.get("diastolic") is not None:
            same = same and pred.get("diastolic") is not None and abs(
                float(pred["diastolic"]) - float(truth["diastolic"])) < 0.05
        return same
    if section == "medications":
        return pred["status"] == truth["status"]
    if section == "incidents":
        return pred.get("pain_score") == truth.get("pain_score")
    return True  # observations: area match is the whole check


def evidence_in_other_section(item: dict, section: str, truth: dict) -> bool:
    """True if the item's quote overlaps the evidence of a truth item in another section.
    Containment works both ways, since a model often copies a filler word ("Und ...") along."""
    ev = normalize(item.get("evidence", ""))
    for other in ALL_SECTIONS:
        if other == section:
            continue
        for t in truth.get(other, []):
            t_ev = normalize(t["evidence"])
            shorter = min(ev, t_ev, key=len)
            if len(shorter) >= MIN_QUOTE_CHARS and (t_ev in ev or ev in t_ev):
                return True
    return False


@dataclass
class SectionCounts:
    correct: int = 0
    wrong_value: int = 0
    hallucinated: int = 0
    misplaced: int = 0
    missed: int = 0
    n_pred: int = 0
    n_truth: int = 0

    def add(self, other: "SectionCounts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def precision(self) -> float | None:
        return self.correct / self.n_pred if self.n_pred else None

    @property
    def recall(self) -> float | None:
        return self.correct / self.n_truth if self.n_truth else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class CaseScore:
    sections: dict[str, SectionCounts] = field(default_factory=dict)
    hallucinated_paths: list[str] = field(default_factory=list)
    hallucinations_flagged: int = 0
    expected_review: int = 0  # truth items that should be flagged...
    expected_review_flagged: int = 0  # ...and were matched + flagged
    correct_unflagged_expected: int = 0  # correct items that needed no review
    overflagged: int = 0  # ...but were flagged anyway

    def critical(self) -> SectionCounts:
        total = SectionCounts()
        for s in CRITICAL_SECTIONS:
            total.add(self.sections[s])
        return total


def score_case(pred: dict, truth: dict, expect_review: list[str], flagged_paths: set[str]) -> CaseScore:
    """pred/truth are ExtractedNote dicts; flagged_paths are paths of predicted items with a review flag."""
    score = CaseScore()
    expect = set(expect_review)
    score.expected_review = len(expect)

    for section in ALL_SECTIONS:
        preds, truths = pred.get(section, []), truth.get(section, [])
        counts = SectionCounts(n_pred=len(preds), n_truth=len(truths))
        unused = set(range(len(preds)))

        for t_idx, t in enumerate(truths):
            t_path = f"{section}[{t_idx}]"
            candidates = [i for i in sorted(unused) if same_entity(section, preds[i], t)]
            if not candidates:
                counts.missed += 1
                continue
            # prefer a candidate whose value also matches
            exact = [i for i in candidates if value_matches(section, preds[i], t)]
            p_idx = (exact or candidates)[0]
            unused.discard(p_idx)
            p_flagged = f"{section}[{p_idx}]" in flagged_paths

            if t_path in expect:
                # For items humans must review anyway, finding + flagging is what counts.
                counts.correct += 1
                score.expected_review_flagged += p_flagged
            elif exact:
                counts.correct += 1
                score.correct_unflagged_expected += 1
                score.overflagged += p_flagged
            else:
                counts.wrong_value += 1

        for p_idx in sorted(unused):
            if evidence_in_other_section(preds[p_idx], section, truth):
                counts.misplaced += 1
                continue
            counts.hallucinated += 1
            path = f"{section}[{p_idx}]"
            if section in CRITICAL_SECTIONS:
                score.hallucinated_paths.append(path)
                score.hallucinations_flagged += path in flagged_paths

        score.sections[section] = counts
    return score


def safety_error_score(counts: SectionCounts) -> float | None:
    """Weighted errors per ground-truth item (lower is better)."""
    if not counts.n_truth:
        return None
    weighted = (WEIGHT_HALLUCINATION * counts.hallucinated + WEIGHT_WRONG_VALUE * counts.wrong_value
                + WEIGHT_MISSED * counts.missed + WEIGHT_MISPLACED * counts.misplaced)
    return weighted / counts.n_truth
