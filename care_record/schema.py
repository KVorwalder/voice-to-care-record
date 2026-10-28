"""Care record schema.

`ExtractedNote` is what the LLM must return. `CareRecord` is the final output of the
pipeline: the extracted note plus review flags and run metadata.

The observation areas follow the six topic areas ("Themenfelder") of the German
SIS (Strukturierte Informationssammlung) from the Strukturmodell for care documentation.

All classes are pydantic models: pydantic checks types, allowed values and the custom rules
below when data is loaded, and raises a ValidationError with a readable message otherwise.

Note: pydantic copies class docstrings into the JSON schema, and that schema is part of the
LLM prompt. Classes that end up in the schema are therefore explained with `#` comments above
the class, so documenting the code does not change what the model sees.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# The six SIS topic areas an observation can belong to.
# Values are the exact strings the LLM must use in the JSON ("area": "selbstversorgung").
class SISArea(str, Enum):
    COGNITION = "kognition_kommunikation"  # 1 Kognitive und kommunikative Fähigkeiten
    MOBILITY = "mobilitaet_beweglichkeit"  # 2 Mobilität und Beweglichkeit
    DISEASE = "krankheitsbezogene_anforderungen"  # 3 Krankheitsbezogene Anforderungen und Belastungen
    SELF_CARE = "selbstversorgung"  # 4 Selbstversorgung
    SOCIAL = "soziale_beziehungen"  # 5 Leben in sozialen Beziehungen
    LIVING = "wohnen_haeuslichkeit"  # 6 Wohnen/Häuslichkeit (stationär) bzw. Haushaltsführung (ambulant)


# Base class for every extracted item.
# Fields: evidence  - verbatim quote from the transcript; required and never empty
#         uncertain - set by the LLM when the note is vague, hedged or contradictory
class Grounded(BaseModel):
    """Base for every extracted item: each value must be backed by a transcript quote."""

    evidence: str = Field(
        min_length=1,  # an empty quote would prove nothing
        description="Exact, verbatim quote from the transcript this item is based on.",
    )
    uncertain: bool = Field(
        default=False,
        description="True if the transcript is vague, hedged or contradictory about this item.",
    )


# A free-text observation assigned to one SIS area.
# Fields: area (SISArea), text (short German description), plus evidence/uncertain.
class Observation(Grounded):
    area: SISArea
    text: str = Field(min_length=1, description="Short German description of the observation.")


# The kinds of vital signs the record knows, with their unit as a comment.
class VitalKind(str, Enum):
    BLOOD_PRESSURE = "blood_pressure"  # mmHg, value = systolic
    PULSE = "pulse"  # bpm
    TEMPERATURE = "temperature"  # °C
    RESPIRATORY_RATE = "respiratory_rate"  # breaths/min
    OXYGEN_SATURATION = "oxygen_saturation"  # %
    BLOOD_GLUCOSE = "blood_glucose"  # mg/dl
    WEIGHT = "weight"  # kg


# One measured vital sign.
# Fields: kind (VitalKind), value (number; systolic for blood pressure),
#         diastolic (only for blood pressure), plus evidence/uncertain.
class VitalSign(Grounded):
    kind: VitalKind
    value: float = Field(description="Numeric value. For blood_pressure: the systolic value.")
    diastolic: float | None = Field(default=None, description="Only for blood_pressure.")

    @model_validator(mode="after")
    def _diastolic_only_for_bp(self):
        """Cross-field rule that a simple type check cannot express.

        Input:  the already type-checked VitalSign
        Output: the same object, or a ValueError (pydantic turns it into a ValidationError)
        """
        # Blood pressure is always two numbers ("130 zu 80"), so the second one is required
        if self.kind == VitalKind.BLOOD_PRESSURE and self.diastolic is None:
            raise ValueError("blood_pressure requires a diastolic value")
        # Every other vital sign is a single number
        if self.kind != VitalKind.BLOOD_PRESSURE and self.diastolic is not None:
            raise ValueError("diastolic is only allowed for blood_pressure")
        return self


# A medication that was given or refused during the shift.
# Fields: name, dose (optional), status ("given" or "refused"), plus evidence/uncertain.
class Medication(Grounded):
    name: str = Field(min_length=1, description="Drug name as spoken, e.g. 'Ramipril'.")
    dose: str | None = Field(default=None, description="Dose if stated, e.g. '5 mg'.")
    status: Literal["given", "refused"]  # Literal: any other string is rejected


# Kinds of incidents that matter for care risk (falls, pain, wounds, anything acute).
class IncidentType(str, Enum):
    FALL = "fall"
    PAIN = "pain"
    WOUND = "wound"
    OTHER = "other"


# An incident or risk during the shift.
# Fields: type (IncidentType), description (short German text),
#         pain_score (0-10 on the numeric rating scale, only if stated), plus evidence/uncertain.
class Incident(Grounded):
    type: IncidentType
    description: str = Field(min_length=1, description="Short German description.")
    pain_score: int | None = Field(default=None, ge=0, le=10, description="NRS 0-10 if stated.")


# The LLM output contract: its JSON schema is put into the prompt, and the LLM answer is
# validated against it. Every list may be empty when the note mentions nothing of that kind.
class ExtractedNote(BaseModel):
    """The structured content of one shift note. This is the LLM output contract."""

    observations: list[Observation] = Field(default_factory=list)
    vital_signs: list[VitalSign] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    incidents: list[Incident] = Field(default_factory=list)
    summary: str = Field(default="", description="Two or three sentence German summary.")


# Names of the item lists in ExtractedNote, used to loop over all extracted items
SECTIONS = ("observations", "vital_signs", "medications", "incidents")


class ReviewFlag(BaseModel):
    """A reason why a human must check part of the record.

    Fields:
        path   - which item is affected, e.g. "vital_signs[0]", or "note" for the whole note
        reason - short category of the problem (see the Literal below)
        detail - human-readable explanation
    """

    path: str
    reason: Literal["uncertain", "ungrounded", "implausible", "invalid", "extraction_failed"]
    detail: str


class CareRecord(BaseModel):
    """Final output of the pipeline for one note.

    Fields:
        transcript   - the text the record was extracted from
        note         - the extracted, schema-valid content
        review_flags - everything a human must check
        status       - "ok" (no flags), "needs_review" (some flags) or "failed" (no usable output)
        meta         - run information: model, attempts, errors, latency
    """

    transcript: str
    note: ExtractedNote
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    status: Literal["ok", "needs_review", "failed"]
    meta: dict = Field(default_factory=dict)

    def flags_for(self, path: str) -> list[ReviewFlag]:
        """All review flags for one item.

        Input:  path of an item, e.g. "vital_signs[0]"
        Output: list of ReviewFlag for that item (empty if it needs no review)
        """
        return [f for f in self.review_flags if f.path == path]
