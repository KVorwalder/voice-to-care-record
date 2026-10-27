"""Care record schema.

`ExtractedNote` is what the LLM must return. `CareRecord` is the final output of the
pipeline: the extracted note plus review flags and run metadata.

The observation areas follow the six topic areas ("Themenfelder") of the German
SIS (Strukturierte Informationssammlung) from the Strukturmodell for care documentation.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SISArea(str, Enum):
    COGNITION = "kognition_kommunikation"  # 1 Kognitive und kommunikative Fähigkeiten
    MOBILITY = "mobilitaet_beweglichkeit"  # 2 Mobilität und Beweglichkeit
    DISEASE = "krankheitsbezogene_anforderungen"  # 3 Krankheitsbezogene Anforderungen und Belastungen
    SELF_CARE = "selbstversorgung"  # 4 Selbstversorgung
    SOCIAL = "soziale_beziehungen"  # 5 Leben in sozialen Beziehungen
    LIVING = "wohnen_haeuslichkeit"  # 6 Wohnen/Häuslichkeit (stationär) bzw. Haushaltsführung (ambulant)


class Grounded(BaseModel):
    """Base for every extracted item: each value must be backed by a transcript quote."""

    evidence: str = Field(
        min_length=1,
        description="Exact, verbatim quote from the transcript this item is based on.",
    )
    uncertain: bool = Field(
        default=False,
        description="True if the transcript is vague, hedged or contradictory about this item.",
    )


class Observation(Grounded):
    area: SISArea
    text: str = Field(min_length=1, description="Short German description of the observation.")


class VitalKind(str, Enum):
    BLOOD_PRESSURE = "blood_pressure"  # mmHg, value = systolic
    PULSE = "pulse"  # bpm
    TEMPERATURE = "temperature"  # °C
    RESPIRATORY_RATE = "respiratory_rate"  # breaths/min
    OXYGEN_SATURATION = "oxygen_saturation"  # %
    BLOOD_GLUCOSE = "blood_glucose"  # mg/dl
    WEIGHT = "weight"  # kg


class VitalSign(Grounded):
    kind: VitalKind
    value: float = Field(description="Numeric value. For blood_pressure: the systolic value.")
    diastolic: float | None = Field(default=None, description="Only for blood_pressure.")

    @model_validator(mode="after")
    def _diastolic_only_for_bp(self):
        if self.kind == VitalKind.BLOOD_PRESSURE and self.diastolic is None:
            raise ValueError("blood_pressure requires a diastolic value")
        if self.kind != VitalKind.BLOOD_PRESSURE and self.diastolic is not None:
            raise ValueError("diastolic is only allowed for blood_pressure")
        return self


class Medication(Grounded):
    name: str = Field(min_length=1, description="Drug name as spoken, e.g. 'Ramipril'.")
    dose: str | None = Field(default=None, description="Dose if stated, e.g. '5 mg'.")
    status: Literal["given", "refused"]


class IncidentType(str, Enum):
    FALL = "fall"
    PAIN = "pain"
    WOUND = "wound"
    OTHER = "other"


class Incident(Grounded):
    type: IncidentType
    description: str = Field(min_length=1, description="Short German description.")
    pain_score: int | None = Field(default=None, ge=0, le=10, description="NRS 0-10 if stated.")


class ExtractedNote(BaseModel):
    """The structured content of one shift note. This is the LLM output contract."""

    observations: list[Observation] = Field(default_factory=list)
    vital_signs: list[VitalSign] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    incidents: list[Incident] = Field(default_factory=list)
    summary: str = Field(default="", description="Two or three sentence German summary.")


SECTIONS = ("observations", "vital_signs", "medications", "incidents")


class ReviewFlag(BaseModel):
    path: str  # e.g. "vital_signs[0]" or "note" for whole-note problems
    reason: Literal["uncertain", "ungrounded", "implausible", "invalid", "extraction_failed"]
    detail: str


class CareRecord(BaseModel):
    transcript: str
    note: ExtractedNote
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    status: Literal["ok", "needs_review", "failed"]
    meta: dict = Field(default_factory=dict)

    def flags_for(self, path: str) -> list[ReviewFlag]:
        return [f for f in self.review_flags if f.path == path]
