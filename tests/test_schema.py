import pytest
from pydantic import ValidationError

from care_record.schema import ExtractedNote, Incident, VitalSign


def test_good_note_validates(good_note):
    note = ExtractedNote.model_validate(good_note)
    assert note.vital_signs[0].diastolic == 80


def test_blood_pressure_requires_diastolic():
    with pytest.raises(ValidationError):
        VitalSign(kind="blood_pressure", value=130, evidence="RR 130")


def test_diastolic_only_for_blood_pressure():
    with pytest.raises(ValidationError):
        VitalSign(kind="pulse", value=72, diastolic=80, evidence="Puls 72")


def test_pain_score_range():
    with pytest.raises(ValidationError):
        Incident(type="pain", description="x", pain_score=11, evidence="Schmerz 11")


def test_evidence_required():
    with pytest.raises(ValidationError):
        VitalSign(kind="pulse", value=72, evidence="")


def test_unknown_vital_kind_rejected():
    with pytest.raises(ValidationError):
        VitalSign(kind="heart_rate", value=72, evidence="Puls 72")
