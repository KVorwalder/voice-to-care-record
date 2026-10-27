from care_record.pipeline import process_transcript
from care_record.schema import ExtractedNote, VitalSign
from care_record.validation import check_plausibility, grounding_score, review, value_in_evidence

from .conftest import FakeLLM


def test_grounding_exact_ignores_case_and_punctuation(transcript):
    assert grounding_score("blutdruck 130 zu 80", transcript) == 1.0


def test_grounding_tolerates_small_differences(transcript):
    assert grounding_score("Puls lag bei 72.", transcript) == 1.0
    assert grounding_score("Puls lag so bei 72", transcript) > 0.85


def test_grounding_rejects_invented_quote(transcript):
    assert grounding_score("Metformin 500 gegeben", transcript) < 0.85


def test_value_must_appear_in_evidence():
    assert value_in_evidence(VitalSign(kind="pulse", value=72, evidence="Puls lag bei 72"))
    assert not value_in_evidence(VitalSign(kind="pulse", value=82, evidence="Puls lag bei 72"))
    assert value_in_evidence(VitalSign(kind="temperature", value=38.5,
                                       evidence="Temperatur achtunddreißig komma fünf"))


def test_plausibility():
    assert check_plausibility(VitalSign(kind="temperature", value=37.2, evidence="x")) is None
    assert check_plausibility(VitalSign(kind="temperature", value=3.8, evidence="x"))
    assert check_plausibility(VitalSign(kind="blood_pressure", value=80, diastolic=120, evidence="x"))


def test_review_flags_hallucinated_medication(transcript, good_note):
    good_note["medications"].append({"name": "Metformin", "status": "given", "evidence": "Metformin 500 gegeben"})
    flags = review(ExtractedNote.model_validate(good_note), transcript)
    assert [(f.path, f.reason) for f in flags] == [("medications[1]", "ungrounded")]


def test_review_flags_uncertain(transcript, good_note):
    good_note["vital_signs"][1]["uncertain"] = True
    flags = review(ExtractedNote.model_validate(good_note), transcript)
    assert [(f.path, f.reason) for f in flags] == [("vital_signs[1]", "uncertain")]


def test_pipeline_ok_and_failed(transcript, good_note):
    assert process_transcript(transcript, FakeLLM(good_note)).status == "ok"
    failed = process_transcript(transcript, FakeLLM("nope", "nope"))
    assert failed.status == "failed" and failed.review_flags[0].reason == "extraction_failed"
