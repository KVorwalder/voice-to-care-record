from care_record.extraction import extract, parse_json

from .conftest import FakeLLM


def test_parse_json_strips_fences():
    assert parse_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_valid_on_first_try(transcript, good_note):
    llm = FakeLLM(good_note)
    result = extract(transcript, llm)
    assert result.valid and result.valid_first_try and result.attempts == 1
    assert len(result.note.vital_signs) == 2


def test_retry_feeds_error_back(transcript, good_note):
    bad = {**good_note, "vital_signs": [{"kind": "blood_pressure", "value": 130, "evidence": "Blutdruck 130 zu 80"}]}
    llm = FakeLLM(bad, good_note)
    result = extract(transcript, llm)
    assert result.valid and not result.valid_first_try and result.attempts == 2
    retry_prompt = llm.calls[1][-1]["content"]
    assert "diastolic" in retry_prompt  # the validation error was sent back


def test_salvage_keeps_valid_items_when_retry_fails(transcript, good_note):
    bad = {**good_note, "medications": [{"name": "Ramipril", "status": "maybe", "evidence": "Ramipril"}]}
    result = extract(transcript, FakeLLM(bad, bad))
    assert not result.valid and result.attempts == 2
    assert len(result.note.vital_signs) == 2  # good items survive
    assert result.note.medications == []
    assert result.dropped[0][0] == "medications[0]"


def test_unparseable_twice_gives_no_note(transcript):
    result = extract(transcript, FakeLLM("I can't do that", "still no json"))
    assert result.note is None and len(result.errors) == 2
