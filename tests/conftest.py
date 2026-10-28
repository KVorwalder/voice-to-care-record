import json

import pytest

from care_record.llm import LLMResponse


class FakeLLM:
    """Returns scripted responses in order and records the messages it was sent."""

    model = "fake/scripted"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        text = self.responses.pop(0)
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        return LLMResponse(text=text, latency_s=0.01)


class OracleLLM:
    """Answers every transcript with its ground truth: the pipeline's best case."""

    model = "fake/oracle"

    def __init__(self, cases):
        self.by_transcript = {c["transcript"]: c["truth"] for c in cases}

    def complete(self, messages):
        transcript = messages[-1]["content"].removeprefix("Transcript:\n")
        return LLMResponse(text=json.dumps(self.by_transcript[transcript], ensure_ascii=False),
                           latency_s=0.0)


TRANSCRIPT = ("So, Frau Berger, Frühdienst. Ähm, Blutdruck 130 zu 80. Puls lag bei 72. "
              "Ramipril 5 mg gegeben. Klagt über Schmerzen im Knie, Schmerzskala 6. "
              "Tochter war zu Besuch. Das war's.")


@pytest.fixture
def transcript():
    return TRANSCRIPT


@pytest.fixture
def good_note():
    return {
        "observations": [{"area": "soziale_beziehungen", "text": "Besuch der Tochter",
                          "evidence": "Tochter war zu Besuch"}],
        "vital_signs": [
            {"kind": "blood_pressure", "value": 130, "diastolic": 80, "evidence": "Blutdruck 130 zu 80"},
            {"kind": "pulse", "value": 72, "evidence": "Puls lag bei 72"},
        ],
        "medications": [{"name": "Ramipril", "dose": "5 mg", "status": "given",
                         "evidence": "Ramipril 5 mg gegeben"}],
        "incidents": [{"type": "pain", "description": "Knieschmerzen", "pain_score": 6,
                       "evidence": "Klagt über Schmerzen im Knie, Schmerzskala 6"}],
        "summary": "Unauffälliger Frühdienst mit Knieschmerzen.",
    }
