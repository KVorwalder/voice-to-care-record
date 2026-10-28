import json

from care_record import evaluate as evaluate_module
from care_record.synth import generate_dataset

from .conftest import OracleLLM


def test_oracle_scores_perfectly(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "SUMMARY_CSV", tmp_path / "summary.csv")
    cases = generate_dataset()
    summary = evaluate_module.evaluate(cases, OracleLLM(cases), out_root=tmp_path / "runs")
    assert summary["precision"] == summary["recall"] == 1.0
    assert summary["hallucination_rate"] == 0
    assert summary["schema_valid_first_try"] == 1.0
    # truth marks hedged/contradictory items uncertain, and implausible values fail the range check
    assert summary["review_recall"] == 1.0
    run_dir = next((tmp_path / "runs").iterdir())
    assert json.loads((run_dir / "summary.json").read_text())["n_cases"] == len(cases)

