import csv
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
    assert summary["overflag_rate"] == 0
    run_dir = next((tmp_path / "runs").iterdir())
    assert json.loads((run_dir / "summary.json").read_text())["n_cases"] == len(cases)


def test_append_summary_keeps_old_runs_aligned_when_columns_change(tmp_path):
    path = tmp_path / "summary.csv"
    evaluate_module.append_summary({"run": "old", "f1": 0.9}, path)
    evaluate_module.append_summary({"run": "new", "misplaced": 2, "f1": 0.95}, path)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0] == {"run": "old", "misplaced": "", "f1": "0.9"}
    assert rows[1] == {"run": "new", "misplaced": "2", "f1": "0.95"}
