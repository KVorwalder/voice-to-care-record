"""Run the pipeline over a labelled dataset and write per-case and summary results."""

import csv
import json
import statistics
import time
import traceback
from datetime import datetime
from pathlib import Path

from .llm import LLM
from .metrics import (ALL_SECTIONS, CRITICAL_SECTIONS, CaseScore, SectionCounts, safety_error_score,
                      score_case)
from .pipeline import process_transcript
from .schema import CareRecord

SUMMARY_CSV = Path("results/summary.csv")


def _transcript_for(case: dict, input_mode: str, audio_dir: Path, whisper_model: str | None) -> tuple[str, float]:
    if input_mode == "text":
        return case["transcript"], 0.0
    from .stt import transcribe

    audio = audio_dir / f"{case['id']}.wav"
    stt = transcribe(audio, model_size=whisper_model) if whisper_model else transcribe(audio)
    return stt.text, stt.latency_s


def _r(x: float | None, nd: int = 3):
    return None if x is None else round(x, nd)


def evaluate(cases: list[dict], llm: LLM, input_mode: str = "text", audio_dir: Path = Path("data/audio"),
             whisper_model: str | None = None, out_root: Path = Path("results/runs")) -> dict:
    run_name = f"{datetime.now():%Y%m%d-%H%M%S}_{input_mode}_{llm.model.replace('/', '_').replace(':', '_')}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, scores, records = [], [], []
    for i, case in enumerate(cases, 1):
        start = time.perf_counter()
        try:
            transcript, stt_latency = _transcript_for(case, input_mode, audio_dir, whisper_model)
            record = process_transcript(transcript, llm)
            record.meta["stt_latency_s"] = round(stt_latency, 3)
        except Exception as e:  # network/provider errors: count as a failed note, keep going
            record = CareRecord(transcript=case["transcript"], note={}, status="failed",
                                meta={"model": llm.model, "attempts": 0, "valid_first_try": False,
                                      "valid_after_retry": False, "errors": [f"{type(e).__name__}: {e}"],
                                      "latency_s": None, "cost_usd": None, "stt_latency_s": None})
            traceback.print_exc()
        total_latency = time.perf_counter() - start

        flagged = {f.path for f in record.review_flags}
        score = score_case(record.note.model_dump(mode="json"), case["truth"], case["expect_review"], flagged)
        scores.append(score)
        crit = score.critical()
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "status": record.status,
            "valid_first_try": record.meta["valid_first_try"],
            "valid_after_retry": record.meta["valid_after_retry"],
            "attempts": record.meta["attempts"],
            "correct": crit.correct,
            "wrong_value": crit.wrong_value,
            "hallucinated": crit.hallucinated,
            "missed": crit.missed,
            "hallucinated_paths": ";".join(score.hallucinated_paths),
            "hallucinations_flagged": score.hallucinations_flagged,
            "obs_correct": score.sections["observations"].correct,
            "obs_pred": score.sections["observations"].n_pred,
            "obs_truth": score.sections["observations"].n_truth,
            "n_flags": len(record.review_flags),
            "latency_s": _r(total_latency),
            "cost_usd": record.meta["cost_usd"],
        })
        records.append({"id": case["id"], "record": record.model_dump(mode="json")})
        print(f"[{i}/{len(cases)}] {case['id']} {case['category']:<20} status={record.status:<12} "
              f"correct={crit.correct} wrong={crit.wrong_value} halluc={crit.hallucinated} "
              f"missed={crit.missed} {total_latency:.1f}s")

    summary = summarize(scores, rows, llm.model, input_mode, run_name)

    with (out_dir / "per_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_summary(summary)
    print(f"\nResults written to {out_dir}")
    return summary


def summarize(scores: list[CaseScore], rows: list[dict], model: str, input_mode: str, run_name: str) -> dict:
    sections = {s: SectionCounts() for s in ALL_SECTIONS}
    for sc in scores:
        for s in ALL_SECTIONS:
            sections[s].add(sc.sections[s])
    crit = SectionCounts()
    for s in CRITICAL_SECTIONS:
        crit.add(sections[s])

    n = len(rows)
    latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    halluc_total = sum(sc.critical().hallucinated for sc in scores)
    expected = sum(sc.expected_review for sc in scores)
    should_not_flag = sum(sc.correct_unflagged_expected for sc in scores)

    summary = {
        "run": run_name,
        "model": model,
        "input": input_mode,
        "n_cases": n,
        "schema_valid_first_try": _r(sum(bool(r["valid_first_try"]) for r in rows) / n),
        "schema_valid_after_retry": _r(sum(bool(r["valid_after_retry"]) for r in rows) / n),
        "precision": _r(crit.precision),
        "recall": _r(crit.recall),
        "f1": _r(crit.f1),
        "hallucination_rate": _r(crit.hallucinated / crit.n_pred if crit.n_pred else None),
        "notes_with_hallucination": _r(sum(sc.critical().hallucinated > 0 for sc in scores) / n),
        "hallucinations_caught_by_flags": _r(sum(sc.hallucinations_flagged for sc in scores) / halluc_total
                                             if halluc_total else None),
        "wrong_values": crit.wrong_value,
        "hallucinations": crit.hallucinated,
        "missed": crit.missed,
        "safety_error_score": _r(safety_error_score(crit)),
        "review_recall": _r(sum(sc.expected_review_flagged for sc in scores) / expected if expected else None),
        "overflag_rate": _r(sum(sc.overflagged for sc in scores) / should_not_flag if should_not_flag else None),
        "latency_mean_s": _r(statistics.mean(latencies), 2) if latencies else None,
        "latency_p95_s": _r(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 2) if latencies else None,
        "cost_per_note_usd": _r(statistics.mean(costs), 5) if len(costs) == n else None,
    }
    for s in ALL_SECTIONS:
        summary[f"{s}_p"] = _r(sections[s].precision)
        summary[f"{s}_r"] = _r(sections[s].recall)
        summary[f"{s}_f1"] = _r(sections[s].f1)
    return summary


def append_summary(summary: dict, path: Path | None = None) -> None:
    path = path or SUMMARY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        if new:
            writer.writeheader()
        writer.writerow(summary)
