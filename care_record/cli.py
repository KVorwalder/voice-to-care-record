"""Command line interface.

    care generate [--seed 42] [--out data/synthetic/cases.jsonl]
    care process <audio_or_text_file> [--model ... | --local]
    care evaluate <dataset.jsonl> [--model ... | --local] [--input text|audio] [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

from .llm import DEFAULT_API_MODEL, DEFAULT_LOCAL_MODEL, LiteLLM


def _model(args) -> str:
    if args.model:
        return args.model
    return DEFAULT_LOCAL_MODEL if args.local else DEFAULT_API_MODEL


def cmd_generate(args) -> None:
    from .synth import generate_dataset, write_jsonl

    cases = generate_dataset(args.seed)
    write_jsonl(cases, Path(args.out))
    print(f"Wrote {len(cases)} cases to {args.out}")


def cmd_process(args) -> None:
    from .pipeline import process_file

    record = process_file(Path(args.file), LiteLLM(_model(args)), whisper_model=args.whisper_model)
    print(json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False))
    if record.review_flags:
        print(f"\n{len(record.review_flags)} item(s) need review:", file=sys.stderr)
        for flag in record.review_flags:
            print(f"  {flag.path:<18} {flag.reason:<12} {flag.detail}", file=sys.stderr)


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate
    from .synth import read_jsonl

    cases = read_jsonl(Path(args.dataset))
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]
    summary = evaluate(cases, LiteLLM(_model(args)), input_mode=args.input, audio_dir=Path(args.audio_dir),
                       whisper_model=args.whisper_model)
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # German umlauts on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="care", description="Voice-to-care-record assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_args(p):
        p.add_argument("--model", help=f"litellm model string (default: {DEFAULT_API_MODEL})")
        p.add_argument("--local", action="store_true", help=f"use the local model ({DEFAULT_LOCAL_MODEL})")
        p.add_argument("--whisper-model", help="faster-whisper model size for audio input")

    p = sub.add_parser("generate", help="generate the synthetic dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/synthetic/cases.jsonl")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("process", help="turn one audio or text note into a care record")
    p.add_argument("file")
    add_model_args(p)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("evaluate", help="evaluate the pipeline on a labelled dataset")
    p.add_argument("dataset")
    add_model_args(p)
    p.add_argument("--input", choices=["text", "audio"], default="text")
    p.add_argument("--audio-dir", default="data/audio")
    p.add_argument("--category", help="only evaluate one case category")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
