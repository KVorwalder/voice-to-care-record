"""Command line interface.

    care generate [--seed 42] [--out data/synthetic/cases.jsonl]
    care process <audio_or_text_file> [--model ...]
    care evaluate <dataset.jsonl> [--model ...] [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

from .llm import DEFAULT_MODEL, LiteLLM


def cmd_generate(args) -> None:
    """`care generate`: create the synthetic dataset and write it as JSON lines.

    Input:  args.seed (random seed), args.out (output file)
    Output: none (writes the file, prints how many cases were written)
    """
    from .synth import generate_dataset, write_jsonl

    cases = generate_dataset(args.seed)
    write_jsonl(cases, Path(args.out))
    print(f"Wrote {len(cases)} cases to {args.out}")


def cmd_process(args) -> None:
    """`care process`: turn one audio or text note into a care record.

    Input:  args.file (audio or text file), args.model (litellm model string)
    Output: none (prints the record as JSON; review flags go to stderr)
    """
    from .pipeline import process_file

    record = process_file(Path(args.file), LiteLLM(args.model))
    print(json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False))
    # List the review flags separately, so they are easy to spot
    if record.review_flags:
        print(f"\n{len(record.review_flags)} item(s) need review:", file=sys.stderr)
        for flag in record.review_flags:
            print(f"  {flag.path:<18} {flag.reason:<12} {flag.detail}", file=sys.stderr)


def cmd_evaluate(args) -> None:
    """`care evaluate`: run and score the pipeline on a labelled dataset.

    Input:  args.dataset (JSON lines file), args.model, args.limit (optional: first N cases)
    Output: none (writes results/, prints the summary as JSON)
    """
    from .evaluate import evaluate
    from .synth import read_jsonl

    cases = read_jsonl(Path(args.dataset))
    if args.limit:
        cases = cases[: args.limit]
    # The file name without extension ("cases_test") identifies the dataset in summary.csv
    summary = evaluate(cases, LiteLLM(args.model), dataset=Path(args.dataset).stem)
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    """Entry point of the `care` command: parse the arguments and run the chosen subcommand.

    Input:  argv - command line arguments (None = use sys.argv)
    Output: none
    """
    if hasattr(sys.stdout, "reconfigure"):  # German umlauts on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="care", description="Voice-to-care-record assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    # Each subcommand gets its own arguments and the function that runs it (set_defaults)
    p = sub.add_parser("generate", help="generate the synthetic dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/synthetic/cases.jsonl")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("process", help="turn one audio or text note into a care record")
    p.add_argument("file")
    p.add_argument("--model", default=DEFAULT_MODEL, help="litellm model string")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("evaluate", help="evaluate the pipeline on a labelled dataset")
    p.add_argument("dataset")
    p.add_argument("--model", default=DEFAULT_MODEL, help="litellm model string")
    p.add_argument("--limit", type=int, help="only evaluate the first N cases")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
