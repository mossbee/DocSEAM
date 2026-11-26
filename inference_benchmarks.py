"""Run DocSEAM on benchmark manifests and save predictions or submission JSON."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from infer import answer, load_model


BENCHMARKS = {
    "dude": ("dude_test.jsonl", 512),
    "mp_docvqa": ("mp_docvqa_test.jsonl", 256),
    "slidevqa": ("slidevqa_test.jsonl", 256),
    "longdocurl": ("longdocurl.jsonl", 2048),
    "mmlongbench_doc": ("mmlongbench_doc.jsonl", 2048),
}


def submission_record(sample, prediction, benchmark):
    if benchmark == "mp_docvqa":
        return {"questionId": sample["questionId"], "answer": prediction, "answer_page": ""}
    unanswerable_phrases = (
        "unanswerable", "does not contain the answer", "cannot be answered",
        "do not contain sufficient information",
    )
    answers = [] if any(phrase in prediction for phrase in unanswerable_phrases) else re.split(
        r"(?<!\n)\n(?!\n)", prediction
    )
    return {"questionId": sample["questionId"], "answer": answers, "answer_confidence": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mossbee/DocSEAM")
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--image-root", required=True, type=Path,
                        help="Directory containing the benchmark image directories")
    parser.add_argument("--output-file", type=Path,
                        help="Defaults to results/<benchmark>.json for DUDE/MP-DocVQA, otherwise .jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-samples", type=int, help="Optional limit on the number of questions")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    filename, default_tokens = BENCHMARKS[args.benchmark]
    test_file = Path(__file__).parent / "benchmarks" / args.benchmark / filename
    is_submission = args.benchmark in ("dude", "mp_docvqa")
    extension = ".json" if is_submission else ".jsonl"
    output_file = args.output_file or Path("results") / (args.benchmark + extension)
    if output_file.resolve() == test_file.resolve():
        parser.error("--output-file must differ from the benchmark manifest")

    model, processor = load_model(args.model, args.device)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with test_file.open(encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_file.parent, delete=False
        ) as output:
            temporary_path = Path(output.name)
            submissions = []
            count = 0
            for line in source:
                if not line.strip():
                    continue
                sample = json.loads(line)
                images = sample.get("images", sample.get("image", []))
                if isinstance(images, str):
                    images = [images]
                paths = [args.image_root / path for path in images]
                prediction = answer(
                    model, processor, paths, sample["user"],
                    args.max_new_tokens or default_tokens,
                )
                if is_submission:
                    submissions.append(submission_record(sample, prediction, args.benchmark))
                else:
                    output.write(json.dumps({**sample, "qwen_answer": prediction}, ensure_ascii=False) + "\n")
                count += 1
                print(f"Finished sample {count}", flush=True)
                if args.max_samples is not None and count >= args.max_samples:
                    break
            if is_submission:
                json.dump(submissions, output, ensure_ascii=False, indent=2)
                output.write("\n")
        os.replace(temporary_path, output_file)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"Saved {count} predictions to {output_file}")


if __name__ == "__main__":
    main()
