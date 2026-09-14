"""Run real all-page validation for every PDF with a paired ground truth.

Responsible for: CLI batch validation running `tools/run_real_validation.py`
across an entire directory of PDFs against ground-truth Parse JSON and Markdown
pairs, with optional `--resume` checkpointing.

Must not: fabricate ground-truth files or skip uncompleted documents without `--resume`.

Next: `tools/run_real_validation.py`, which executes single-document extraction
and evaluation comparison.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("ground_truths", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("Balanced", "High Accuracy"), default="Balanced")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed document outputs and retry an empty failed document directory.",
    )
    return parser


def _paired_documents(sources: Path, ground_truths: Path) -> list[tuple[Path, Path, Path]]:
    pairs: list[tuple[Path, Path, Path]] = []
    for source in sorted(sources.glob("*.pdf")):
        oracle = ground_truths / f"{source.stem}.parse.json"
        markdown = ground_truths / f"{source.stem}.parse.md"
        missing = [path for path in (oracle, markdown) if not path.is_file()]
        if missing:
            names = ", ".join(path.name for path in missing)
            raise FileNotFoundError(f"Missing ground truth for {source.name}: {names}")
        pairs.append((source, oracle, markdown))
    if not pairs:
        raise FileNotFoundError(f"No PDF files found in {sources}")
    return pairs


def _use_source_filenames(output: Path, stem: str) -> None:
    names = {
        "document.md": f"{stem}.parse.md",
        "parse-result.json": f"{stem}.parse.json",
        "manifest.json": f"{stem}.manifest.json",
        "landingai-comparison.json": f"{stem}.comparison.json",
        "run-summary.json": f"{stem}.run-summary.json",
    }
    for current, renamed in names.items():
        output.joinpath(current).replace(output / renamed)


def _completed_output(output: Path, stem: str) -> bool:
    return all(
        (output / f"{stem}.{suffix}").is_file()
        for suffix in (
            "parse.md",
            "parse.json",
            "manifest.json",
            "comparison.json",
            "run-summary.json",
        )
    )


def main() -> int:
    args = _parser().parse_args()
    pairs = _paired_documents(args.sources, args.ground_truths)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    validator = Path(__file__).with_name("run_real_validation.py")
    for source, oracle, _ in pairs:
        page_count = len(PdfReader(source).pages)
        output = args.output / source.stem
        if args.resume and _completed_output(output, source.stem):
            print(f"REAL_RUN_SKIP {source.name} reason=completed", flush=True)
            continue
        if output.is_dir():
            output.rmdir()
        print(f"REAL_RUN_START {source.name} pages=1-{page_count}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(validator),
                str(source),
                str(oracle),
                str(output),
                "--mode",
                args.mode,
                "--pages",
                *(str(page) for page in range(1, page_count + 1)),
            ],
            check=True,
        )
        _use_source_filenames(output, source.stem)
        print(f"REAL_RUN_DONE {source.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
