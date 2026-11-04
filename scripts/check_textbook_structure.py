"""Check textbook source inclusion and numbered headings; optionally inspect PDF.

This is a structural smoke test, NOT proof of mathematical correctness.
No third-party Python libraries. PDF mode requires poppler-utils.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
AUTONUMBERED_HEADING = re.compile(r"^#{2,6}\s+\d+(?:\.\d+)+\s+", re.MULTILINE)
DUPLICATED_PDF_HEADING = re.compile(
    r"^\s*\d+\.\d+(?:\.\d+)?\s+\d+\.\d+(?:\.\d+)?\s+",
    re.MULTILINE
)


def validate_chapters():
    script = (ROOT / "scripts" / "build_pdf.sh").read_text(encoding="utf-8")
    match = re.search(r"CHAPTERS=\((.*?)\)", script, re.DOTALL)
    if match is None:
        raise ValueError("No CHAPTERS array in build_pdf.sh")
    names = re.findall(r'"([^"]+\.md)"', match.group(1))
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicated CHAPTERS entries")
    if names[-1] != "appendix.md":
        raise ValueError("Appendix must appear after all chapters")
    actual = {p.name for p in (ROOT / "chapters").glob("*.md")}
    if actual != set(names):
        raise ValueError(
            f"CHAPTERS mismatch: omitted={sorted(actual-set(names))}, "
            f"missing={sorted(set(names)-actual)}"
        )
    for name in names:
        body = (ROOT / "chapters" / name).read_text(encoding="utf-8")
        match = AUTONUMBERED_HEADING.search(body)
        if match:
            line_no = body.count("\n", 0, match.start()) + 1
            raise ValueError(
                f"Duplicate number risk in chapters/{name}:{line_no}: "
                f"{match.group(0).strip()}"
            )
    print(f"PASS chapter inventory and section numbering: {len(names)} source files")


def validate_pdf(path: Path):
    if not path.is_file():
        raise ValueError(f"PDF does not exist: {path}")
    info = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"^Pages:\s*(\d+)", info, re.MULTILINE)
    if not match or int(match.group(1)) < 30:
        raise ValueError("Missing or unexpectedly short textbook PDF")
    body = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)
    duplicate = DUPLICATED_PDF_HEADING.search(body)
    if duplicate:
        raise ValueError(f"Duplicate PDF section numbering: {duplicate.group(0)!r}")
    for text in ("Transformer", "马尔可夫链", "数学期望"):
        if text not in body:
            raise ValueError(f"PDF text missing expected concept: {text}")
    print(f"PASS PDF readable text and numbering: {match.group(1)} pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=None)
    args = parser.parse_args()
    try:
        validate_chapters()
        if args.pdf is not None:
            validate_pdf(args.pdf)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
