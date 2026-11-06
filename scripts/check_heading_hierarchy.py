"""Verify three-level manuscript structure and two-level PDF contents.

Run from repository root:
  python scripts/check_heading_hierarchy.py
  python scripts/check_heading_hierarchy.py --pdf dist/Agent_Algorithm_Textbook_2026.pdf

This checks structure and representative TOC pages; it does not replace
full-page visual inspection or mathematical peer review.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
HEADER = ROOT / "book" / "header.tex"
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING = re.compile(r"^(#{1,4})\s+(.+)$")
FORBIDDEN = ("写作原则", "本书借鉴", "全书沿着这一原则")


def check_source():
    header = HEADER.read_text(encoding="utf-8")
    if not re.search(r"\\setcounter\{tocdepth\}\{1\}", header):
        raise AssertionError("TOC must stop at chapter/section level")
    if not re.search(r"\\setcounter\{secnumdepth\}\{2\}", header):
        raise AssertionError("Headings must number chapter/section/subsection")

    files = sorted(CHAPTERS.glob("[0-9][0-9]_*.md"))
    files += sorted((CHAPTERS / "beginner").glob("[0-9][0-9]_*.md"))
    files.append(CHAPTERS / "appendix.md")
    if not files or not all(file.is_file() for file in files):
        raise AssertionError("manuscript chapter files missing")

    chapters = sections = subsections = 0
    for file in files:
        raw = file.read_text(encoding="utf-8")
        if any(marker in raw for marker in FORBIDDEN):
            raise AssertionError(f"Editorial manuscript text in {file}")
        chapter_active = section_active = False
        fence_marker = ""
        local_levels = [0, 0, 0]
        for lineno, line in enumerate(raw.splitlines(), start=1):
            match = FENCE.match(line)
            if match:
                marker = match.group(1)
                if not fence_marker:
                    fence_marker = marker[0]
                elif fence_marker == marker[0]:
                    fence_marker = ""
                continue
            if fence_marker:
                continue
            # Pandoc does not recognize a display equation delimited by
            # solitary dollar-sign lines; it escapes LaTeX control sequences
            # and XeLaTeX then fails much later with a misleading math error.
            if line.strip() == "$":
                raise AssertionError(
                    f"Standalone $ delimiter in {file}:{lineno}; use $ or \\[...\\]"
                )
            match = HEADING.match(line)
            if not match:
                continue
            level = len(match.group(1))
            if level > 3:
                raise AssertionError(f"Unexpected fourth heading depth in {file}")
            local_levels[level-1] += 1
            if level == 1:
                chapters += 1
                chapter_active = True
                section_active = False
            elif level == 2:
                if not chapter_active:
                    raise AssertionError(f"Section without chapter in {file}")
                sections += 1
                section_active = True
            elif level == 3:
                if not section_active:
                    raise AssertionError(f"Subsection without section in {file}")
                subsections += 1
        if file.parent.name == "beginner" and (
            local_levels[0] != 1 or local_levels[1] < 3 or local_levels[2] < 3
        ):
            raise AssertionError(f"Beginner chapter requires chapter/section/subsection: {file}")
    if chapters < 23 or sections < chapters or subsections < 20:
        raise AssertionError("Heading hierarchy unexpectedly small")
    print(f"PASS structure: {chapters} chapters, {sections} sections, {subsections} subsections")


def check_pdf(pdf_path):
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)", info, re.M)
    if not match or int(match.group(1)) < 50:
        raise AssertionError("PDF missing or implausibly short")
    output = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    pages = output.split("\f")
    start = None
    for i, page in enumerate(pages):
        if re.search(r"^\s*目录\s*$", page, flags=re.M):
            start = i
            break
    if start is None:
        raise AssertionError("Rendered PDF has no visible TOC")
    # The frontmatter uses roman-numbered pages, while body text uses Arabic
    # page numbers. Stop BEFORE the first body page; scanning the entire
    # manuscript would mistake correctly numbered body subsections for TOC.
    toc_pages = []
    for page in pages[start:]:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        if not lines:
            continue
        footer = lines[-1]
        if toc_pages and re.fullmatch(r"[0-9]+", footer):
            break
        toc_pages.append(page)
    if not toc_pages:
        raise AssertionError("Could not isolate roman-numbered TOC pages")
    toc_text = "\n".join(toc_pages)
    if not re.search(r"\d+\.\d+\s+", toc_text):
        raise AssertionError("TOC appears to omit sections")
    if re.search(r"^\s*\d+\.\d+\.\d+\s+", toc_text, flags=re.M):
        raise AssertionError("Subsection was shown in two-level TOC")
    print(f"PASS PDF: {match.group(1)} pages; chapter/section TOC only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    check_source()
    if args.pdf:
        check_pdf(args.pdf)


if __name__ == "__main__":
    main()
