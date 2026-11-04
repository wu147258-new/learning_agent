#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.build" "$ROOT/dist"

# The ordered chapter sources are the canonical manuscript.
# Chapter numbering comes from LaTeX, never from the file names.
shopt -s nullglob
CHAPTERS=("$ROOT/chapters/00_front.md")
for path in "$ROOT"/chapters/beginner/[0-9][0-9]_*.md; do
  CHAPTERS+=("$path")
done
for path in "$ROOT"/chapters/[0-9][0-9]_*.md; do
  [[ "$(basename "$path")" == "00_front.md" ]] && continue
  CHAPTERS+=("$path")
done
CHAPTERS+=("$ROOT/chapters/appendix.md")

: > "$ROOT/book/source.md"
for path in "${CHAPTERS[@]}"; do
  [[ -f "$path" ]] || { echo "Missing manuscript source: $path" >&2; exit 1; }
  cat "$path" >> "$ROOT/book/source.md"
  printf '\n\n' >> "$ROOT/book/source.md"
done
echo "Assembled ${#CHAPTERS[@]} manuscript sources"

pandoc "$ROOT/book/source.md" --from=markdown+raw_tex --standalone --to=latex \
  --number-sections --wrap=auto --highlight-style=tango \
  -V documentclass=ctexbook -V classoption=openany,oneside -V fontsize=11pt \
  -H "$ROOT/book/header.tex" -o "$ROOT/.build/agent-textbook.tex"

for pass in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error \
    -output-directory="$ROOT/.build" "$ROOT/.build/agent-textbook.tex" \
    > "$ROOT/.build/latex.log" 2>&1 || {
      tail -100 "$ROOT/.build/latex.log"
      exit 1
    }
done
cp "$ROOT/.build/agent-textbook.pdf" "$ROOT/dist/Agent_Algorithm_Textbook_2026.pdf"
echo "Built $ROOT/dist/Agent_Algorithm_Textbook_2026.pdf"
