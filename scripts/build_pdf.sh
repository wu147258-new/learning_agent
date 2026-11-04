#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.build" "$ROOT/dist"
pandoc "$ROOT/book/source.md" --from=markdown+raw_tex --standalone --to=latex --number-sections --wrap=auto --highlight-style=tango -V documentclass=ctexbook -V classoption=openany,oneside -V fontsize=11pt -H "$ROOT/book/header.tex" -o "$ROOT/.build/agent-textbook.tex"
for i in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error -output-directory="$ROOT/.build" "$ROOT/.build/agent-textbook.tex" > "$ROOT/.build/latex.log" 2>&1 || { tail -80 "$ROOT/.build/latex.log"; exit 1; }
done
cp "$ROOT/.build/agent-textbook.pdf" "$ROOT/dist/Agent_Algorithm_Textbook_2026.pdf"
echo "Built $ROOT/dist/Agent_Algorithm_Textbook_2026.pdf"
