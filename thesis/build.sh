#!/usr/bin/env bash
# Compile the thesis. minted needs -shell-escape + pygmentize (lives in ../.venv/bin).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/chapters
PATH="$(cd ../.venv/bin && pwd):$PATH" latexmk -pdf -shell-escape -outdir=build "$@" main.tex
if [[ -f build/main.pdf ]]; then
  cp build/main.pdf main.pdf
fi
