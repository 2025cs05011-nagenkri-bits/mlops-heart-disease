#!/bin/bash
# download_data.sh
#
# Downloads the UCI Heart Disease dataset from the upstream archive into
# Raw_data/heart+disease/. The dataset is already committed to the repo,
# so this script is only needed if the directory is missing or you want a
# clean re-fetch from source.
#
# Usage:
#   ./scripts/download_data.sh           # download if missing
#   ./scripts/download_data.sh --force   # re-download even if present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$APP_DIR/Raw_data/heart+disease"
URL="https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

if [ -d "$RAW_DIR" ] && [ -f "$RAW_DIR/cleveland.data" ] && [ "$FORCE" -eq 0 ]; then
  echo "[+] Raw_data/heart+disease/cleveland.data already present - nothing to do."
  echo "    Use --force to re-download from $URL"
  exit 0
fi

mkdir -p "$APP_DIR/Raw_data"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[*] Downloading UCI Heart Disease archive..."
curl -fSL -o "$TMP/heart-disease.zip" "$URL"

echo "[*] Extracting to $RAW_DIR ..."
rm -rf "$RAW_DIR"
mkdir -p "$RAW_DIR"
unzip -q -o "$TMP/heart-disease.zip" -d "$RAW_DIR"

echo "[+] Done. Files:"
ls "$RAW_DIR" | sed 's/^/    /'

echo
echo "Next step - regenerate the cleaned 80/20 split:"
echo "    cd $APP_DIR && python -m src.data.prepare"
