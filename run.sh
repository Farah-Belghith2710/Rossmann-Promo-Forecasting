#!/usr/bin/env bash
# One-command run on Mac or Linux. The first time, it sets up Python for the project (a few minutes).
# Extra options are passed on, e.g.:  ./run.sh --sample
set -e
cd "$(dirname "$0")"

if [ -z "$JAVA_HOME" ] && ! command -v java >/dev/null; then
  echo "Java was not found. PySpark needs Java 17 or 21: install it from https://adoptium.net (Temurin)."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Setting up Python for this project, the first time only. This takes a few minutes..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { rm -rf .venv; exit 1; }
fi

.venv/bin/python -m rossmann.pipeline "$@"
