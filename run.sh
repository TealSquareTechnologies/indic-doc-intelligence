#!/usr/bin/env bash
#
# Launcher for Linux and macOS. Sets everything up and starts the platform.
#
#   ./run.sh                 set everything up and start
#   ./run.sh --skip-search   skip the large question answering dependencies
#   ./run.sh --skip-ocr      skip the recognition engine and language data
#   ./run.sh --reinstall     force dependencies to be installed again

set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"
VENV_PY=""
SKIP_SEARCH=0
SKIP_OCR=0
REINSTALL=0

# Virtual environments use bin/ on Unix and Scripts/ on Windows.
resolve_venv_python() {
    for candidate in "${VENV_DIR}/bin/python" "${VENV_DIR}/Scripts/python.exe"; do
        if [ -x "$candidate" ]; then
            VENV_PY="$candidate"
            return 0
        fi
    done
    VENV_PY=""
    return 1
}

usage() {
    cat <<'TEXT'

  Usage: ./run.sh [options]

    --skip-search   Do not install the large embedding dependencies
    --skip-ocr      Do not install the recognition engine or language data
    --reinstall     Install dependencies again even if present
    --help          Show this message

TEXT
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-search) SKIP_SEARCH=1 ;;
        --skip-ocr)    SKIP_OCR=1 ;;
        --reinstall)   REINSTALL=1 ;;
        -h|--help)     usage; exit 0 ;;
        *)
            echo "  Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

printf '\n  Indic Document Intelligence\n  ===========================\n\n'

# ------------------------------------------------------------------ interpreter
echo "  [1/6] Locating Python..."
BOOT_PY=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' >/dev/null 2>&1; then
        BOOT_PY="$candidate"
        break
    fi
done

if [ -z "$BOOT_PY" ]; then
    cat <<'TEXT'

  Python 3.11 or newer was not found.

    Debian or Ubuntu:  sudo apt-get install python3 python3-venv
    macOS:             brew install python

TEXT
    exit 1
fi
echo "        using $BOOT_PY ($("$BOOT_PY" -V 2>&1))"

# ------------------------------------------------------------------ environment
echo "  [2/6] Preparing the virtual environment..."
if resolve_venv_python; then
    echo "        already present"
else
    echo "        creating $VENV_DIR"
    if ! "$BOOT_PY" -m venv "$VENV_DIR"; then
        cat <<'TEXT'

  Could not create the virtual environment.
  On Debian or Ubuntu the venv module ships separately:

    sudo apt-get install python3-venv

TEXT
        exit 1
    fi
    if ! resolve_venv_python; then
        printf '\n  The virtual environment was created but has no usable Python.\n\n'
        exit 1
    fi
    REINSTALL=1
fi

# ------------------------------------------------------------ core dependencies
echo "  [3/6] Checking dependencies..."
need_core=$REINSTALL
if [ "$need_core" -eq 0 ]; then
    if ! "$VENV_PY" -c \
        'import fastapi, uvicorn, pydantic_settings, cv2, numpy, openai, pytesseract, multipart, pypdf, pypdfium2' \
        >/dev/null 2>&1; then
        need_core=1
    fi
fi

if [ "$need_core" -eq 1 ]; then
    printf '        installing core packages, this takes a minute...\n\n'
    "$VENV_PY" -m pip install --upgrade pip --quiet --disable-pip-version-check
    if ! "$VENV_PY" -m pip install -e ".[dev]" --disable-pip-version-check; then
        cat <<'TEXT'

  Installing dependencies failed. The reason from pip is above.
  The usual cause is no network connection.

TEXT
        exit 1
    fi
    echo
else
    echo "        core packages present"
fi

# ------------------------------------------------------------- text recognition
echo "  [4/6] Checking text recognition..."
if [ "$SKIP_OCR" -eq 1 ]; then
    echo "        skipped"
elif command -v tesseract >/dev/null 2>&1; then
    echo "        engine ready"
    "$VENV_PY" -m app.ocr.tessdata || true
else
    echo "        Tesseract is not installed. Installing it now."
    installed=0
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y -qq tesseract-ocr && installed=1
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y -q tesseract && installed=1
    elif command -v brew >/dev/null 2>&1; then
        brew install tesseract && installed=1
    fi

    if [ "$installed" -eq 1 ] && command -v tesseract >/dev/null 2>&1; then
        echo "        engine ready"
    else
        cat <<'TEXT'

        Tesseract could not be installed automatically. Install it with your
        package manager, for example:

          Debian or Ubuntu:  sudo apt-get install tesseract-ocr
          Fedora:            sudo dnf install tesseract
          macOS:             brew install tesseract

        Everything else still works; recognition will be unavailable.

TEXT
    fi
    "$VENV_PY" -m app.ocr.tessdata || true
fi

# ---------------------------------------------------------- search dependencies
echo "  [5/6] Checking question answering support..."
if [ "$SKIP_SEARCH" -eq 1 ]; then
    echo "        skipped, search will report itself unavailable"
elif "$VENV_PY" -c 'import sentence_transformers' >/dev/null 2>&1; then
    echo "        ready"
else
    echo "        installing the embedding stack."
    echo "        This is a large one-time download and can take several minutes."
    echo "        Press Ctrl+C to skip it; everything else will still work,"
    printf '        or use "./run.sh --skip-search" next time.\n\n'
    "$VENV_PY" -m pip install -e ".[search]" --disable-pip-version-check ||
        printf '\n        could not install it, continuing without search\n'
    echo
fi

# -------------------------------------------------------------- configuration
echo "  [6/6] Checking configuration..."
if [ ! -f ".env" ]; then
    cp ".env.example" ".env"
    cat <<'TEXT'

        Created .env from the example file.
        Open it and set CONSCIOUSNX_API_KEY before correction,
        translation and question answering will work.

TEXT
else
    echo "        .env present"
fi

printf '\n  Starting. The interface will open in your browser.\n  Press Ctrl+C to stop.\n\n'

exec "$VENV_PY" run.py
