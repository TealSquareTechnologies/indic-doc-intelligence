"""Language data management.

Run directly to fetch whatever is missing:

    python -m app.ocr.tessdata
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from app.core.config import REPO_ROOT, get_settings
from app.core.logging import get_logger
from app.ocr.languages import LANGUAGES, supported_codes

logger = get_logger(__name__)

# Data sets, smallest first.
VARIANTS = {
    "fast": "tessdata_fast",
    "standard": "tessdata",
    "best": "tessdata_best",
}
DEFAULT_VARIANT = "fast"

BASE_URL = "https://raw.githubusercontent.com/tesseract-ocr/{repository}/main/{code}.traineddata"
DOWNLOAD_TIMEOUT_SECONDS = 180

# Orientation and script detection data, used by Tesseract for page analysis.
EXTRA_CODES = ("osd",)


def tessdata_dir() -> Path:
    """The directory holding this project's traineddata files."""
    configured = get_settings().tessdata_dir.strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else REPO_ROOT / path
    return REPO_ROOT / "tessdata"


def installed_codes(directory: Path | None = None) -> set[str]:
    """The language codes already present in the directory."""
    directory = directory or tessdata_dir()
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob("*.traineddata")}


def missing_codes(codes: list[str] | None = None, directory: Path | None = None) -> list[str]:
    """Which of the requested codes are not present yet."""
    wanted = list(codes or supported_codes()) + list(EXTRA_CODES)
    present = installed_codes(directory)
    return [code for code in dict.fromkeys(wanted) if code not in present]


def download_language(code: str, directory: Path, variant: str = DEFAULT_VARIANT) -> int:
    """Fetch one traineddata file. Returns the number of bytes written."""
    repository = VARIANTS.get(variant, VARIANTS[DEFAULT_VARIANT])
    url = BASE_URL.format(repository=repository, code=code)
    directory.mkdir(parents=True, exist_ok=True)

    target = directory / f"{code}.traineddata"
    partial = target.with_suffix(".partial")

    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"could not download {code}.traineddata: {exc}") from exc

    if not payload:
        raise RuntimeError(f"{code}.traineddata came back empty")

    partial.write_bytes(payload)
    partial.replace(target)
    return len(payload)


def configure_environment() -> Path | None:
    """Point Tesseract at the project's language data, if there is any."""
    directory = tessdata_dir()
    if not any(directory.glob("*.traineddata")):
        return None

    os.environ["TESSDATA_PREFIX"] = str(directory)
    return directory


def ensure(
    codes: list[str] | None = None, variant: str = DEFAULT_VARIANT
) -> tuple[list[str], list[str]]:
    """Download any missing language data. Returns the codes fetched and those that failed."""
    directory = tessdata_dir()
    pending = missing_codes(codes, directory)
    if not pending:
        return [], []

    fetched: list[str] = []
    failed: list[str] = []

    for code in pending:
        try:
            size = download_language(code, directory, variant)
        except RuntimeError as exc:
            logger.warning("%s", exc)
            failed.append(code)
            continue
        logger.info("Fetched %s.traineddata (%.1f MB)", code, size / (1024 * 1024))
        fetched.append(code)

    return fetched, failed


def _describe(code: str) -> str:
    for language in LANGUAGES:
        if language.code == code:
            return language.name
    return code


def main(argv: list[str] | None = None) -> int:
    """Fetch missing language data, reporting progress as it goes."""
    parser = argparse.ArgumentParser(
        prog="python -m app.ocr.tessdata",
        description="Download Tesseract language data into the project.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default=DEFAULT_VARIANT,
        help=f"Which upstream data set to use (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--languages",
        help="Comma separated codes to fetch. Defaults to every supported language.",
    )
    parser.add_argument(
        "--list", action="store_true", help="Show what is present and what is missing"
    )
    args = parser.parse_args(argv)

    directory = tessdata_dir()
    codes = (
        [part.strip() for part in args.languages.split(",") if part.strip()]
        if args.languages
        else None
    )

    if args.list:
        present = sorted(installed_codes(directory))
        pending = missing_codes(codes, directory)
        print(f"  language data directory: {directory}")
        print(f"  present: {', '.join(present) if present else 'none'}")
        print(f"  missing: {', '.join(pending) if pending else 'none'}")
        return 0

    pending = missing_codes(codes, directory)
    if not pending:
        print("        language data complete")
        return 0

    print(f"        fetching {len(pending)} language files into {directory.name}/")
    fetched: list[str] = []
    failed: list[str] = []

    for code in pending:
        label = _describe(code)
        try:
            size = download_language(code, directory, args.variant)
        except RuntimeError as exc:
            print(f"          {label} ({code}) failed: {exc}", file=sys.stderr)
            failed.append(code)
            continue
        print(f"          {label} ({code}) {size / (1024 * 1024):.1f} MB")
        fetched.append(code)

    if failed:
        print(f"        could not fetch: {', '.join(failed)}", file=sys.stderr)
        return 1 if not fetched else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
