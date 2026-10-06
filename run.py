"""Start the platform and open the interface in a browser.

Prefer the launcher scripts: run.bat on Windows, ./run.sh elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent

# Allow running straight from a checkout, without installing the package first.
sys.path.insert(0, str(REPO_ROOT / "src"))

LAUNCHER = "run.bat" if os.name == "nt" else "./run.sh"
VENV_PYTHON = (
    REPO_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    / ("python.exe" if os.name == "nt" else "python")
)


def _report_missing_dependency(missing: str) -> int:
    """Explain how to get a working environment instead of showing a traceback."""
    lines = ["", f"  Missing dependency: {missing}", ""]

    if VENV_PYTHON.exists():
        lines += [
            f"  You ran {sys.executable},",
            "  but this project's dependencies live in its own environment.",
            "",
            "  Either use the launcher:",
            "",
            f"      {LAUNCHER}",
            "",
            "  or run this file with the environment's Python:",
            "",
            f'      "{VENV_PYTHON}" run.py',
            "",
        ]
    else:
        lines += [
            f"  This Python has no dependencies installed: {sys.executable}",
            "",
            "  Use the launcher. It creates the environment, installs everything",
            "  and starts the platform:",
            "",
            f"      {LAUNCHER}",
            "",
        ]

    print("\n".join(lines), file=sys.stderr)
    return 1


def main() -> int:
    """Run the server with the browser interface enabled."""
    try:
        from app.core.config import get_settings
        from app.core.logging import configure_logging
    except ModuleNotFoundError as exc:
        return _report_missing_dependency(exc.name or str(exc))

    configure_logging()

    # The browser is opened from the startup hook, once ready to serve.
    os.environ["OPEN_BROWSER"] = "true"
    get_settings.cache_clear()
    settings = get_settings()

    import uvicorn

    print(f"\n  Indic Document Intelligence\n  Opening {settings.browser_url}\n")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
