"""Launch the Streamlit application through the installed script.

This is the `agentic-extractor` console-script entry point (see
`[project.scripts]` in pyproject.toml). It only resolves `app.py`'s path and
hands off to Streamlit's own CLI; it must not contain any extraction logic
itself. Next: `app.py` for what actually runs."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> None:
    app = Path(__file__).resolve().parents[2] / "app.py"
    sys.argv = ["streamlit", "run", str(app)]
    raise SystemExit(stcli.main())
