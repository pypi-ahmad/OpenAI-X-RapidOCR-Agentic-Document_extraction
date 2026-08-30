"""Launch the Streamlit application through the installed script."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> None:
    app = Path(__file__).resolve().parents[2] / "app.py"
    sys.argv = ["streamlit", "run", str(app)]
    raise SystemExit(stcli.main())
