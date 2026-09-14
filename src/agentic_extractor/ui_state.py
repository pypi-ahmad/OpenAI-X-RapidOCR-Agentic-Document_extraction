"""UI state vocabulary kept independent from Streamlit.

Responsible for: declaring framework-agnostic session lifecycle states
(`AppState`) and download filename formatting helpers (`output_file_name`).

Must not: import Streamlit or access `st.session_state`; remains pure Python.

Next: `streamlit_app.py`, which binds this state vocabulary to UI rendering.
"""

from enum import StrEnum
from pathlib import Path


def output_file_name(source_name: object, suffix: str) -> str:
    """Build a download name from the original file stem and artifact suffix."""
    stem = Path(str(source_name or "document")).stem.strip() or "document"
    return f"{stem}{suffix}"


class AppState(StrEnum):
    IDLE = "Idle"
    VALIDATING = "Validating"
    READY = "Ready"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    PARTIAL_FAILURE = "Partial failure"
    FAILED = "Failed"
