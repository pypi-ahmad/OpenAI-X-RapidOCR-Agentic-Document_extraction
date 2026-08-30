"""UI state vocabulary kept independent from Streamlit."""

from enum import StrEnum


class AppState(StrEnum):
    IDLE = "Idle"
    VALIDATING = "Validating"
    READY = "Ready"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    PARTIAL_FAILURE = "Partial failure"
    FAILED = "Failed"
