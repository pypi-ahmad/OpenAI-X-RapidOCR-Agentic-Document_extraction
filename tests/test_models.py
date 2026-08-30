import pytest
from pydantic import ValidationError

from agentic_extractor.models import Capability, DocumentRequest, ProcessingMode


def test_classify_requires_classes() -> None:
    with pytest.raises(ValidationError, match="allowed class"):
        DocumentRequest(file_name="a.pdf", file_bytes=b"x", capabilities={Capability.CLASSIFY})


def test_extract_requires_schema() -> None:
    with pytest.raises(ValidationError, match="JSON Schema"):
        DocumentRequest(file_name="a.pdf", file_bytes=b"x", capabilities={Capability.EXTRACT})


def test_split_override_requires_an_audit_reason() -> None:
    with pytest.raises(ValidationError, match="split override reason"):
        DocumentRequest(
            file_name="packet.pdf",
            file_bytes=b"pdf",
            capabilities={Capability.PARSE, Capability.SPLIT},
            split_boundaries=[3],
        )


def test_document_request_has_no_cloud_consent_contract() -> None:
    request = DocumentRequest(file_name="a.pdf", file_bytes=b"x")
    assert "cloud_consent" not in type(request).model_fields


def test_only_dual_engine_modes_are_exposed() -> None:
    assert [mode.value for mode in ProcessingMode] == ["Balanced", "High Accuracy"]
