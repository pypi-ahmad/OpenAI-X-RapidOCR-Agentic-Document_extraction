from __future__ import annotations

import base64
import io
import json
import zipfile
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from agentic_extractor.api import create_app
from agentic_extractor.models import UsageRecord
from agentic_extractor.ocr import OCRResource
from agentic_extractor.openai_refiner import (
    CloudEvidence,
    CloudResult,
    ExtractedField,
    MarkdownWorkflowResult,
    OpenAIConfigurationError,
    OpenAIRefiner,
)


class FakeRefiner:
    def validate_configuration(self) -> None:
        return None

    def refine(self, *args, **kwargs):
        return CloudResult(refined_markdown="GPT reviewed", reviewed_pages=[1]), UsageRecord(
            call_count=1
        )


def _png() -> str:
    output = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(output, "PNG")
    return base64.b64encode(output.getvalue()).decode()


def test_api_validates_submit_and_exposes_opaque_job_status() -> None:
    client = TestClient(create_app(refiner=FakeRefiner()))
    bad = client.post("/api/v1/jobs/parse", json={"file_name": "x.png", "content_base64": "!"})
    assert bad.status_code == 422
    response = client.post(
        "/api/v1/jobs/parse",
        json={
            "file_name": "x.png",
            "content_base64": _png(),
            "mode": "Balanced",
        },
    )
    assert response.status_code == 201
    job_id = response.json()["job_id"]
    assert len(job_id) >= 20
    status = client.get(f"/api/v1/jobs/{job_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["state"] in {"ACCEPTED", "REVIEW_REQUIRED"}
    assert body["result"]["selected_pages"] == [1]
    assert "content_base64" not in status.text


def test_api_unknown_jobs_and_invalid_schema_have_clear_errors() -> None:
    client = TestClient(create_app(refiner=FakeRefiner()))
    assert client.get("/api/v1/jobs/not-a-job").status_code == 404
    created = client.post(
        "/api/v1/jobs/parse",
        json={
            "file_name": "x.png",
            "content_base64": _png(),
            "mode": "Balanced",
        },
    ).json()
    response = client.post(
        f"/api/v1/jobs/{created['job_id']}/extract",
        json={
            "schema_version": "1",
            "schema": {"type": "not-a-json-schema-type"},
        },
    )
    assert response.status_code == 422


def test_api_rejects_invalid_page_numbers_and_exposes_openapi() -> None:
    client = TestClient(create_app(refiner=FakeRefiner()))
    response = client.post(
        "/api/v1/jobs/parse",
        json={
            "file_name": "x.png",
            "content_base64": _png(),
            "selected_pages": [0],
        },
    )
    assert response.status_code == 422
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/api/v1/jobs/{job_id}/extraction" in openapi.json()["paths"]
    schemas = openapi.json()["components"]["schemas"]
    assert "cloud_consent" not in schemas["ParseJobRequest"]["properties"]
    assert "cloud_consent" not in schemas["ExtractionJobRequest"]["properties"]
    extraction = schemas["ExtractionResult"]["properties"]
    assert extraction["fields"]["items"]["$ref"].endswith("/ValidatedField")
    assert extraction["review_items"]["items"]["$ref"].endswith("/ReviewItem")
    assert extraction["checkboxes"]["items"]["$ref"].endswith("/CheckboxRecord")
    paths = openapi.json()["paths"]
    parse_errors = paths["/api/v1/jobs/parse"]["post"]["responses"]
    extract_errors = paths["/api/v1/jobs/{job_id}/extract"]["post"]["responses"]
    assert parse_errors["503"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )
    assert extract_errors["503"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )


def test_api_setup_errors_are_typed_and_openai_precedes_rapidocr(monkeypatch) -> None:
    class InvalidRefiner:
        def validate_configuration(self) -> None:
            raise OpenAIConfigurationError("Configure OPENAI_API_KEY before extraction.")

        def refine(self, *args, **kwargs):
            raise AssertionError("Refinement must not start after failed configuration.")

    rapidocr_calls = 0

    def rapidocr() -> OCRResource:
        nonlocal rapidocr_calls
        rapidocr_calls += 1
        raise AssertionError("RapidOCR must not initialize before OpenAI validation")

    monkeypatch.setattr("agentic_extractor.api.create_rapidocr_engine", rapidocr)
    client = TestClient(create_app(refiner=InvalidRefiner()))
    response = client.post(
        "/api/v1/jobs/parse",
        json={"file_name": "x.png", "content_base64": _png()},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "openai_unavailable",
        "message": "Configure OPENAI_API_KEY before extraction.",
        "action": "Add valid OpenAI configuration and retry.",
        "retryable": True,
    }
    assert rapidocr_calls == 0


def test_api_rapidocr_setup_error_is_actionable(monkeypatch) -> None:
    def unavailable() -> OCRResource:
        raise RuntimeError("RapidOCR import failed")

    monkeypatch.setattr("agentic_extractor.api.create_rapidocr_engine", unavailable)
    client = TestClient(create_app(refiner=FakeRefiner()))
    response = client.post(
        "/api/v1/jobs/parse",
        json={"file_name": "x.png", "content_base64": _png()},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "rapidocr_unavailable"
    assert "uv sync --all-groups" in detail["action"]


def test_api_processing_failure_is_retained_as_typed_job_error() -> None:
    class FailingRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            raise RuntimeError("provider interrupted")

    client = TestClient(create_app(refiner=FailingRefiner()))
    created = client.post(
        "/api/v1/jobs/parse",
        json={"file_name": "x.png", "content_base64": _png()},
    )

    assert created.status_code == 201
    assert created.json()["state"] == "FAILED"
    status = client.get(created.json()["status_url"]).json()
    assert status["error"]["code"] == "processing_failed"
    assert status["error"]["retryable"] is True
    assert "provider interrupted" in status["error"]["message"]
    artifacts = client.get(f"/api/v1/jobs/{created.json()['job_id']}/artifacts")
    assert artifacts.status_code == 409
    assert artifacts.json()["detail"]["code"] == "artifacts_unavailable"


def test_api_expires_process_local_jobs() -> None:
    now = 100.0
    client = TestClient(create_app(refiner=FakeRefiner(), clock=lambda: now))
    created = client.post(
        "/api/v1/jobs/parse",
        json={"file_name": "x.png", "content_base64": _png()},
    ).json()

    now += 3601
    response = client.get(created["status_url"])
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "job_not_found"


def test_api_exposes_typed_artifact_metadata_and_download_headers() -> None:
    client = TestClient(create_app(refiner=FakeRefiner()))
    created = client.post(
        "/api/v1/jobs/parse",
        json={
            "file_name": "x.png",
            "content_base64": _png(),
            "mode": "Balanced",
        },
    ).json()
    job_id = created["job_id"]

    response = client.get(f"/api/v1/jobs/{job_id}/artifacts")
    assert response.status_code == 200
    artifacts = response.json()["artifacts"]
    assert {item["name"] for item in artifacts} == {
        "document.md",
        "parse-result.json",
        "annotated.pdf",
        "document.html",
        "bundle.zip",
    }
    assert all(item["bytes"] > 0 for item in artifacts)
    assert all(item["download_url"].startswith(f"/api/v1/jobs/{job_id}/") for item in artifacts)

    download = client.get(f"/api/v1/jobs/{job_id}/artifacts/document.md")
    assert download.status_code == 200
    assert download.headers["content-disposition"] == 'attachment; filename="document.md"'


def test_extract_reuses_the_canonical_parse_without_rerunning_rapidocr() -> None:
    class Engine:
        calls = 0

        def __call__(self, image, *, return_word_box):
            self.calls += 1
            return SimpleNamespace(
                boxes=[[[1, 1], [20, 1], [20, 10], [1, 10]]],
                txts=["Invoice 42"],
                scores=[0.99],
                elapse_list=[],
                word_results=[],
                elapse=0.01,
            )

    class Refiner:
        def validate_configuration(self) -> None:
            pass

        def refine(self, pages, image_pages, capabilities, allowed_classes, extraction_schema):
            return CloudResult(refined_markdown="Invoice 42", reviewed_pages=[1]), UsageRecord(
                call_count=1
            )

        def refine_markdown(self, markdown, pages, capabilities, allowed_classes, schema):
            assert markdown.endswith("Invoice 42")
            assert capabilities == {"Extract"}
            return (
                CloudResult(
                    refined_markdown=markdown,
                    reviewed_pages=[1],
                    extracted_fields=[
                        ExtractedField(
                            path="invoice_id",
                            value="42",
                            status="verified",
                            confidence=0.9,
                            evidence=[CloudEvidence(page=1, block_id="p1-b1", quote="Invoice")],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    engine = Engine()
    client = TestClient(create_app(ocr_resource=OCRResource(engine, "CPU"), refiner=Refiner()))
    created = client.post(
        "/api/v1/jobs/parse",
        json={"file_name": "x.png", "content_base64": _png()},
    ).json()
    response = client.post(
        f"/api/v1/jobs/{created['job_id']}/extract",
        json={
            "schema_version": "1",
            "schema": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
            },
        },
    )

    assert response.status_code == 200
    assert engine.calls == 1
    assert response.json()["state"] == "ACCEPTED"


def test_mocked_dual_engine_api_flow_exports_audited_bundle() -> None:
    class Engine:
        calls = 0

        def __call__(self, image, *, return_word_box):
            self.calls += 1
            return SimpleNamespace(
                boxes=[[[1, 1], [20, 1], [20, 10], [1, 10]]],
                txts=["Invoice 42"],
                scores=[0.99],
                elapse_list=[],
                word_results=[],
                elapse=0.01,
            )

    class Models:
        def retrieve(self, model: str) -> None:
            assert model == "gpt-5.6-luna"

    class Responses:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            prompt = kwargs["input"][0]["content"][0]["text"]
            fields = []
            if "invoice_id" in prompt:
                fields = [
                    ExtractedField(
                        path="invoice_id",
                        value="42",
                        status="verified",
                        confidence=0.95,
                        evidence=[CloudEvidence(page=1, block_id="p1-b1", quote="Invoice 42")],
                    )
                ]
            parsed = (
                MarkdownWorkflowResult(reviewed_pages=[1], extracted_fields=fields)
                if kwargs["text_format"] is MarkdownWorkflowResult
                else CloudResult(refined_markdown="Invoice 42", reviewed_pages=[1])
            )
            return SimpleNamespace(
                output_parsed=parsed,
                usage=SimpleNamespace(
                    input_tokens=10,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens=5,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=1),
                    total_tokens=15,
                ),
            )

    engine = Engine()
    responses = Responses()
    refiner = OpenAIRefiner(client=SimpleNamespace(models=Models(), responses=responses))
    client = TestClient(create_app(ocr_resource=OCRResource(engine, "CPU"), refiner=refiner))
    created = client.post(
        "/api/v1/jobs/parse",
        json={"file_name": "x.png", "content_base64": _png(), "mode": "Balanced"},
    ).json()
    job_id = created["job_id"]
    extracted = client.post(
        f"/api/v1/jobs/{job_id}/extract",
        json={
            "schema_version": "1",
            "schema": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
            },
        },
    )

    assert extracted.status_code == 200
    assert extracted.json()["state"] == "ACCEPTED"
    assert engine.calls == 1
    assert len(responses.calls) == 2
    assert all(call["model"] == "gpt-5.6-luna" for call in responses.calls)
    assert all(call["reasoning"] == {"effort": "medium"} for call in responses.calls)
    assert all(item["type"] == "input_text" for item in responses.calls[1]["input"][0]["content"])

    bundle = client.get(f"/api/v1/jobs/{job_id}/artifacts/bundle.zip")
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        assert set(archive.namelist()) == {
            "document.md",
            "parse-result.json",
            "annotated.pdf",
            "document.html",
            "manifest.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["manifest_version"] == 3
    assert manifest["agent_workflow"]["current_state"] == "ACCEPTED"
    assert manifest["processing"]["gpt_model"] == "gpt-5.6-luna"
    assert manifest["processing"]["reasoning_effort"] == "medium"
    assert manifest["usage_and_cost"]["gpt"]["call_count"] == 2
