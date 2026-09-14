from __future__ import annotations

import uuid
from typing import Any

import pytest

from agentic_extractor.layout import PPDocLayoutResource


class FakeLayoutClient:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def predict(self, pages: list[Any]) -> dict[int, dict[str, Any]]:
        self.calls.append([page.number for page in pages])
        return {
            page.number: {
                "boxes": [
                    {
                        "cls_id": 22,
                        "label": "text",
                        "score": 0.99,
                        "coordinate": [0, 0, page.image.width, page.image.height],
                        "polygon_points": [
                            [0, 0],
                            [page.image.width, 0],
                            [page.image.width, page.image.height],
                            [0, page.image.height],
                        ],
                        "order": 1,
                    }
                ]
            }
            for page in pages
        }


@pytest.fixture
def layout_resource() -> PPDocLayoutResource:
    return PPDocLayoutResource(
        client=FakeLayoutClient(),
        device="CPU",
        version="3.4.0-test",
        paddle_version="3.2.0-test",
        warning="PP-DocLayoutV3 test CPU fallback.",
        cache_namespace=uuid.uuid4().hex,
    )


@pytest.fixture(autouse=True)
def use_fake_layout_runtime(monkeypatch: pytest.MonkeyPatch, layout_resource) -> None:
    monkeypatch.setattr(
        "agentic_extractor.layout.create_pp_doclayout_engine", lambda: layout_resource
    )
    monkeypatch.setattr(
        "agentic_extractor.pipeline.create_pp_doclayout_engine", lambda: layout_resource
    )
    monkeypatch.setattr(
        "agentic_extractor.workflow.create_pp_doclayout_engine", lambda: layout_resource
    )
    monkeypatch.setattr("agentic_extractor.api.create_pp_doclayout_engine", lambda: layout_resource)
