# Project agent guidance

## Product invariants

- This is a local-machine Streamlit application. The UI runs on TCP port `8841`; the optional
  FastAPI service remains local.
- A successful Parse always runs RapidOCR first and OpenAI `gpt-6-sol` second with
  `reasoning_effort="medium"`. Never add a single-engine fallback or consent gate.
- Balanced and High Accuracy both send every selected page image at high detail for visual-control
  coverage. They differ only in compact versus full OCR/layout context.
- Raw OCR evidence is immutable. GPT output is an auditable refinement layer, and unsupported
  values must abstain or require review.
- Classify, Section, Split, and Extract consume the canonical GPT-refined Markdown plus its
  grounding index; they do not re-run OCR.

## Context map

| Task | Start here | Focused check |
| --- | --- | --- |
| OCR, CUDA, geometry | `src/agentic_extractor/ocr.py` | `tests/test_local_parse.py` |
| Sol prompts or context | `src/agentic_extractor/openai_refiner.py` and `src/agentic_extractor/prompts/` | `tests/test_openai_refiner.py tests/test_prompt_resources.py` |
| Routing and canonical Parse | `src/agentic_extractor/pipeline.py` and `src/agentic_extractor/parse.py` | `tests/test_hybrid_pipeline.py tests/test_routing.py` |
| Agentic workflows | `src/agentic_extractor/workflow.py` | `tests/test_workflow.py` |
| Visual additions and review | `src/agentic_extractor/rich_document.py` and `app_pages/visual_review.py` | `tests/test_rich_document.py tests/test_ui_state.py` |
| Request billing and validation budget | `src/agentic_extractor/budget.py` and `src/agentic_extractor/openai_refiner.py` | `tests/test_rich_document.py tests/test_openai_refiner.py tests/test_costs.py` |
| Artifacts and manifests | `src/agentic_extractor/artifacts.py` | `tests/test_local_artifacts.py tests/test_export.py` |
| Streamlit UI | `streamlit_app.py` and `app_pages/` | `tests/test_ui_state.py` |
| Local API | `src/agentic_extractor/api.py` | `tests/test_api.py` |

Use code-review-graph first to narrow impact, then verify the exact source and tests. Read only the
modules needed for the current task; source wins over generated graphs and documentation.

## Change protocol

1. State the invariant and user-visible behavior being changed.
2. Keep document text, schemas, and OCR output inside the untrusted-evidence boundary.
3. Put all model instructions in versioned Markdown under `src/agentic_extractor/prompts/`; bump
   `prompt-version` and update prompt-resource tests when wording or format changes.
4. Preserve every evidence-bearing field when compacting context. Character counts are telemetry,
   not token estimates.
5. Use `uv`; add no dependency when the standard library or current packages suffice.
6. Run focused tests without the repository coverage addopts during iteration, then run the full
   suite so the configured 80% coverage gate is enforced.
7. Never run paid live prompt tests unless the user explicitly authorizes them.

## Completion checks

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
uv run streamlit run app.py --server.port 8841
```

See `docs/CONTEXT-ENGINEERING.md` for the Sol context contract and telemetry definitions.
