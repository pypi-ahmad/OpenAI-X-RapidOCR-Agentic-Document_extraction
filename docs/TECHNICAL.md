# Technical reference

This document details implementation-level architectural choices, system invariants, exception handling mechanisms, and persistence boundaries based directly on the codebase.

## Stack choices and rationale

| Technology / library | Location in codebase | Rationale evident in code |
|---|---|---|
| `uv` + Hatchling dual environment structure | `pyproject.toml`, `tools/pp_doclayout/pyproject.toml` | PaddleX 3.4 and PaddlePaddle GPU 3.2 require OpenCV package builds that conflict with the root application's dependencies. PP-DocLayoutV3 is isolated in a standalone sub-project with its own lockfile and invoked via a subprocess pipe. |
| RapidOCR pinned Git revision | `[tool.uv.sources]` in `pyproject.toml` | The application pins commit `e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7` from `https://github.com/RapidAI/RapidOCR.git` to guarantee reproducible OCR output and bounding polygon structures across installations. |
| `onnxruntime-gpu` with runtime fallback | `src/agentic_extractor/ocr.py` | CUDA execution provider availability is checked dynamically after detector session initialization. If CUDA registration fails or returns empty, the engine updates its provenance to CPU rather than halting execution. |
| Pydantic v2 data contracts | `src/agentic_extractor/models.py`, `src/agentic_extractor/api.py` | Pydantic defines data models across all application layers. In `api.py`, FastAPI generates its OpenAPI schema directly from these models, preventing drift between internal workflow state and the HTTP interface. |
| Versioned Markdown prompts | `src/agentic_extractor/prompts/`, `src/agentic_extractor/prompt_resources.py` | Prompts are stored as distinct `.md` template files with YAML frontmatter. This allows version tagging (`prompt-version`) and runtime SHA-256 hashing for auditability in telemetry and manifests. |
| Streamlit web UI | `app.py`, `streamlit_app.py`, `app_pages/` | Provides an interactive local interface with session state (`st.session_state`) for uploading files, inspecting bounding-box overlays, and running grounded conversational queries. |
| FastAPI + Uvicorn | `src/agentic_extractor/api.py` | Exposes a programmatic, asynchronous HTTP interface implementing the same canonical extraction pipeline without running the Streamlit frontend. |
| In-process LRU caches | `src/agentic_extractor/cache.py` | In-memory caching for rendered pages, OCR results, layout predictions, and table structures avoids external caching infrastructure. Cache keys incorporate exact image pixel hashes and engine version tags. |
| pypdf and pypdfium2 | `src/agentic_extractor/ingest.py`, `src/agentic_extractor/artifacts.py` | `pypdfium2` performs high-fidelity page rasterization at 150 DPI; `pypdf` extracts document page counts and handles PDF annotation and page extraction. |
| OpenCV (cv2) | `src/agentic_extractor/checkbox_vision.py`, `src/agentic_extractor/quality.py` | Provides computer-vision heuristics for checkbox proposal generation, pixel fill-ratio calculation, image blur detection (Laplacian variance), and contrast assessment. |

## System invariants

The codebase enforces the following invariants through type constraints, validation gates, and unit test assertions:

1. **Mandatory three-engine pipeline**: Every successful document Parse requires RapidOCR, PP-DocLayoutV3, and OpenAI `gpt-5.6-luna`. The system contains no fallback path to single-engine or offline-only extraction (`RapidOCRSetupError`, `PPDocLayoutSetupError`, and `OpenAIConfigurationError` halt execution immediately).
2. **Immutable raw OCR evidence**: `Block` objects produced by RapidOCR are immutable. Refinements from `gpt-5.6-luna` or human reviewers are captured as additive corrections and events; raw OCR coordinates, text, and scores are never overwritten in place.
3. **Untrusted document boundary**: Uploaded document content, extracted text, and user-provided schemas are treated as untrusted data. They cannot alter application routing, prompt policies, or tool execution.
4. **Exhaustive High Accuracy review contract**: In `High Accuracy` mode, every RapidOCR block with confidence strictly below `0.85` must resolve to an explicit outcome (`accepted`, `rejected`, `abstained`, or `missing`). Any unresolved or rejected block sets the workflow state to `REVIEW_REQUIRED`.
5. **Checkbox automation triple-agreement gate**: A checkbox state (`CHECKED` or `UNCHECKED`) is automated only when the OpenCV pixel detector, Luna visual crop review, and unique RapidOCR label grounding (confidence `>=0.85`) are in full agreement. Any disagreement forces `REVIEW_REQUIRED`.
6. **Subprocess contract validation**: Data returned by the PP-DocLayoutV3 worker process is treated as an untrusted boundary. Coordinates, confidence scores, and reading orders are parsed into strict models; invalid payloads raise `LayoutContractError`.
7. **Document chat citation isolation**: Chat answers must cite valid excerpt IDs supplied in the retrieved context. If citation verification fails or the model returns unsubstantiated content, the response fails closed with an insufficient-evidence message.

## Error handling

The application handles errors using typed exception hierarchies mapped to specific failure boundaries:

| Exception class | Origin module | Failure condition and system behavior |
|---|---|---|
| `IngestError` | `src/agentic_extractor/ingest.py` | Raised when file signatures are unrecognized, files exceed 50 MiB, pages exceed 200, images exceed 25 megapixels, or images are unreadable. Causes immediate validation failure. |
| `RapidOCRSetupError` | `src/agentic_extractor/workflow.py` | Raised when RapidOCR fails initialization or ONNX Runtime encounters fatal initialization errors. Halts workflow execution. |
| `PPDocLayoutSetupError` | `src/agentic_extractor/layout.py` | Raised when the isolated worker executable is missing or fails health checks. Prompts the user to run `uv sync --project tools/pp_doclayout --locked`. |
| `LayoutContractError` | `src/agentic_extractor/layout.py` | Raised when the worker process returns malformed JSON, invalid bounding box coordinates, or unrecognized layout labels. |
| `OpenAIConfigurationError` | `src/agentic_extractor/openai_refiner.py` | Raised during preflight checks if `OPENAI_API_KEY` is missing, empty, or fails authentication. Prevents local OCR from running unnecessarily. |
| `GPTRefinementError` | `src/agentic_extractor/pipeline.py` | Raised when an OpenAI API request fails, times out, or returns a response that cannot be parsed into the expected JSON schema. |
| `EvaluationDataError` | `src/agentic_extractor/evaluation_data.py` | Raised if an unapproved evaluation pack is used as benchmark gold data without explicit `human_approved` verification. |
| `ApiError` | `src/agentic_extractor/api.py` | Pydantic model for HTTP error payloads (`code`, `message`, `action`, `retryable`), used by `HTTPException` across FastAPI endpoints (returning 400, 404, 409, or 503). |

## Persistence paths

The application does not use an external database or automated filesystem logging:

| Data category | Storage location | Persistence lifetime |
|---|---|---|
| Rendered pages cache | In-memory `RENDER_CACHE` (`cache.py`) | Process lifetime; up to 256 entries / 128 MiB. Discarded on process restart. |
| OCR results cache | In-memory `OCR_CACHE` (`cache.py`) | Process lifetime; up to 256 entries / 128 MiB. Keyed by page pixel hash and engine version. |
| Layout results cache | In-memory `LAYOUT_CACHE` (`cache.py`) | Process lifetime; up to 256 entries / 128 MiB. Keyed by page pixel hash and model signature. |
| Table structures cache | In-memory `TABLE_CACHE` (`cache.py`) | Process lifetime; up to 256 entries / 128 MiB. Keyed by table crop image pixel hash. |
| UI session state | `st.session_state` (`ui_state.py`) | Browser session lifetime. Cleared when the user clicks **Reset session** or closes the browser tab. |
| API jobs and uploads | `_Job` dictionary in `create_app` (`api.py`) | Process lifetime; jobs expire after 3600 seconds. Expired jobs are removed lazily during subsequent lookups. |
| Generated artifacts | `LocalArtifacts` in-memory properties (`artifacts.py`) | Generated lazily on demand (PDF, HTML, ZIP) and held in memory. Written to disk only when the user explicitly downloads them. |
| Evaluation gold datasets | `evaluation-data/` directory | Durable files committed to the repository; human-approved evaluation packs. |
