<!-- generated-by: gsd-doc-writer -->
# Development

## Local setup

This project uses `uv` to manage Python 3.13 and all dependencies, including
development tools. Fork the GitHub repository, clone your fork, and synchronize
the full development environment:

```powershell
git clone https://github.com/<your-account>/OpenAI-X-RapidOCR-Agentic-Document_extraction.git
cd OpenAI-X-RapidOCR-Agentic-Document_extraction
git remote add upstream https://github.com/pypi-ahmad/OpenAI-X-RapidOCR-Agentic-Document_extraction.git
uv sync --all-groups
```

Every successful extraction requires valid OpenAI configuration. Configure the
required environment outside the repository, then follow
[Configuration](CONFIGURATION.md) to verify variable presence without printing
secret values.

```powershell
uv run streamlit run app.py --server.port 8841
```

The Streamlit server is bound to `127.0.0.1` on port `8841` by default. On
Windows, `run_app.cmd` provides an equivalent launcher that clears a verified
listener on that port before starting the app.

## Application entry points

| Entry point | Role |
| --- | --- |
| `app.py` | Streamlit navigation and page configuration. |
| `streamlit_app.py` | Parse workspace and session lifecycle. |
| `app_pages/` | Classify, Section, Split, Extract, and document-chat pages. |
| `agentic-extractor` | Installed console script that launches `app.py` through Streamlit. |
| `agentic_extractor.api:app` | Local FastAPI application served separately with Uvicorn. |
| `run_app.cmd` | Windows launcher for the Streamlit UI; verifies and clears only TCP listeners on port `8841`, then keeps foreground logs visible. |

The Streamlit UI and FastAPI application both call the canonical workflow; do
not add extraction logic directly to either entry point. See
[Architecture](architecture.md) for engine order, public boundaries, and data
flow, and [API](API.md) for the local HTTP contract.

## Build and development commands

The project has no package-manager script table. Use these `uv` commands from
the repository root instead:

| Command | Description |
| --- | --- |
| `uv sync --all-groups` | Synchronize runtime and development dependencies into the project environment. |
| `uv run streamlit run app.py --server.port 8841` | Run the local Streamlit interface. |
| `uv run agentic-extractor` | Start the Streamlit interface through the installed console script. |
| `uv run uvicorn agentic_extractor.api:app --host 127.0.0.1 --port 8842` | Run the local FastAPI surface. |
| `uv run ruff format .` | Format Python source files. |
| `uv run ruff format --check .` | Check formatting without modifying files. |
| `uv run ruff check .` | Run lint checks. |
| `uv run ty check` | Run static type checking. |
| `uv run pytest` | Run the test suite with coverage enforcement. |

No separate application build step is required for local development. Hatchling
is the configured Python package build backend.

## Module boundaries

- Put upload decoding and page-range rules in `src/agentic_extractor/ingest.py`,
  local engine handling in `src/agentic_extractor/ocr.py`, and evidence-backed
  layout reconstruction in `src/agentic_extractor/parse.py`.
- Keep OpenAI request policy and response parsing in
  `src/agentic_extractor/openai_refiner.py`. Dual-engine sequencing and
  evidence-gated merging belong in `src/agentic_extractor/pipeline.py`;
  ADE-style state transitions and deterministic adjudication belong in
  `src/agentic_extractor/workflow.py`.
- Keep the High Accuracy block-review threshold centralized as
  `LOW_CONFIDENCE_THRESHOLD` in `src/agentic_extractor/parse.py`. Blocks below
  `0.85` are flagged in the existing page-level Parse prompt; High Accuracy
  requires GPT to confirm the original text with grounded evidence, return a
  grounded correction, or abstain for each flagged block. The merge records
  accepted, rejected, abstained, and missing outcomes; every outcome except an
  accepted one is review-required. Balanced uses the same prompt flag without
  enforcing per-block outcome coverage.
- Treat files under `src/agentic_extractor/prompts/` as versioned runtime
  resources. When prompt text or its contract changes, update its
  `prompt-version` metadata and the exact version expectations in
  `tests/test_prompt_resources.py`; keep High Accuracy prompt-contract tests in
  `tests/test_openai_refiner.py` and outcome enforcement tests in
  `tests/test_hybrid_pipeline.py`.
- Keep shared Pydantic contracts in `src/agentic_extractor/models.py`, artifact
  construction in `src/agentic_extractor/artifacts.py`, and token pricing in
  `src/agentic_extractor/costs.py`.
- Route both UI and API work through these canonical boundaries. Raw RapidOCR
  evidence is immutable; corrections belong in the auditable refinement layer.
- Keep document-chat contracts and local Markdown retrieval in
  `src/agentic_extractor/document_chat.py`, the Luna request boundary in
  `OpenAIRefiner.answer_document_question`, and page rendering in
  `app_pages/chat.py`. Chat sources must contain generated Parse Markdown and
  display metadata only—never original bytes, page images, OCR objects, or
  artifact payloads.
- Keep CUDA eligibility and fallback logic in `src/agentic_extractor/ocr.py`.
  Detection accepts either an enumerated CUDA plugin device or the conventional
  CUDA execution provider with a GPU device, then verifies the detector
  session provider after OCR initializes. Cover detection and CPU fallback
  changes in `tests/test_local_parse.py`.

## Streamlit navigation and session state

`app.py` owns page configuration and top navigation. Parse is the default page;
Classify, Section, Split, Extract, and Chat are separate `st.Page` entries.
Keep workflow controls in the Parse sidebar and result rendering on the
corresponding page under `app_pages/`. New pages should read canonical results
from session state instead of re-running OCR or duplicating pipeline logic.

The Parse page initializes shared keys with `setdefault`. A completed or
partially completed Parse is registered in `processed_documents` as a frozen
`ProcessedMarkdownDocument`; the registry contains only generated Markdown,
selected-page metadata, status, and failed-page numbers. `current_processed_document_id`
keeps reruns of the same upload attached to one entry. `usage_history` is shared
by Parse and document chat so both kinds of Luna calls appear in the existing
usage panel. The application reset action clears the entire Streamlit session.

Document chat is intentionally session-only. It permits up to 12 processed
documents, defaults to the newest entry, and clears `chat_messages` whenever
`chat_scope` changes. Local retrieval splits Markdown by page markers and
headings, limits excerpts and context characters, and sends only those excerpts
plus the six most recent visible messages to Luna. An answer is rendered only
when its citation IDs match supplied excerpts; unknown or missing citations fail
closed as insufficient evidence. The retrieval limits are deterministic character
budgets, not token estimates; provider-reported token usage remains the source for
cost telemetry. Cover these boundaries in
`tests/test_document_chat.py` and navigation/state behavior in
`tests/test_navigation.py`.

## Code style

- **Ruff** provides formatting and linting through `pyproject.toml`. It uses a
  100-character line length, targets Python 3.13, and enables the `E`, `F`,
  `I`, `UP`, `B`, and `SIM` lint rule groups. Run `uv run ruff format .` before
  submitting changes, then `uv run ruff check .`.
- **ty** provides static type checking. Its Python version is configured as
  3.13 in `pyproject.toml`; run `uv run ty check`.
- **pytest** is configured in `pyproject.toml` to use strict markers, test the
  `tests` directory, and require at least 80% coverage for
  `agentic_extractor`.

## Branch conventions

The repository's current default development branch is `main`. No feature-branch
naming or commit-message convention is documented; use a short descriptive
branch name and keep commits focused unless maintainers request another format.

## Pull request process

No pull-request template or repository-enforced review workflow is present in
this checkout. For locally prepared changes:

- Preserve unrelated work and keep changes within the responsible module.
- Add focused regression tests under `tests/` for behavior changes.
- For High Accuracy review changes, cover the strict `<0.85` boundary,
  grounded acceptance, and missing, rejected, and abstained outcomes; also
  prove Balanced enforcement remains unchanged.
- Run formatting, lint, type, and test checks from the command table above.
- Document any user-visible behavior or configuration change without including
  credentials.

See [Contributing](../CONTRIBUTING.md) for contribution guidelines,
[Testing](TESTING.md) for focused test commands and coverage behavior, and
[Context engineering](CONTEXT-ENGINEERING.md) before changing Luna payloads or
prompt order.
