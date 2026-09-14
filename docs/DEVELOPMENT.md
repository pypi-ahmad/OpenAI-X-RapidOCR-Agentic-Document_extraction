<!-- generated-by: gsd-doc-writer -->
# Development

## Local setup

This project uses `uv` to manage the Python 3.13 application environment,
development tools, and the separate PP-DocLayoutV3 worker environment. Fork
the GitHub repository, clone your fork, and synchronize both environments:

```powershell
git clone https://github.com/<your-account>/OpenAI-X-RapidOCR-Agentic-Document_extraction.git
cd OpenAI-X-RapidOCR-Agentic-Document_extraction
git remote add upstream https://github.com/pypi-ahmad/OpenAI-X-RapidOCR-Agentic-Document_extraction.git
uv sync --all-groups
uv sync --project tools/pp_doclayout --locked
```

Every successful extraction requires RapidOCR, PP-DocLayoutV3, and valid OpenAI
configuration. Set `OPENAI_API_KEY` in the process that launches the app. Never
put its value in source files, tests, fixtures, logs, screenshots, or commits.
See [Configuration](CONFIGURATION.md) for optional environment variables and
safe readiness checks.

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
| `app_pages/` | HTML, Classify, Section, Split, Extract, Chat, and Diagnostics pages. |
| `agentic-extractor` | Installed console script that launches `app.py` through Streamlit. |
| `agentic_extractor.api:app` | Local FastAPI application served separately with Uvicorn. |
| `run_app.cmd` | Windows launcher for the Streamlit UI; verifies and clears only TCP listeners on port `8841`, then keeps foreground logs visible. |

The Streamlit UI and FastAPI application both call the canonical workflow; do
not add extraction logic directly to either entry point. See
[Architecture](ARCHITECTURE.md) for engine order, public boundaries, and data
flow, and [API](API.md) for the local HTTP contract.

## Build and development commands

The project has no package-manager script table. Use these `uv` commands from
the repository root instead:

| Command | Description |
| --- | --- |
| `uv sync --all-groups` | Synchronize runtime and development dependencies into the project environment. |
| `uv sync --project tools/pp_doclayout --locked` | Synchronize the required isolated PP-DocLayoutV3 worker from its lockfile. |
| `uv run streamlit run app.py --server.port 8841` | Run the local Streamlit interface. |
| `uv run agentic-extractor` | Start the Streamlit interface through the installed console script. |
| `uv run uvicorn agentic_extractor.api:app --host 127.0.0.1 --port 8842` | Run the local FastAPI surface. |
| `uv build` | Build the wheel and source distribution through Hatchling. |
| `uv run ruff format .` | Format Python source files. |
| `uv run ruff format --check .` | Check formatting without modifying files. |
| `uv run ruff check .` | Run lint checks. |
| `uv run ty check` | Run static type checking. |
| `uv run pytest -o addopts="" tests/test_<area>.py` | Run a focused test file during iteration without the repository-wide coverage addopts. |
| `uv run pytest` | Run the test suite with coverage enforcement. |

No separate application build step is required for local development. Hatchling
is the configured Python package build backend.

## Module boundaries

- Put upload decoding and page-range rules in `src/agentic_extractor/ingest.py`,
  local engine handling in `src/agentic_extractor/ocr.py`, and evidence-backed
  layout reconstruction in `src/agentic_extractor/parse.py`.
- Keep the required PP-DocLayoutV3 worker boundary, layout ordering, table
  routing, and layout/table cache integration in
  `src/agentic_extractor/layout.py`. The isolated worker and its dependency
  lock belong under `tools/pp_doclayout/`; table grounding and review contracts
  belong in `src/agentic_extractor/table_structure.py`.
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
- Keep stage names, duration accumulation, and bottleneck summaries in
  `src/agentic_extractor/timing.py`. Record elapsed wall time at the stage that
  owns the work instead of reconstructing it later from unrelated totals.
- Keep bounded process caches in `src/agentic_extractor/cache.py`. Cache keys
  must cover the exact rendered or model-consumed pixels plus the settings and
  engine namespace that can change a result. Cached values may save local work,
  but they must not change page numbers, block IDs, or raw evidence.
- Build core artifact metadata in `src/agentic_extractor/artifacts.py`. The
  annotated PDF, coordinate HTML, and ZIP are intentionally generated on first
  access and cached in memory; metadata inspection must not trigger that work.
- Route both UI and API work through these canonical boundaries. Raw RapidOCR
  evidence is immutable; corrections belong in the auditable refinement layer.
- Keep document-chat contracts and local Markdown retrieval in
  `src/agentic_extractor/document_chat.py`, the Luna request boundary in
  `OpenAIRefiner.answer_document_question`, and page rendering in
  `app_pages/chat.py`. Chat sources must contain generated Parse Markdown and
  display metadata only, never original bytes, page images, OCR objects, or
  artifact payloads.
- Keep CUDA eligibility and fallback logic in `src/agentic_extractor/ocr.py`.
  Detection accepts either an enumerated CUDA plugin device or the conventional
  CUDA execution provider with a GPU device, then verifies the detector
  session provider after OCR initializes. Cover detection and CPU fallback
  changes in `tests/test_local_parse.py`.

## Streamlit navigation and session state

`app.py` owns page configuration and top navigation. Parse is the default page;
HTML, Classify, Section, Split, Extract, Chat, and Diagnostics are separate
`st.Page` entries. Keep workflow controls in the Parse sidebar and result
rendering on the corresponding page under `app_pages/`. New pages should read
canonical results from session state instead of re-running OCR or duplicating
pipeline logic.

The Parse page initializes shared keys with `setdefault`. A completed or
partially completed Parse is registered in `processed_documents` as a JSON-mode
dictionary produced from `ProcessedMarkdownDocument`; each entry contains its
document ID, display name, generated Markdown, selected-page metadata, status,
and failed-page numbers. `current_processed_document_id`
keeps reruns of the same upload attached to one entry. `usage_history` is shared
by Parse and document chat so both kinds of Luna calls appear in the existing
usage panel. The application reset action clears the entire Streamlit session.

Document chat is intentionally session-only. It permits up to 12 processed
documents, defaults to the newest entry, and saves and restores `chat_messages`
for each `chat_scope`; a previously unseen scope starts empty. Local retrieval
splits Markdown by page markers and
headings, limits excerpts and context characters, and sends only those excerpts
plus the six most recent visible messages to Luna. An answer is rendered only
when its citation IDs match supplied excerpts; unknown or missing citations fail
closed as insufficient evidence. The retrieval limits are deterministic character
budgets, not token estimates; provider-reported token usage remains the source for
cost telemetry. Cover these boundaries in
`tests/test_document_chat.py` and navigation/state behavior in
`tests/test_navigation.py`.

## Timing, caches, and artifacts

`LocalParseResult.timings` holds wall-clock measurements owned by the stages
listed in `src/agentic_extractor/timing.py`. Use `record_stage_timing` when a
stage can run more than once. `stage_timing_summary` ignores invalid and zero
values, sorts measured stages from slowest to fastest, and keeps
`total_seconds` separate from the sum of stage rows. The Diagnostics page and
artifact manifest both consume this summary. Add focused coverage in
`tests/test_timing.py` when stage names or summary behavior change.

Rendered pages, successful OCR results, layout results, and table structures
use four independent process-memory LRU caches. Each cache is limited to 256
entries and 128 MiB. The Reset session action does not clear these caches;
`clear_local_caches` exists for tests and controlled maintenance. Changes to
cache keys, serialization, eviction, or grounding rebinding belong with
`tests/test_local_cache.py`.

`build_local_artifacts` finalizes Markdown, Parse JSON, the manifest, and
checkbox crops first. The annotated PDF, coordinate HTML, and ZIP remain
pending until an allowlisted property or `get` request needs them. Their bytes
are generated once under a lock, kept in process memory, and reflected in
artifact metadata and timing analysis. Preserve this lazy boundary and test it
in `tests/test_local_artifacts.py` and `tests/test_export.py`.

## Code style

- Ruff provides formatting and linting through `pyproject.toml`. It uses a
  100-character line length, targets Python 3.13, and enables the `E`, `F`,
  `I`, `UP`, `B`, and `SIM` lint rule groups. Run `uv run ruff format .` before
  submitting changes, then `uv run ruff check .`.
- ty provides static type checking. Its Python version is configured as
  3.13 in `pyproject.toml`; run `uv run ty check`.
- pytest is configured in `pyproject.toml` to use strict markers, test the
  `tests` directory, and require at least 80% coverage for
  `agentic_extractor`.

During iteration, override the repository addopts for a narrow test file, for
example `uv run pytest -o addopts="" tests/test_timing.py`. Before handing off
work, run the full `uv run pytest` command so the 80% coverage check runs.
Tests marked `live` can make paid OpenAI requests and must run only with
explicit authorization.

## Branch conventions

The repository's default development branch is `main`. `CONTRIBUTING.md`
documents short descriptive branch names and focused commits; no stricter
repository-enforced convention is configured.

## Pull request process

For locally prepared changes:

- Preserve unrelated work and keep changes within the responsible module.
- Add focused regression tests under `tests/` for behavior changes.
- Keep secrets and private source documents out of code, fixtures, logs,
  screenshots, generated documentation, and commits.
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
