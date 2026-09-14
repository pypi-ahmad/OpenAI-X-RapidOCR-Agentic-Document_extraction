<!-- generated-by: gsd-doc-writer -->
# Testing

## Test framework and setup

The suite uses [pytest](https://docs.pytest.org/) (>=9.1.1) and
`pytest-cov` (>=7.1.0). The project targets Python 3.13 and uses `uv` to
manage its environment. Synchronize the development dependencies before
running tests:

```powershell
uv sync --all-groups
```

Tests are collected from `tests/`. The suite uses fakes and monkeypatching for
OpenAI, RapidOCR, and layout boundaries. The autouse fixture in
`tests/conftest.py` supplies a fake PP-DocLayoutV3 runtime, while individual
modules provide fake OpenAI Responses clients and OCR engines.

With `RUN_LIVE_PROMPT_EVAL` unset, the paid prompt-quality module is skipped.
Keep this variable unset for normal local and full-suite runs.

Prompt-quality smoke benchmarks are explicitly opt-in because they make paid,
non-deterministic requests to `gpt-5.6-luna`. To authorize them with valid
OpenAI configuration:

```powershell
$env:RUN_LIVE_PROMPT_EVAL = "1"
uv run pytest -o addopts="" tests/test_prompt_quality_live.py -m live -s
```

These curated cases check grounded extraction, unsupported-field abstention,
checkbox coverage, and checkbox states. They are regression signals, not a
claim of accuracy on unseen documents.

## Running tests

Run the complete suite, including the configured coverage check:

```powershell
uv run pytest
```

During iteration, disable the repository-wide coverage addopts and run only the
affected modules. The final validation must still use the unmodified full-suite
command above so the 80% coverage gate is enforced:

```powershell
uv run pytest -o addopts="" tests/test_document_chat.py tests/test_navigation.py
```

Run one test module:

```powershell
uv run pytest -o addopts="" tests/test_pipeline.py
```

Run an individual test by node ID:

```powershell
uv run pytest -o addopts="" tests/test_pipeline.py::test_pipeline_fails_when_gpt_refinement_fails
```

Run the mocked dual-engine API flow, including its audited ZIP manifest:

```powershell
uv run pytest -o addopts="" `
  tests/test_api.py::test_mocked_dual_engine_api_flow_exports_audited_bundle
```

Run the complete mandatory dual-engine and pricing regression set:

```powershell
uv run pytest -o addopts="" `
  tests/test_pipeline.py tests/test_hybrid_pipeline.py `
  tests/test_openai_refiner.py tests/test_costs.py
```

This set verifies the RapidOCR-first/Luna-second contract, with local layout and
table analysis between those engines. It also covers the fixed `gpt-5.6-luna`
model and `medium` reasoning effort, mode-specific context, missing-engine
failures, reported-token accounting, cached-input and cache-write pricing, and
the long-context pricing multiplier.

Run the local OCR, layout, table, checkbox, redaction, and visual-routing tests:

```powershell
uv run pytest -o addopts="" `
  tests/test_local_parse.py tests/test_layout.py tests/test_table_structure.py `
  tests/test_table_worker_contract.py tests/test_checkbox_vision.py `
  tests/test_redaction_vision.py tests/test_visual_routing.py
```

These tests cover CPU/CUDA selection, bounded OCR workers, layout and table
normalization, immutable RapidOCR grounding, local visual candidates, Luna
adjudication boundaries, and safe publication into canonical Markdown.

Run focused application-boundary checks:

```powershell
uv run pytest -o addopts="" tests/test_timing.py tests/test_local_cache.py
uv run pytest -o addopts="" tests/test_local_artifacts.py tests/test_export.py
uv run pytest -o addopts="" tests/test_api.py tests/test_launcher.py
uv run pytest -o addopts="" tests/test_ui_state.py tests/test_navigation.py
```

Run the document-only chat grounding and navigation checks:

```powershell
uv run pytest -o addopts="" `
  tests/test_document_chat.py tests/test_navigation.py tests/test_prompt_resources.py
```

These tests prove that chat receives generated Markdown rather than original
files, retrieves bounded document-scoped excerpts, limits recent conversation
history, refuses unknown citations, fails closed for off-topic or insufficient
evidence responses, serializes untrusted prompt-injection text as data, and
clears chat history when the selected document scope changes. Prompt-resource
tests also verify that reusable Luna instructions are packaged as versioned
Markdown files.

Run agentic workflow validation independently:

```powershell
uv run pytest -o addopts="" tests/test_workflow.py tests/test_routing.py tests/test_schema_input.py
```

This group covers classification, hierarchical sections, document splitting,
schema input, grounded extraction, abstention and review behavior, validation
rules, immutable source evidence, bounded repair, and the required
RapidOCR-then-GPT execution order.

Run the CUDA provider-detection and High Accuracy block-review regressions:

```powershell
uv run pytest -o addopts="" `
  tests/test_local_parse.py tests/test_openai_refiner.py tests/test_hybrid_pipeline.py
```

`tests/test_ui_state.py` and `tests/test_navigation.py` use Streamlit's
`AppTest` API to exercise UI state, configuration messaging, reruns, usage and
timing metrics, page navigation, document-scoped chat history, and explicit
session reset without starting a browser. The artifact tests inspect generated
Markdown, JSON, HTML, annotated PDF, ZIP, and manifest content. API tests use
`TestClient` with fake engine boundaries; the launcher test verifies that
`run_app.cmd` targets only confirmed TCP port `8841` listeners.

The cache tests verify byte and entry bounds, content- and DPI-sensitive keys,
engine namespaces, page-ID rebinding, ordered mixed hits and misses, and reuse
of successful OCR/layout/table results without caching failures. Timing tests
verify exclusive-stage ranking and ignore missing or invalid values.

Run evaluation-data schema gates and text-free oracle-comparison tests:

```powershell
uv run pytest -o addopts="" tests/test_evaluation_data.py tests/test_oracle_eval.py
```

The tracked review pack is intentionally pending human review. Its tests check
object-level table and checkbox labels, normalized geometry, and the strict
human-approval gate. Oracle tests compare aggregate Markdown and structure
metrics without copying document text into the report.

For a manual Streamlit launch check, start the app in the foreground and verify
that it reports the local URL without an import or configuration exception:

```powershell
uv run streamlit run app.py --server.port 8841
```

This command occupies the terminal until stopped. It verifies startup only; it
does not replace the `AppTest` UI assertions.

Checkbox coverage includes all-page visual routing, structured discovery,
bounded crop verification, grounding and confidence gates, business-rule
review, human corrections, Markdown markers, overlay geometry, ZIP crops, API
schemas, and prompt-resource packaging.

High Accuracy coverage verifies that prompt context flags only OCR blocks with
scores strictly below the fixed `0.85` threshold, that each such block is
recorded as accepted, abstained, rejected, or missing, and that only an accepted
grounded outcome resolves review. It also verifies that Balanced mode
does not enforce this per-block review contract. CUDA regression tests cover
the conventional `CUDAExecutionProvider` eligibility path when plugin-device
enumeration reports only CPU, plus the no-device CPU fallback and warning.

Pytest is configured with strict marker checking and emits a terminal report
for missing coverage. No watch-mode command is configured.

## Writing new tests

Add tests under `tests/` using the `test_*.py` naming convention. Name test
functions `test_<behavior>()` and keep them focused on observable outcomes.

Use pytest fixtures such as `monkeypatch` to replace external or expensive
boundaries. Existing examples include:

- `tests/test_pipeline.py`, which replaces document loading and OCR functions
  and supplies fake refiners.
- `tests/conftest.py`, whose autouse fixture supplies an isolated fake layout
  runtime to every test.
- `tests/test_openai_refiner.py`, which supplies a small fake OpenAI Responses
  client and verifies the strict low-confidence prompt flag boundary.
- `tests/test_hybrid_pipeline.py`, which verifies High Accuracy per-block
  refinement outcomes and resulting human-review state.
- `tests/test_local_parse.py`, which fakes ONNX Runtime provider discovery to
  test CUDA eligibility and CPU fallback without requiring GPU hardware.
- `tests/test_api.py`, which exercises the FastAPI app through
  `fastapi.testclient.TestClient` with a fake refiner.
- `tests/test_ui_state.py`, which drives the Streamlit entry point through
  `streamlit.testing.v1.AppTest`.
- `tests/test_navigation.py`, which uses `AppTest` for multipage navigation,
  document-scoped chat history, and global session reset behavior.
- `tests/test_local_cache.py` and `tests/test_timing.py`, which cover local
  cache correctness and exclusive-stage timing summaries.
- `tests/test_evaluation_data.py`, which validates the review-pack schema and
  refuses to treat draft or partially reviewed annotations as human gold data.
- `tests/test_document_chat.py`, which verifies Markdown-only retrieval,
  document scope, bounded history, Luna request construction, citation
  validation, fail-closed rendering, and prompt-injection resistance.
- `tests/test_navigation.py`, which verifies the separate workflow pages and
  document-chat selection behavior through Streamlit `AppTest`.
- `tests/test_prompt_resources.py`, which checks that every reusable model
  prompt is a versioned packaged `.md` resource and that rendering preserves
  untrusted-data delimiters.
- `tests/test_workflow.py`, which covers state transitions, evidence grounding,
  schema and business-rule validation, review and abstention decisions, and
  the required RapidOCR, local layout, and GPT stage ordering while preserving
  the mandatory dual-engine contract.

For parameterized inputs, use `@pytest.mark.parametrize`, as demonstrated in
`tests/test_ingest.py`. The project declares a `live` marker for tests that
use the live OpenAI API; strict marker checking requires any new custom marker
to be declared in `pyproject.toml`.

## Coverage requirements

The default test command enforces at least 80% coverage for the
`agentic_extractor` package.

| Type | Threshold |
| --- | --- |
| Overall coverage | 80% |
| Lines | No separate threshold configured |
| Branches | No separate threshold configured |
| Functions | No separate threshold configured |
| Statements | No separate threshold configured |

Coverage settings are defined in `pyproject.toml` under
`[tool.pytest.ini_options]`.

## CI integration

No CI workflow configuration was detected in the repository. Run
the tests and configured static checks locally before submitting changes:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Ruff targets Python 3.13 with a 100-character line length and enables the
`E`, `F`, `I`, `UP`, `B`, and `SIM` rule families. `ty` uses Python 3.13 for
type checking.

## Related documentation

Compare a generated public Parse result against a locally supplied approved oracle without printing
document text:

```powershell
uv run python -m agentic_extractor.oracle_eval `
  "LandingAI Output/Masked_Amerigroup_RealSolutions_1.parse.json" `
  "Our App Output/parse-result.json" `
  --output ".planning/tmp/oracle-comparison.json"
```

The source, oracle, generated-output, and temporary report paths are intentionally ignored by Git.

- [Development guide](DEVELOPMENT.md): local setup, project commands, and
  coding conventions.
- [Local API reference](API.md): endpoint contracts and API testing context.
- [Context engineering](CONTEXT-ENGINEERING.md): lossless payload, batching, and telemetry checks.
