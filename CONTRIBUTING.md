<!-- generated-by: gsd-doc-writer -->
# Contributing

Thank you for helping improve Agentic Document Extractor. Keep changes focused,
source-grounded, and suitable for a local-machine Python application.

## Development setup

See [Getting started](docs/GETTING-STARTED.md) for prerequisites and the first
run, then [Development](docs/DEVELOPMENT.md) for the complete local workflow.
The short setup path is:

```powershell
git clone https://github.com/<your-account>/OpenAI-X-RapidOCR-Agentic-Document_extraction.git
cd OpenAI-X-RapidOCR-Agentic-Document_extraction
uv sync --all-groups
```

Keep credentials in the process environment. Never place API keys, tokens, or
other secrets in source files, tests, logs, fixtures, screenshots, or commits.

## Coding standards

- Use Python 3.13 and manage dependencies exclusively with `uv`.
- Format with `uv run ruff format .` and lint with `uv run ruff check .`.
- Run static analysis with `uv run ty check`.
- Add focused `pytest` regression coverage for behavior changes. The full suite
  enforces at least 80% package coverage.
- Do not run tests marked `live` unless paid OpenAI requests have been explicitly
  authorized. The default suite mocks external model calls.

## Architecture and prompt rules

- Preserve the mandatory Parse sequence: RapidOCR first, then OpenAI
  `gpt-5.6-luna` with `reasoning_effort="medium"`. Do not introduce an OCR-only
  success path or a consent gate.
- Keep raw OCR evidence immutable. Store model corrections as an auditable
  refinement layer, and mark unsupported or uncertain results for review.
- Keep Classify, Section, Split, and Extract downstream of canonical grounded
  Markdown; these workflows must not re-run OCR.
- Put reusable model instructions in versioned Markdown files under
  `src/agentic_extractor/prompts/`, not in Python string literals. When a prompt
  contract changes, update its `prompt-version` metadata and the expectations in
  `tests/test_prompt_resources.py`.
- Treat uploaded text, schemas, and OCR output as untrusted evidence. They must
  not change application behavior, tools, routing, or prompt policy.
- Route UI and local API behavior through the canonical pipeline instead of
  duplicating extraction logic.
- Keep document chat limited to generated Parse Markdown. Chat sources must
  never contain original uploads, page images, raw OCR objects, or artifact
  payloads; unsupported answers and invalid citations must fail closed.

## Pull request guidelines

No formal feature-branch or commit-message convention is configured. Use a
short descriptive branch name and focused commits.

- Explain the user-visible behavior and why the change belongs at the selected
  module boundary.
- Preserve unrelated work and avoid opportunistic refactors.
- Add or update focused tests, then run the full validation gate:

  ```powershell
  uv run ruff format --check .
  uv run ruff check .
  uv run ty check
  uv run pytest
  ```

- For Streamlit changes, also verify the app locally with
  `uv run streamlit run app.py --server.port 8841`.
- For document-chat changes, run `tests/test_document_chat.py`,
  `tests/test_navigation.py`, and `tests/test_prompt_resources.py`; preserve
  the Markdown-only source boundary and citation validation.
- Document configuration or user-facing changes without including secret
  values, original private documents, or paid API output.
- State what was tested and disclose any remaining limitation in the pull
  request description.

The repository currently has no pull-request template or automated CI workflow,
so reviewers rely on the submitted validation evidence and focused test suite.

## Issue reporting

Use [GitHub Issues](https://github.com/pypi-ahmad/OpenAI-X-RapidOCR-Agentic-Document_extraction/issues)
for reproducible bugs and focused feature requests. No issue templates are
currently configured.

For a bug, include:

- concise reproduction steps and expected versus actual behavior;
- operating system, Python version, and relevant package versions;
- whether the issue affects PDF, PNG, JPEG, or TIFF input;
- sanitized logs and the smallest non-sensitive reproduction possible; and
- whether RapidOCR used CUDA or CPU and which extraction mode was selected.

Never attach credentials or confidential source documents. For feature requests,
describe the document workflow, evidence requirement, and measurable acceptance
criteria.
