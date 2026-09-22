# Contributing

This guide covers development standards, testing expectations, and contributions to Agentic document extractor.

## Development setup

The repository requires Windows, Python `>=3.13`, and `uv`.

1. Clone the repository:

   ```powershell
   git clone https://github.com/pypi-ahmad/OpenAI-X-RapidOCR-Agentic-Document_extraction.git
   cd OpenAI-X-RapidOCR-Agentic-Document_extraction
   ```

2. Synchronize main dependencies and development tools:

   ```powershell
   uv sync --all-groups
   ```

3. Synchronize the isolated PP-DocLayoutV3 environment:

   ```powershell
   uv sync --project tools/pp_doclayout --locked
   ```

4. Configure your OpenAI API key in your session environment (never commit or hardcode credentials):

   ```powershell
   $env:OPENAI_API_KEY = "your-api-key"
   ```

## Coding and verification standards

Run the formatting, linting, typing, and test checks locally:

- Code formatting: Check formatting with `uv run ruff format --check .` and reformat with `uv run ruff format .`.
- Linting: Run `uv run ruff check .` to check for rule violations.
- Type checking: Validate static types with `uv run ty check`.
- Unit and integration testing: Run `uv run pytest`. The test suite requires at least 80% code coverage across `src/agentic_extractor`.
- Focused test iteration: Run specific test modules without coverage addopts using:

  ```powershell
  uv run pytest -o addopts="" tests/test_local_parse.py
  ```

- Live prompt tests: The suite marks paid model evaluations with `@pytest.mark.live`. These are skipped by default. Do not run live prompt tests unless explicitly authorized:

  ```powershell
  $env:RUN_LIVE_PROMPT_EVAL = "1"
  uv run pytest tests/test_prompt_quality_live.py -m live -s
  ```

## Repository architecture rules

Contributions follow these architecture rules:

1. Three-engine pipeline: Every successful extraction must execute RapidOCR first, PP-DocLayoutV3 second, and OpenAI `gpt-6-sol` third. Do not add single-engine fallback paths or bypass any of the three engines.
2. Immutable OCR evidence: Raw `Block` objects produced by RapidOCR must remain immutable. Model proposals and corrections must be recorded as additive refinement layers.
3. Downstream workflows: The Classify, Section, Split, and Extract workflows must consume the canonical refined Markdown and grounding index rather than re-running OCR.
4. Prompt resource management: Model prompt templates must be placed as versioned Markdown files in `src/agentic_extractor/prompts/` rather than hardcoded in Python code. Updates to prompt templates require incrementing the `prompt-version` metadata and updating expectations in `tests/test_prompt_resources.py`.
5. Untrusted document data: Uploaded text, extracted OCR content, and user-provided schemas are treated as untrusted evidence. They must not alter application routing, policy, or tool execution.
6. Local scope: Keep UI and API endpoints bound to local loopback (`127.0.0.1`). Do not introduce public hosting, external authentication systems, or multi-tenant database models.

## Branch and testing expectations

The repository has no remote CI pipelines, such as GitHub Actions, or issue templates. Run validation locally.

Before submitting changes, make sure these commands pass:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

If your changes touch the Streamlit user interface, launch the app locally and verify the UI on port `8841`:

```powershell
uv run streamlit run app.py --server.port 8841
```

If your changes affect the local API, run `tests/test_api.py` and verify Swagger documentation on port `8842`:

```powershell
uv run uvicorn agentic_extractor.api:app --host 127.0.0.1 --port 8842
```

## Issue reporting

When reporting issues, include:
- A concise description of the observed behavior versus expected behavior.
- Python version, operating system details, and hardware environment (GPU model or CPU).
- Whether the failure occurs on PDF, PNG, JPEG, or TIFF input files.
- Relevant console traceback messages.
- Do not include sensitive document contents or API credentials in issue reports.
