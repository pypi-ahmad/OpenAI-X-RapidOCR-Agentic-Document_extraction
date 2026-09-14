# Agentic document extractor

Agentic document extractor processes PDF, PNG, JPEG, and TIFF files on a local machine. It runs RapidOCR for text and coordinate detection, PP-DocLayoutV3 in an isolated worker process for region and table parsing, and OpenAI `gpt-5.6-luna` (medium reasoning effort) for refinement. The project includes a Streamlit interface, an optional local FastAPI service, schema validation, and artifact export (annotated PDF, coordinate HTML, and ZIP bundles).

## Requirements

Runtime and environment requirements:

- Operating system: Windows (required). The PP-DocLayoutV3 worker configuration in `tools/pp_doclayout/pyproject.toml` is constrained to `sys_platform == 'win32'`, worker invocation in `src/agentic_extractor/layout.py` calls `.venv/Scripts/python.exe`, and the background service launcher `run_app.cmd` is a Windows batch script.
- Python version: Python `>=3.13` for the main application (specified in `pyproject.toml` and `.python-version`); `>=3.13,<3.14` for the isolated layout worker environment (`tools/pp_doclayout/pyproject.toml`).
- Package manager: `uv` is required to resolve dependencies and manage the dual virtual environments.
- OpenAI API credentials: A valid `OPENAI_API_KEY` environment variable with access to the `gpt-5.6-luna` model.
- Hardware and accelerator: Optional NVIDIA GPU with CUDA 12 support. RapidOCR uses `onnxruntime-gpu` with runtime fallback to CPU. The PP-DocLayoutV3 worker environment includes `paddlepaddle-gpu==3.2.0` with CPU fallback.

## Setup and installation

1. Clone the repository:

   ```powershell
   git clone https://github.com/pypi-ahmad/OpenAI-X-RapidOCR-Agentic-Document_extraction.git
   cd OpenAI-X-RapidOCR-Agentic-Document_extraction
   ```

2. Synchronize root dependencies and development tools:

   ```powershell
   uv sync --all-groups
   ```

3. Synchronize the isolated PP-DocLayoutV3 worker environment:

   ```powershell
   uv sync --project tools/pp_doclayout --locked
   ```

   The worker environment isolates PaddleX 3.4 and PaddlePaddle GPU 3.2 from the main application's dependencies due to incompatible OpenCV requirements.

4. Set the OpenAI API key in your terminal session:

   ```powershell
   $env:OPENAI_API_KEY = "your-api-key"
   ```

## Run commands

### Streamlit web interface

The Streamlit web application is the primary interactive user interface, running on TCP port `8841`.

Start using `uv`:

```powershell
uv run streamlit run app.py --server.port 8841
```

Or run using the installed console script:

```powershell
uv run agentic-extractor
```

Or launch using the Windows batch launcher:

```cmd
run_app.cmd
```

`run_app.cmd` verifies that `uv` exists, stops any existing process listening on TCP port `8841`, and starts Streamlit in the foreground.

Once started, access the web interface at `http://127.0.0.1:8841`.

### Local HTTP API

The FastAPI service exposes programmatic job submission, status queries, schema extraction, and artifact downloads on TCP port `8842`.

Start the API service:

```powershell
uv run uvicorn agentic_extractor.api:app --host 127.0.0.1 --port 8842
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8842/docs`.

## Configuration

### Environment variables

The application reads the following environment variables:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | Unset | API key for `gpt-5.6-luna` refinement and document chat requests. |
| `OPENAI_BASE_URL` | No | Unset | Optional custom base URL for the OpenAI client. |
| `ADE_OCR_MAX_WORKERS` | No | `4` | Maximum parallel worker processes for CPU-based RapidOCR page processing (integer from `1` to `4`). |
| `CUDA_PATH` | No | Unset | Optional CUDA installation root directory on Windows; used by `src/agentic_extractor/ocr.py` to register `bin/x64` and `bin` on `PATH` for ONNX Runtime DLL discovery. |
| `RUN_LIVE_PROMPT_EVAL` | No | Unset | Test suite variable. Set to `1` to enable live paid OpenAI model tests in `tests/test_prompt_quality_live.py`. |

### Configuration files

- `pyproject.toml`: Root package definition, dependencies, pinned RapidOCR Git revision (`e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7`), build settings via Hatchling, Ruff linter/formatter rules, and pytest options.
- `tools/pp_doclayout/pyproject.toml`: Dependency specification and index configuration for the locked PP-DocLayoutV3 worker environment.
- `.python-version`: Pins Python version `3.13`.
- `.streamlit/config.toml`: Streamlit server settings (`address = "127.0.0.1"`, `port = 8841`, `maxUploadSize = 50`), browser metrics disabling, and UI dark theme styling.

## Repository map

```text
OpenAI-X-RapidOCR-Agentic-Document_extraction/
|-- app.py                          Streamlit navigation root and session initialization
|-- streamlit_app.py                Primary Parse interface, previews, and controls
|-- run_app.cmd                     Windows batch launcher for Streamlit on port 8841
|-- pyproject.toml                  Root project manifest, dependencies, and tool configs
|-- uv.lock                         Pinned dependency lockfile for the root project
|-- app_pages/                      Auxiliary Streamlit pages
|   |-- chat.py                     Document-grounded session chat page
|   |-- classify.py                 Document and page classification page
|   |-- diagnostics.py              System readiness, stage timings, and bottleneck analysis
|   |-- extract.py                  Structured schema extraction interface
|   |-- html.py                     Interactive coordinate-positioned HTML viewer page
|   |-- section.py                  Document hierarchical sectioning page
|   |-- split.py                    Logical document boundary splitting page
|-- src/agentic_extractor/          Core application package
|   |-- api.py                      FastAPI service routes and job lifecycle handling
|   |-- artifacts.py                Artifact generation (Markdown, JSON, PDF, HTML, ZIP)
|   |-- cache.py                    Bounded process-memory LRU caches for pages, OCR, and layout
|   |-- capabilities.py             Validation for classification, sectioning, splitting, extraction
|   |-- checkbox_vision.py          OpenCV checkbox detection and pixel state analysis
|   |-- cli.py                      Console script entry point for `agentic-extractor`
|   |-- config.py                   Application limits, model definitions, and settings
|   |-- costs.py                    Token usage tracking and cost calculations
|   |-- document_chat.py            Lexical chunking, retrieval, and conversation handling
|   |-- evaluation_data.py          Review pack schema for evaluation datasets
|   |-- export.py                   ZIP bundle assembly helper
|   |-- ingest.py                   File validation, signature checking, and image decoding
|   |-- landing_contract.py         LandingAI-compatible `parse-result.json` projection
|   |-- layout.py                   PP-DocLayoutV3 worker client and layout merging
|   |-- layout_html.py              Standalone HTML document generator with SVG overlays
|   |-- models.py                   Pydantic and dataclass models across all layers
|   |-- ocr.py                      RapidOCR execution, worker pools, and ONNX Runtime device handling
|   |-- openai_refiner.py           OpenAI client calls, structured outputs, and visual review
|   |-- oracle_eval.py              Evaluation comparison against ground-truth Parse results
|   |-- parse.py                    Reading order sorting, chunking, and Markdown synthesis
|   |-- pipeline.py                 Dual-engine pipeline orchestration and refinement merge
|   |-- prompt_resources.py         Versioned prompt loader and SHA-256 integrity verifier
|   |-- prompts/                    Packaged versioned Markdown prompt templates
|   |-- quality.py                  Image quality heuristics and optional preprocessing
|   |-- redaction_vision.py         Visual redaction mask detection
|   |-- schema_input.py             JSON Schema, guided field, and Markdown schema parsers
|   |-- table_structure.py          Table structure validation, cell grounding, and HTML rendering
|   |-- timing.py                   Wall-clock stage timing collection and bottleneck ranking
|   |-- ui_state.py                 Streamlit session state management and document caching
|   |-- visual_routing.py           Heuristic selection of high-detail image crops for Luna review
|   |-- workflow.py                 End-to-end agent workflow state machine
|-- tools/                          Auxiliary tools and worker environments
|   |-- pp_doclayout/               Isolated PaddleX PP-DocLayoutV3 worker and lockfile
|   |-- run_corpus_validation.py    Multi-document batch evaluation script
|   |-- run_real_validation.py      Single-document evaluation comparison script
|-- tests/                          Test suite with 320+ unit and integration tests
`-- docs/                           Detailed architectural and operational documentation
```

## How to run tests

The test suite uses `pytest` and requires at least 80% code coverage on `agentic_extractor`.

Run the full test suite:

```powershell
uv run pytest
```

Run a specific test file without the coverage threshold addopts:

```powershell
uv run pytest -o addopts="" tests/test_local_parse.py
```

Run code formatting and static type checks:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

Run live OpenAI prompt evaluations (requires valid API key and incurred API usage):

```powershell
$env:RUN_LIVE_PROMPT_EVAL = "1"
uv run pytest tests/test_prompt_quality_live.py -m live -s
```

## Known limitations

- Windows dependency: The PP-DocLayoutV3 worker runtime is configured only for Windows (`tools/pp_doclayout/pyproject.toml` contains `sys_platform == 'win32'`), and `src/agentic_extractor/layout.py` targets the Windows virtual environment path `.venv/Scripts/python.exe`.
- Mandatory three-engine processing: RapidOCR, PP-DocLayoutV3, and OpenAI `gpt-5.6-luna` are all required for extraction. There is no offline-only, OCR-only, or single-engine fallback mode.
- In-memory job and cache lifecycle: The FastAPI service and Streamlit app store jobs, upload data, and rendered page caches in process memory. API jobs expire after 3,600 seconds with lazy cleanup on subsequent lookups. Restarting the process discards all jobs and cached data.
- Upload constraints: Maximum upload size is 50 MiB, maximum page count is 200 pages, and maximum image resolution is 25,000,000 pixels. Multi-frame TIFFs are rejected.
- Security model: The application and API run on `127.0.0.1` for local use. They include no authentication, authorization, or rate limiting.
- Document chat retrieval: Document chat uses local lexical ranking rather than semantic vector embeddings. Questions with vocabulary that diverges significantly from the generated Markdown text may fail to retrieve relevant passages.
- Heuristic scoring: Checkbox detection, redaction detection, layout confidence, and image quality metrics use heuristic thresholds rather than calibrated probability models.

## Documentation index

- [Architecture](docs/ARCHITECTURE.md): Data flow diagrams, module responsibilities, state lifecycle, and external systems.
- [Technical reference](docs/TECHNICAL.md): Stack choices, system invariants, error handling protocols, and persistence paths.
- [Runbook](docs/RUNBOOK.md): Startup, shutdown, process management, diagnostics, and failure recovery.
- [Contributing guide](CONTRIBUTING.md): Standards for code, tests, documentation, and pull requests.
- [Local API specification](docs/API.md): Endpoint descriptions, request and response contracts, and error structures.
- [Configuration guide](docs/CONFIGURATION.md): Complete list of application settings and runtime environments.
- [Context engineering](docs/CONTEXT-ENGINEERING.md): Prompt packaging, visual crop routing, and Luna context contracts.
- [Development guide](docs/DEVELOPMENT.md): Detailed local development workflows and commands.
- [Getting started](docs/GETTING-STARTED.md): Step-by-step onboarding walkthrough.
- [Testing reference](docs/TESTING.md): Test suite organization and test fixture descriptions.
