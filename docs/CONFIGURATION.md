<!-- generated-by: gsd-doc-writer -->
# Configuration

The application is configured with process environment variables for the OpenAI
client and with the checked-in Streamlit configuration in
`.streamlit/config.toml`. Resource limits and model policy are code-level
defaults in `src/agentic_extractor/config.py` and are not currently overridable
through environment variables or a project config file.

## Environment variables

Set these variables in the process that starts Streamlit or the FastAPI server.
The UI checks whether the API key is present, and processing performs a model-access
preflight before RapidOCR starts. The application does not display or write either
value.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Required for extraction | None | OpenAI credential used to create the client and validate access to the configured model. Extraction stops before OCR if it is absent or unusable. |
| `OPENAI_BASE_URL` | Optional | Unset (`None`) | Overrides the base URL passed to the OpenAI client. Leave it unset to use the client's standard endpoint. Any custom endpoint must support the configured model and request contract. <!-- VERIFY: Confirm compatibility of any external custom endpoint before use. --> |
| `RUN_LIVE_PROMPT_EVAL` | Optional; tests only | Unset | Set to `1` to enable the live prompt evaluation tests in `tests/test_prompt_quality_live.py`. Other values leave those tests skipped. |

For example, in PowerShell:

```powershell
$env:OPENAI_API_KEY = "<your key>"
uv run streamlit run app.py --server.port 8841
```

## Config file format

`.streamlit/config.toml` configures the local Streamlit server and UI theme.
The repository currently uses TOML with these top-level sections:

```toml
[server]
address = "127.0.0.1"
port = 8841
maxUploadSize = 50

[browser]
gatherUsageStats = false

[theme]
base = "dark"
```

- `[server]` binds Streamlit to the loopback address, uses port `8841`, and
  limits uploads to 50 MiB.
- `[browser]` disables Streamlit usage-stat collection.
- `[theme]` and `[theme.sidebar]` define the dark theme, colors, radii, and
  sidebar appearance.

## Required vs optional settings

`OPENAI_API_KEY` is required to perform an extraction. When processing is
requested, the Streamlit and API paths construct an OpenAI client and validate
access to the configured model before RapidOCR runs. A missing or invalid key
blocks processing with an actionable configuration error. `OPENAI_BASE_URL` is
optional and becomes `None` when not set.

`RUN_LIVE_PROMPT_EVAL` is optional and affects only the live prompt evaluation
tests. It does not change application startup or extraction behavior.

RapidOCR is also required at runtime, but it is installed as a Python
dependency rather than configured through an environment variable. `pyproject.toml`
installs `onnxruntime-gpu` and pins RapidOCR to a specific upstream Git revision.
The OCR adapter prefers CUDA by default. It requests CUDA when ONNX Runtime
reports a CUDA execution-provider device, or when `CUDAExecutionProvider` is
available and ONNX Runtime identifies the device as a GPU. If CUDA initialization
fails, it creates a CPU engine instead. After the first OCR call initializes the
model sessions, the adapter inspects the detector session's active providers; if
`CUDAExecutionProvider` is absent, it records CPU use in engine provenance. There
is no environment-variable switch for selecting the OCR device.

## Defaults

The following values are code-defined application policy, not current
environment-variable settings. Most are defined by `Settings` in
`src/agentic_extractor/config.py`; the High Accuracy threshold is defined in
`src/agentic_extractor/parse.py`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `max_upload_bytes` | `52,428,800` (50 MiB) | Maximum uploaded document size. |
| `max_pages` | `200` | Maximum document pages. |
| `max_image_pixels` | `25,000,000` | Maximum decoded pixels per image page. |
| `render_dpi` | `150` | PDF render resolution. |
| `job_ttl_seconds` | `3600` | Lifetime of an in-memory API job. Expired jobs are removed during job lookup and return `job_not_found`. |
| `model` | `gpt-5.6-luna` | Declared model policy value. The request boundary also uses `gpt-5.6-luna`. |
| `reasoning_effort` | `medium` | Declared reasoning policy value. The request boundary also uses `medium`. |
| `cloud_batch_characters` | `80,000` | Rendered evidence-character budget for each cloud-refinement batch. Oversized single pages are isolated, not truncated. |
| High Accuracy block review threshold | `0.85` | Fixed code policy: OCR blocks with a score strictly below `0.85` require a grounded refinement outcome in High Accuracy mode. A score equal to `0.85` is not below the threshold. This is not user-configurable. |

## Usage pricing

The usage dashboard and artifact manifest use the rates defined in
`src/agentic_extractor/costs.py`. These are application policy values rather than
environment-variable overrides.

| Token category | USD per 1 million tokens |
| --- | ---: |
| Uncached input | `$0.20` |
| Cached input | `$0.02` |
| Cache-write input | `$0.25` |
| Output | `$1.20` |

For a request above `272,000` input tokens, the calculator applies a `2.0x`
multiplier to all input categories and a `1.5x` multiplier to output. A request
at exactly `272,000` input tokens does not receive these multipliers. If input or
output usage is missing or inconsistent, the calculator reports the cost as
unavailable instead of fabricating a value. When cached-input or cache-write
breakdowns are missing but total input and output are valid, it labels the result
as an estimate.

## Configuration diagnostics

The Streamlit sidebar reports configuration status without showing credential
values:

- `OpenAI API configured` means only that `OPENAI_API_KEY` exists in the current
  process. Before processing, the app calls the model-access preflight; an absent
  or invalid configuration blocks extraction before RapidOCR runs.
- `RapidOCR installed` reports the installed package version without initializing
  OCR models. Engine creation still fails with an actionable setup error if the
  package cannot be imported.
- OCR provenance records the active device after model initialization. A CUDA
  warning means ONNX Runtime did not expose an active CUDA provider and RapidOCR
  is using CPU; it does not mean OCR evidence was fabricated or skipped.

Document chat requires OpenAI configuration but consumes only previously generated
Markdown. It does not initialize RapidOCR or read the original uploaded document.

## Per-environment overrides

No `.env`, `.env.development`, `.env.production`, or `.env.test` files are
included, and the code does not branch on an environment name. Configure each
environment by supplying its own process-level `OPENAI_API_KEY` and, only when
needed, `OPENAI_BASE_URL` before starting the process.

For local Streamlit use, the checked-in `.streamlit/config.toml` binds the app
to `127.0.0.1:8841`. A launch command can still pass `--server.port` to choose
a different port. For example:

```powershell
uv run streamlit run app.py --server.port 8841
```

On Windows, `run_app.cmd` uses this same port explicitly. Before starting, it
finds TCP listeners on port `8841`, prints their unique process IDs, verifies
that each process still owns a listener on that port, and force-stops only
those verified processes. Streamlit then runs in the foreground so its logs
remain visible; the launcher pauses after either an error or a normal exit.
