<!-- generated-by: gsd-doc-writer -->
# Configuration

The application is configured with process environment variables for the OpenAI
client and with the checked-in Streamlit configuration in
`.streamlit/config.toml`. Most resource limits and model policy are code-level
defaults in `src/agentic_extractor/config.py`. The environment exposes the OpenAI
connection, a bounded CPU OCR worker cap, and optional Windows CUDA Toolkit
discovery; it does not expose model, pricing, cache-size, or layout-model overrides.

## Environment variables

Set application variables in the process that starts Streamlit or the FastAPI
server. The UI checks whether the API key is present, and processing performs a
model-access preflight before RapidOCR starts. The application does not display
or write the OpenAI credential.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Required for extraction and document chat | None | OpenAI credential used to create the client and validate access to the configured model. Extraction stops before OCR if it is absent or unusable; document chat is disabled when it is absent. |
| `OPENAI_BASE_URL` | Optional | Unset (`None`) | Overrides the base URL passed to the OpenAI client. Leave it unset to use the client's standard endpoint. Any custom endpoint must support the configured model and request contract. <!-- VERIFY: Confirm compatibility of any external custom endpoint before use. --> |
| `ADE_OCR_MAX_WORKERS` | Optional | `4` | Maximum concurrent local CPU page workers, from `1` through `4`, for quality/preprocessing and CPU RapidOCR. OCR workers are additionally limited to one per four logical CPUs. CUDA OCR remains serial. |
| `CUDA_PATH` | Optional; Windows RapidOCR only | Unset | CUDA Toolkit root used to register `bin\x64` or `bin` DLLs before ONNX Runtime device detection. It helps discovery but does not force CUDA. PP-DocLayoutV3 uses CUDA runtime packages from its own uv environment instead. |
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

## Runtime environments

The application and PP-DocLayoutV3 use separate uv environments. Synchronize
both from the repository root:

```powershell
uv sync --all-groups
uv sync --project tools/pp_doclayout --locked
```

The root `pyproject.toml` requires Python `>=3.13`, installs RapidOCR from the
pinned Git revision recorded there, and installs ONNX Runtime GPU plus the
application and development dependencies. The isolated
`tools/pp_doclayout/pyproject.toml` is Windows-only, requires Python
`>=3.13,<3.14`, and installs PaddleX `3.4.0`, `paddlepaddle-gpu` `3.2.0`, and
packaged CUDA 12 runtime libraries. Its interpreter is expected at
`tools/pp_doclayout/.venv/Scripts/python.exe`.

## Required vs optional settings

`OPENAI_API_KEY` is required to perform an extraction or use document chat.
When extraction is requested, the Streamlit and API paths construct an OpenAI
client and validate access to the configured model before RapidOCR runs. A
missing or invalid key blocks processing with an actionable configuration error.
Document chat uses the same client configuration but reads only processed
Markdown registered in the current Streamlit session. `OPENAI_BASE_URL` is
optional and becomes `None` when not set.

`RUN_LIVE_PROMPT_EVAL` is optional and affects only the live prompt evaluation
tests. It does not change application startup or extraction behavior.

RapidOCR is also required at runtime, but it is installed as a Python
dependency rather than enabled by an environment variable. `pyproject.toml`
installs `onnxruntime-gpu` and pins RapidOCR to a specific upstream Git revision.
The OCR adapter prefers CUDA. It requests CUDA when ONNX Runtime reports a CUDA
execution-provider device, or when `CUDAExecutionProvider` is available and
ONNX Runtime identifies the device as a GPU. If CUDA initialization fails or no
usable CUDA device is exposed, it creates a CPU engine and records a warning.
After the first OCR call initializes the sessions, the adapter inspects the
detector's active providers; if `CUDAExecutionProvider` is absent, provenance is
changed to CPU. `CUDA_PATH` only assists DLL discovery on Windows. There is no
environment-variable switch that forces either OCR device.

PP-DocLayoutV3 is required for every extraction and runs in a separate locked environment:

```powershell
uv sync --project tools/pp_doclayout --locked
```

The worker requires Python 3.13 and locks PaddleX 3.4.0, `paddlepaddle-gpu` 3.2.0 from
Paddle's CUDA 12.6 package index, model `PP-DocLayoutV3`, and a `0.5` detection
threshold. It registers the CUDA runtime DLLs installed in its own uv environment
and proves the selected device with a real inference probe. When CUDA is not
available, a successful CPU probe is accepted with a warning. When CUDA is
detected but GPU model initialization falls back to CPU, the parent closes and
retries the worker once; a second GPU failure blocks extraction rather than
caching that fallback. Failure to initialize on an available device also blocks
extraction. There is no alternate layout model.

GPU layout requests use batches of two pages; CPU requests use one page at a
time. Table regions scoring at least `0.85` additionally use
`PP-LCNet_x1_0_table_cls`, followed by `SLANeXt_wired` or `SLANet_plus`. These
table models initialize lazily and do not introduce a second OCR engine.

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
| `cloud_batch_characters` | `80,000` | Rendered evidence-character budget for packed compact-page refinement requests. Full/high-resolution pages and oversized compact pages are isolated. Evidence is never truncated. |
| OpenAI request retries | `2` | Maximum automatic retries configured on the shared OpenAI client. |
| OpenAI request timeout | `120` seconds | Timeout configured on the shared OpenAI client. |
| High Accuracy block review threshold | `0.85` | Fixed code policy: OCR blocks with a score strictly below `0.85` require a grounded refinement outcome in High Accuracy mode. A score equal to `0.85` is not below the threshold. This is not user-configurable. |

## Process caches

Rendered pages, successful RapidOCR results, PP-DocLayoutV3 results, and table
structures use four independent, process-memory least-recently-used caches. Each
cache is fixed at 128 MiB and 256 entries. These limits are not configurable.

Cache keys include exact RGB page or crop hashes and the relevant render, engine,
model, device, version, and threshold signatures. Failed OCR pages are not
cached, and an injected OCR engine without an identifiable cache namespace
bypasses the OCR cache. Cache data is not written to disk, survives the
Streamlit session Reset action, and is lost when the server process exits.
Cache hits, misses, and avoided OCR time are included in processing diagnostics.

## Bounded concurrency

`ADE_OCR_MAX_WORKERS` controls page preparation and CPU RapidOCR concurrency:

- Page quality analysis and preprocessing use at most the selected page count,
  the configured worker cap, and the number of logical CPUs.
- CPU RapidOCR uses dedicated engine instances. Its worker count is at most the
  number of cache misses, `ADE_OCR_MAX_WORKERS`, and one worker per four logical
  CPUs. ONNX Runtime CPU threads are divided across the active workers.
- CUDA RapidOCR is serialized to one worker. An OCR resource also locks its warm
  worker pool so two documents do not share the same engines concurrently.
- PP-DocLayoutV3 serializes worker requests. Its internal page batch is two on
  GPU and one on CPU.

The configured cap must parse as an integer from `1` through `4`; otherwise the
application raises `ADE_OCR_MAX_WORKERS must be an integer from 1 through 4.` or
`ADE_OCR_MAX_WORKERS must be between 1 and 4.` when the setting is read.

## Per-stage timing

Every Parse result carries wall-clock timing data in `result.timings`. The
instrumented stages cover:

- OpenAI preflight; RapidOCR and layout initialization;
- document ingest, page preparation, RapidOCR wall time, and bounded OCR retry;
- layout detection, table structure, evidence routing, checkbox and redaction
  detection, and visual-review planning;
- GPT refinement, grounded merge, optional Markdown workflows, checkbox
  verification, deterministic validation, and bounded object repair; and
- canonical result and core artifact finalization plus on-demand annotated PDF,
  HTML, and ZIP generation.

The Diagnostics page ignores non-finite, negative, and zero stage values, sorts
the remaining stages slowest-first, and reports each stage's share of measured
stage time. It also retains the separate pipeline total, because the sum of
instrumented stages is diagnostic telemetry rather than a token, price, or
guaranteed end-to-end accounting measure.

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
- `PP-DocLayoutV3 runtime installed` means only that the isolated lockfile and
  expected Python executable exist. Processing still initializes and probes the
  model, then records its actual device, Paddle versions, and layout/table
  timings. Table models initialize only when a qualifying table region is found.
  <!-- VERIFY: Ensure PP-DocLayoutV3 and table-model assets are available in environments without network access. -->

Document chat requires OpenAI configuration but consumes only previously generated
Markdown. It does not initialize RapidOCR or read the original uploaded document.

## Per-environment overrides

No `.env`, `.env.development`, `.env.production`, or `.env.test` files are
included, and the code does not branch on an environment name. Configure each
environment with process-level variables before startup. At minimum, extraction
and chat need `OPENAI_API_KEY`; add `OPENAI_BASE_URL`, `ADE_OCR_MAX_WORKERS`, or
Windows `CUDA_PATH` only when their documented overrides are required.

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

The optional FastAPI surface has no checked-in server configuration or automatic
startup. Run it separately on loopback port `8842`:

```powershell
uv run uvicorn agentic_extractor.api:app --host 127.0.0.1 --port 8842
```

Both server commands bind their user-facing surfaces to `127.0.0.1`. The API is
designed for trusted local-machine use and does not implement authentication or
rate limiting; exposing it on a non-loopback address is outside the supported
configuration. OpenAI requests remain external even though the UI, API,
RapidOCR, PP-DocLayoutV3, caches, and job state run locally.
