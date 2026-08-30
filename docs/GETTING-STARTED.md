<!-- generated-by: gsd-doc-writer -->
# Getting started

Run the Agentic Document Extractor locally to turn PDF and image documents into
grounded Markdown, structured results, and downloadable artifacts. Every
successful extraction uses RapidOCR followed by OpenAI `gpt-5.6-luna`.

## Prerequisites

- Windows or Linux. The locked `onnxruntime-gpu` dependency does not provide a
  macOS wheel.
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management.
- Python `>=3.13`; `uv` manages the project interpreter (the repository pins
  Python 3.13 in `.python-version`).
- An `OPENAI_API_KEY` for an account with access to `gpt-5.6-luna`.
- An NVIDIA GPU is optional. The adapter requests CUDA only when ONNX Runtime
  reports a usable CUDA execution-provider device; otherwise RapidOCR uses CPU.

The supported v1 uploads are PDF, PNG, JPG/JPEG, and TIFF where Pillow can
decode the source reliably. Documents are limited to 50 MiB and 200 pages.

## Installation

1. Clone the public GitHub repository.

   ```bash
   git clone https://github.com/pypi-ahmad/OpenAI-X-RapidOCR-Agentic-Document_extraction.git
   ```

2. Change into the project directory.

   ```bash
   cd OpenAI-X-RapidOCR-Agentic-Document_extraction
   ```

3. Create or synchronize the local environment, including development tools.

   ```bash
   uv sync --all-groups
   ```

4. Set the API key in the process environment. Do not add its value to source
   files.

   ```powershell
   $env:OPENAI_API_KEY = "<your key>"
   ```

`OPENAI_BASE_URL` is optional and normally remains unset. Environment variables
must be available to the process that starts Streamlit. The application checks
configuration status without displaying or writing secret values.

## First run

Start the Streamlit application:

```powershell
uv run streamlit run app.py --server.port 8841
```

Open [http://127.0.0.1:8841](http://127.0.0.1:8841), upload a PDF, PNG, JPEG,
or TIFF document, select a mode, then choose **Extract document**. The
application accepts uploads up to 50 MiB.

On Windows, [`../run_app.cmd`](../run_app.cmd) is the recommended convenience
launcher. It:

- checks that `uv` is available;
- finds and prints unique listener PIDs specifically on TCP port `8841`;
- revalidates each PID before stopping it;
- confirms the port is free;
- starts Streamlit in the foreground so logs remain visible; and
- keeps the console open after either an error or normal exit.

Because the launcher is explicitly designed to reclaim port `8841`, close an
unrelated application using that port before launching if it must not be
stopped.

## First extraction

1. Confirm that the sidebar reports both RapidOCR and OpenAI as available.
   Missing or invalid configuration blocks processing; there is no OCR-only
   fallback.
2. Upload a supported PDF or image. File signatures and readability are checked,
   so changing an unsupported file's extension does not make it valid.
3. For a PDF, choose an inclusive range satisfying
   `1 <= start_page <= end_page <= total_pages`. The initial range selects the
   full document. Images are treated as one page and do not show irrelevant
   range controls.
4. Choose **Balanced** for compact grounded context with targeted visual review,
   or **High Accuracy** for image and full-evidence review of every selected
   page. In High Accuracy, each RapidOCR block with recognition confidence
   strictly below `0.85` must receive a grounded, accepted confirmation or
   correction from the existing GPT page-refinement request. Missing, rejected,
   or abstained outcomes make the run `REVIEW_REQUIRED`; this check does not add
   another GPT call and is not enforced in Balanced mode. Both modes always call
   GPT with medium reasoning effort.
5. Optionally configure classification, splitting, or extraction fields, then
   choose **Extract document**.
6. Follow the progress percentage and review any warnings or failed pages. A
   completed run exposes rendered and raw Markdown, grounded blocks, workflow
   results, usage, and artifact downloads.

Only selected pages are sent to the extraction pipeline and included in
artifacts. Review the source preview and page-quality diagnostics before a paid
run when scan quality is uncertain.

## Common setup issues

### Extraction is disabled or says OpenAI is not configured

Set `OPENAI_API_KEY` in the shell that launches Streamlit, then restart the
application. The key must be available to the running process; the application
does not read it from source files.

If the key exists but model validation fails, verify that the credential is
valid and can access `gpt-5.6-luna`. Configure `OPENAI_BASE_URL` only when a
compatible endpoint explicitly requires it.

### RapidOCR is unavailable or cannot initialize

Synchronize dependencies again:

```powershell
uv sync --all-groups
```

RapidOCR requires the repository's pinned dependency set, including ONNX and
ONNX Runtime. CUDA is eligible only when ONNX Runtime reports a CUDA
execution-provider device (or both the CUDA provider and a GPU on older provider
APIs). If CUDA initialization fails, the adapter initializes RapidOCR on CPU.
After the first OCR call, it also verifies that the detector session actually
uses `CUDAExecutionProvider`; if not, it records CPU use and a warning. A
RapidOCR initialization failure blocks extraction rather than falling back to
GPT alone.

### Port 8841 is already in use

Use `run_app.cmd` on Windows to print, revalidate, and stop only listeners on
that port. To inspect listeners without stopping them, run:

```powershell
Get-NetTCPConnection -LocalPort 8841 -State Listen
```

Port `8842` is reserved by the project documentation for the optional local API
and should not be used as the routine Streamlit alternative.

### The upload is rejected

Confirm that the content is a readable PDF, PNG, JPEG, or TIFF rather than
trusting the filename extension. Also confirm that the decoded document is not
empty, does not exceed 50 MiB or 200 pages, and that image dimensions remain
within the configured pixel limit.

### A run requires review

`REVIEW_REQUIRED` is a successful safety outcome when evidence, validation, or
confidence is insufficient for automatic acceptance. Inspect field provenance,
validation results, abstention reasons, and source overlays instead of treating
the value as verified.

## Next steps

- Return to the [project README](../README.md) for features, artifacts, and
  limitations.
- Read the [architecture guide](architecture.md) for the pipeline and evidence
  boundary.
- Review [configuration](CONFIGURATION.md) for application settings.
- Use the [local API guide](API.md) for typed programmatic access on port `8842`.
- See the [development guide](DEVELOPMENT.md) for local workflow, commands, and
  contribution conventions.
- Follow the [testing guide](TESTING.md) before changing the implementation. The
  default tests use mocked OpenAI responses and do not make paid API calls.
