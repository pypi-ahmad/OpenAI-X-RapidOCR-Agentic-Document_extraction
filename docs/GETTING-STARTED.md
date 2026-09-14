<!-- generated-by: gsd-doc-writer -->
# Getting started

Run the Agentic Document Extractor locally to turn PDF and image documents into
grounded Markdown, structured results, and downloadable artifacts. Every
successful extraction uses RapidOCR, PP-DocLayoutV3, and then OpenAI
`gpt-5.6-luna`.

## Prerequisites

- Windows. The required PP-DocLayoutV3 worker uses a Windows-only uv
  environment and expects its interpreter under `tools/pp_doclayout/.venv/Scripts`.
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management.
- Python 3.13. The root project requires Python `>=3.13`, the layout worker
  requires Python `>=3.13,<3.14`, and `.python-version` pins `3.13`.
- An `OPENAI_API_KEY` for an account with access to `gpt-5.6-luna`.
- An NVIDIA GPU is optional. RapidOCR and PP-DocLayoutV3 prefer a usable CUDA
  device and can run on CPU when CUDA is unavailable.

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

3. Create or synchronize the application environment, including development
   tools.

   ```bash
   uv sync --all-groups
   ```

4. Create the required isolated PP-DocLayoutV3 environment from its lockfile.

   ```bash
   uv sync --project tools/pp_doclayout --locked
   ```

5. Set the API key in the process environment. Do not add its value to source
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
or single-frame TIFF document, select a mode, then choose **Extract document**.
The application accepts uploads up to 50 MiB.

On Windows, [`run_app.cmd`](../run_app.cmd) is a convenience launcher. It:

- checks that `uv` is available;
- finds and prints listener PIDs on TCP port `8841`;
- revalidates each PID before stopping it;
- confirms the port is free;
- starts Streamlit in the foreground so logs remain visible; and
- keeps the console open after an error or exit.

Because the launcher reclaims port `8841`, close any unrelated application using that port before launching if it should not be stopped.

## First extraction

1. Open **Diagnostics** and confirm that the OpenAI key is configured, RapidOCR
   is installed, and the PP-DocLayoutV3 runtime is installed. The **Extract
   document** button remains disabled when any required local readiness check
   fails. Model access is validated after you start processing; there is no
   reduced-engine fallback.
2. Upload a supported PDF or image. File signatures and readability are checked,
   so changing an unsupported file's extension does not make it valid.
3. For a PDF, choose an inclusive range satisfying
   `1 <= start_page <= end_page <= total_pages`. The initial range selects the
   full document. Images are treated as one page and do not show irrelevant
   range controls.
4. Choose **Balanced** for compact grounded OCR/layout context, or **High
   Accuracy** for full relevant OCR evidence. Pages without a page-wide review
   region receive a low-detail overview; locally identified uncertainty receives
   high-detail crops, and a page-wide high-detail region replaces that overview.
   In High Accuracy, each RapidOCR block
   with recognition confidence strictly below `0.85` must receive a grounded,
   accepted confirmation or correction from the existing GPT page-refinement
   request. Missing, rejected, or abstained outcomes make the run
   `REVIEW_REQUIRED`; this check does not add another GPT call and is not
   enforced in Balanced mode. Both modes always call GPT with medium reasoning
   effort.
5. Parse is always enabled. Optionally enable **Classify**, **Section**,
   **Split**, or **Extract**. Classify requires an allowlist; split overrides
   require a reason; and Extract accepts guided fields, pasted or uploaded JSON
   Schema, or pasted or uploaded Markdown field definitions.
6. Leave **Atomic grounding** enabled to include LandingAI-shaped
   `atomic_grounding` parts arrays in the structured Parse JSON. Disabling it
   omits those arrays without removing node-level page, range, or box grounding.
7. Choose **Extract document**.
8. Follow the progress percentage and review any warnings or failed pages. A
   completed run exposes rendered and raw Markdown, grounded blocks, workflow
   results, usage, and artifact downloads.

The pipeline loads the full upload to validate and inspect the document, then
limits OCR, layout analysis, GPT processing, and generated artifacts to the
selected pages. Markdown and canonical Parse JSON are ready when processing
finishes. The annotated PDF is generated when its result tab is opened; the
source-faithful HTML view and ZIP bundle are generated when you choose their
prepare actions. The ZIP includes an export manifest and any generated checkbox
crops.
Review the source preview and page-quality diagnostics before a paid run when
scan quality is uncertain.

Optional workflows consume the canonical GPT-refined Markdown and its grounding
index; they do not run OCR again. Their results appear on the separate
**Classify**, **Section**, **Split**, and **Extract** pages. Uncertain or
unsupported results remain review-required or abstained instead of being marked
verified.

## Document chat

After at least one Parse completes, open **Chat** from the top navigation. The
newest processed document is selected initially; you can place up to 12
session-processed documents in scope. Chat history is stored per selected
document scope. Switching scope loads its saved history, or starts empty for a
new scope, so context is not mixed across scopes.

Chat sends Luna only retrieved excerpts from the selected documents' generated
Markdown and up to six recent visible conversation messages. It never sends the
original upload, page images, OCR objects, or raw document bytes. Answers cite
the excerpt IDs shown under the response. Off-topic requests are redirected to
the selected documents, and unsupported answers report insufficient evidence.

Processed chat sources and messages live only in Streamlit session state. A
browser-session reset or application restart removes them; process the document
again to restore it to the chat selector.

## Diagnostics

Open **Diagnostics** before a first extraction to check the OpenAI configuration,
RapidOCR package, and isolated PP-DocLayoutV3 runtime without exposing secret
values. After processing a document, the page also shows deduplicated warnings,
failed pages, per-stage timing, the slowest measured stage, and RapidOCR runtime
details for the current result.

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

### PP-DocLayoutV3 is unavailable or cannot initialize

Synchronize the isolated worker environment from the repository root:

```powershell
uv sync --project tools/pp_doclayout --locked
```

Restart the application after synchronization. The worker prefers GPU and can
use CPU when CUDA is unavailable. If it detects a CUDA device but GPU model
initialization repeatedly fails, verify the NVIDIA driver before retrying.

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
within the configured 25,000,000-pixel limit. Multi-frame TIFF images are not
accepted; convert them to PDF first.

### A run requires review

`REVIEW_REQUIRED` is a successful safety outcome when evidence, validation, or
confidence is insufficient for automatic acceptance. Inspect field provenance,
validation results, abstention reasons, and source overlays instead of treating
the value as verified.

### Chat is disabled or no documents are listed

Complete a Parse in the current Streamlit session first and keep at least one
processed document selected on the **Chat** page. Chat also requires
`OPENAI_API_KEY` in the Streamlit process environment. Restarting the app or
resetting the session clears the session-only processed-document registry.

## Next steps

- Return to the [project README](../README.md) for features, artifacts, and
  limitations.
- Read the [architecture guide](ARCHITECTURE.md) for the pipeline and evidence
  boundary.
- Review [configuration](CONFIGURATION.md) for application settings.
- Use the [local API guide](API.md) for typed programmatic access on port `8842`.
- See the [development guide](DEVELOPMENT.md) for local workflow, commands, and
  contribution conventions.
- Follow the [testing guide](TESTING.md) before changing the implementation. The
  default tests use mocked OpenAI responses and do not make paid API calls.
