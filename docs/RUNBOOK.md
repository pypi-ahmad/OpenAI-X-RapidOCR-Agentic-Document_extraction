# Runbook

This runbook provides operational instructions for starting, stopping, diagnosing, and troubleshooting the local Streamlit user interface and the local FastAPI service.

## Starting services

### Streamlit web interface (TCP port 8841)

Choose one of three start methods:

1. Direct execution via `uv`:

   ```powershell
   uv run streamlit run app.py --server.port 8841
   ```

2. Console script entry point:

   ```powershell
   uv run agentic-extractor
   ```

3. Windows batch launcher:

   ```cmd
   run_app.cmd
   ```

   `run_app.cmd` verifies that `uv` is available on `PATH`, identifies and terminates any verified process already listening on TCP port `8841`, and starts Streamlit in the foreground so console output remains visible.

### Local HTTP API (TCP port 8842)

The FastAPI service is started independently:

```powershell
uv run uvicorn agentic_extractor.api:app --host 127.0.0.1 --port 8842
```

Both services require `OPENAI_API_KEY` to be set in the process environment before launch. Neither service starts the other, and they do not share in-memory state or process caches.

## Stopping services

- **Streamlit**: Press `Ctrl+C` in the running terminal, or close the console window opened by `run_app.cmd`.
- **FastAPI**: Press `Ctrl+C` in the Uvicorn terminal window.
- **State on termination**: All in-process state—including active jobs in `_Job`, rendered page caches in `cache.py`, and Streamlit session state in `st.session_state`—is discarded when the process exits. No graceful shutdown flush is required.

## Logs and output location

- **Standard console streams**: Neither the application nor the API writes log files to disk. All log messages, warnings, and error tracebacks are printed directly to `stdout` and `stderr` in the launching terminal.
- **Capturing console logs**: If log persistence is needed, redirect terminal output in PowerShell:

  ```powershell
  uv run streamlit run app.py --server.port 8841 2>&1 | Tee-Object -FilePath app.log
  ```

- **In-app diagnostics**: The Streamlit **Diagnostics** page (`app_pages/diagnostics.py`) provides real-time information on model readiness, stage durations, bottleneck identification, and page-level warnings.
- **User downloads**: The only files written to disk are those explicitly requested and saved by the user via the browser download buttons or the API artifact download endpoints (`/api/v1/jobs/{job_id}/artifacts/{artifact_name}`).

## Common failures and troubleshooting

| Symptom / error string | Root cause | Remediation |
|---|---|---|
| `OpenAI is not configured. Add OPENAI_API_KEY to the launcher environment...` | `OPENAI_API_KEY` environment variable is missing, empty, or unreadable in the launching process. | Set `$env:OPENAI_API_KEY = "your-key"` in the terminal and restart the service. |
| `OpenAIConfigurationError: ...` | API key is invalid or lacks access permissions for model `gpt-5.6-luna`. | Verify key validity and ensure the OpenAI account has active access to `gpt-5.6-luna`. |
| `PP-DocLayoutV3 is unavailable. Run 'uv sync --project tools/pp_doclayout --locked' and retry.` | The isolated layout worker virtual environment at `tools/pp_doclayout/.venv` is missing or incomplete. | Run `uv sync --project tools/pp_doclayout --locked` from the repository root. |
| `The installed worker returned an incompatible V3 result contract` (`LayoutContractError`) | The worker process crashed, timed out, or returned malformed JSON over the subprocess pipe. | Inspect terminal logs for PaddleX traceback; ensure NVIDIA CUDA 12 drivers or CPU libraries are functioning properly. |
| `RapidOCR is unavailable. Run 'uv sync --all-groups', then verify RapidOCR and ONNX Runtime.` | RapidOCR or ONNX Runtime failed during module initialization. | Run `uv sync --all-groups` to refresh installed wheels and check ONNX Runtime DLL dependencies. |
| `TCP port 8841 is still occupied by PID(s)...` | A previous Streamlit instance or other application is bound to port `8841`. | Use `run_app.cmd` to terminate existing listeners, or manually check `Get-NetTCPConnection -LocalPort 8841 -State Listen` and stop the process. |
| `HTTP 404 job_not_found` | The requested job ID does not exist, or the 3600-second job TTL has elapsed. | Resubmit the document using `POST /api/v1/jobs/parse`. |
| `HTTP 409 parse_unavailable` | Schema extraction (`POST /api/v1/jobs/{job_id}/extract`) was called before Parse processing succeeded. | Ensure the document is parsed successfully before requesting schema extraction. |
| `HTTP 404 artifact_not_found` | The requested artifact name is not in the allowlist (`document.md`, `parse-result.json`, `annotated.pdf`, `document.html`, `bundle.zip`). | Request one of the supported artifact names listed above. |
| `Decoded document is empty or exceeds the 50 MiB limit` (`IngestError`) | Uploaded file size exceeds the 50 MiB limit in `src/agentic_extractor/config.py`. | Reduce file size or crop pages before upload. |
| `PDF exceeds the 200 page limit` (`IngestError`) | Uploaded PDF has more than 200 pages. | Use page selection controls to process an inclusive range of up to 200 pages. |
| Workflow state is `REVIEW_REQUIRED` | Safety gate triggered due to low OCR confidence (`<0.85`), checkbox disagreement, or validation rule mismatch. | Expected behavior. Inspect the review items in the UI or in `job.workflow.review_items`. |

## Health verification

1. **Streamlit interface**: Load `http://127.0.0.1:8841` in a browser. The sidebar should indicate system status without displaying an `OpenAI is not configured` banner.
2. **FastAPI interface**: Navigate to `http://127.0.0.1:8842/docs` to verify the OpenAPI Swagger interface, or execute:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8842/openapi.json
   ```

3. **Inference engine readiness**: Open the **Diagnostics** page in the Streamlit application to check detector and model readiness before starting extraction runs.
