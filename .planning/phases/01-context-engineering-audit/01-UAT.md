---
status: diagnosed
phase: 01-context-engineering-audit
source:
  - 01-01-SUMMARY.md
started: 2026-08-30T10:18:00Z
updated: 2026-08-30T10:18:00Z
---

# Phase 1 Acceptance Checks

## Current Test

[testing complete]

## Tests

### 1. Full unit suite

expected: All unit and integration tests pass at the configured coverage threshold.
result: pass
evidence: `uv run pytest -q` — 134 passed, 2 skipped, 91.73% coverage.

### 2. Dual-engine, artifact, usage, and API regression suite

expected: Mandatory RapidOCR/Luna behavior, artifacts, pricing, and local API contracts pass.
result: pass
evidence: Focused suite — 49 passed.

### 3. Streamlit local health

expected: The running app responds on TCP port 8841.
result: pass
evidence: `http://127.0.0.1:8841/_stcore/health` — HTTP 200 `ok`.

### 4. Ruff lint

expected: Ruff reports no lint violations.
result: pass
evidence: `uv run ruff check .` — passed.

### 5. Ruff formatting

expected: Every tracked Python file matches the configured formatter.
result: issue
reported: "ruff format --check reports five files would be reformatted"
severity: minor

### 6. Static typing

expected: ty reports no diagnostics.
result: issue
reported: "ty reports two unsafe subscripts of object-typed document metadata in hybrid pipeline tests"
severity: minor

## Summary

total: 6
passed: 4
issues: 2
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "All Python files satisfy the configured Ruff formatter"
  status: failed
  reason: "`uv run ruff format --check .` identifies five mechanically unformatted files."
  severity: minor
  test: 5
  root_cause: "Existing formatting drift in app_pages/extract.py, artifacts.py, costs.py, pipeline.py, and tests/test_api.py."
  artifacts:
    - path: "app_pages/extract.py"
      issue: "Ruff formatter drift at line 27."
    - path: "src/agentic_extractor/artifacts.py"
      issue: "Ruff formatter drift near lines 283 and 293."
    - path: "src/agentic_extractor/costs.py"
      issue: "Ruff formatter drift near line 81."
    - path: "src/agentic_extractor/pipeline.py"
      issue: "Ruff formatter drift near line 173."
    - path: "tests/test_api.py"
      issue: "Ruff formatter drift near lines 264 and 379."
  missing:
    - "Apply the configured Ruff formatter only to the five reported files."
  debug_session: ""

- truth: "The project passes its configured ty static-type check"
  status: failed
  reason: "`uv run ty check` reports two not-subscriptable diagnostics."
  severity: minor
  test: 6
  root_cause: "Two assertions in tests/test_hybrid_pipeline.py subscript a value typed as object without narrowing it first."
  artifacts:
    - path: "tests/test_hybrid_pipeline.py"
      issue: "Unsafe nested subscripts at lines 203 and 233."
  missing:
    - "Narrow or cast the metadata value before asserting nested review status."
  debug_session: ""
