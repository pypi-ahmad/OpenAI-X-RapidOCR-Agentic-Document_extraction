---
phase: 01-context-engineering-audit
reviewed: 2026-08-30T10:12:15Z
depth: deep
files_reviewed: 19
files_reviewed_list:
  - AGENTS.md
  - README.md
  - docs/CONFIGURATION.md
  - docs/CONTEXT-ENGINEERING.md
  - docs/DEVELOPMENT.md
  - docs/TESTING.md
  - docs/architecture.md
  - src/agentic_extractor/openai_refiner.py
  - src/agentic_extractor/prompts/block-context-compact.md
  - src/agentic_extractor/prompts/block-context-full.md
  - src/agentic_extractor/prompts/markdown-workflow.md
  - src/agentic_extractor/prompts/page-context-compact.md
  - src/agentic_extractor/prompts/page-context-full.md
  - src/agentic_extractor/prompts/refinement.md
  - streamlit_app.py
  - tests/test_local_artifacts.py
  - tests/test_openai_refiner.py
  - tests/test_prompt_resources.py
  - tests/test_ui_state.py
findings:
  critical: 1
  warning: 1
  info: 0
  total: 2
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-08-30T10:12:15Z
**Depth:** deep
**Files Reviewed:** 19
**Status:** issues_found

## Summary

The review traced Parse context construction, batching, OpenAI request composition, usage
aggregation, Streamlit telemetry, artifact manifest propagation, and the focused tests. The
provider boundary correctly keeps the fixed model/effort policy, sends every batch page image at
high detail, records context telemetry, and preserves telemetry through usage aggregation and the
manifest. Focused tests passed (`35 passed`).

The compact row path does not satisfy the Phase 1 lossless-context contract: it removes polygon
geometry and changes numeric evidence through rounding. The only test named as a lossless
round-trip exercises the full row, allowing the Balanced compact path to regress undetected.

## Critical Issues

### CR-01: Balanced compact rows discard and alter OCR evidence

**Classification:** BLOCKER

**File:** `src/agentic_extractor/openai_refiner.py:45,921-927`

**Related:** `docs/CONTEXT-ENGINEERING.md:3,13-16,56-58`; `docs/architecture.md:218-219`

**Issue:** `_FULL_BLOCK_COLUMNS` is the only schema containing `polygon`, while the compact row
omits that source geometry entirely. The compact branch also rounds `bbox` values to four decimal
places and OCR confidence to three decimal places before serialization. Consequently, an
unrouted Balanced page does not deliver the same evidence-bearing block data that RapidOCR
produced: polygon evidence is absent and the remaining geometry/confidence values are modified.
This contradicts CTX-01 and the submitted maintenance rule that context optimization must not
remove evidence fields. It also makes the architecture's “lossless row serialization” claim
incorrect. A page image does not repair this provenance loss because GPT can no longer cite or
compare the original polygon and exact confidence supplied by RapidOCR.

**Fix:** Define one canonical evidence-column set that includes the original polygon and serialize
the exact `ocr_score`, `bbox`, and `polygon` values in both modes. Keep the mode distinction in
additional layout/page context rather than deleting or rounding source evidence. For example:

```python
_BLOCK_EVIDENCE_COLUMNS = [
    "id", "type", "text", "confidence", "bbox", "polygon", "requires_gpt_review"
]

row = [
    block.id,
    block.type,
    block.text,
    block.ocr_score,
    block.bbox,
    block.polygon,
    requires_gpt_review,
]
```

Update both page prompt declarations, prompt versions, documentation, and regression tests
together. If dropping polygon or rounding is an intentional product tradeoff, remove the
“lossless” acceptance claim and obtain an explicit product decision instead of presenting the
payload as lossless.

## Warnings

### WR-01: The lossless regression test exercises only the full-context branch

**Classification:** WARNING

**File:** `tests/test_openai_refiner.py:278-312`

**Issue:** `test_context_rows_are_lossless_and_smaller_than_repeated_key_objects` calls
`_block_context(block, True)` only. It verifies the full row against the source block, then checks
only that this serialization is shorter than a repeated-key object. It never exercises
`full_context=False`, never round-trips a compact row against its declared columns, and therefore
does not detect the polygon omission or numeric rounding in the mode the optimization primarily
targets. The test name and Phase 1 validation result overstate the covered behavior.

**Fix:** Parameterize the test across compact and full modes. Reconstruct a field mapping by
zipping the rendered page's declared columns to each JSON row, and assert every required source
evidence value is exactly equal for both branches. Retain a separate size assertion so losslessness
and compression are independently diagnosed.

---

_Reviewed: 2026-08-30T10:12:15Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
