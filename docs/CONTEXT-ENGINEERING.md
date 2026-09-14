<!-- generated-by: gsd-doc-writer -->
# Context engineering

The application preserves canonical evidence while bounding the context sent to models. Parse
reconciliation may sample or truncate model-facing block context, and document chat retrieves
only relevant generated-Markdown excerpts; neither operation mutates the stored canonical OCR
evidence. Accuracy and auditability take priority over token reduction.

## Runtime context contract

RapidOCR remains the source of local text, geometry, confidence, and initial layout. Each Parse
request sends a low-detail overview of every selected page to GPT-5.6-luna unless a page-wide
high-detail region replaces that overview. Other locally planned uncertainty regions are attached
at high detail. Balanced uses compact OCR rows by default and full rows for routed uncertainty or
complexity; High Accuracy uses full rows for every selected page.

Block columns are declared once per page and values are serialized as deterministic JSON rows.
Both compact and full rows retain block ID, type, text, exact confidence, normalized bounding box,
OCR polygon, and the low-confidence review flag. Full page context additionally retains page status,
warnings, and layout signals. Values within each serialized block row are not rounded or rewritten;
later bounded calls may select fewer blocks or truncate copied text without changing canonical evidence.

Optional Classify, Section, Split, and Extract calls receive the canonical refined Markdown plus a
lossless grounding table containing page, block ID, chunk ID, type, confidence, bounding box, and
raw source text. This deliberate duplication lets GPT cite exact raw evidence after Markdown
structure has been refined.

## Prompt order and batching

The stable `policy.md` resource is sent as OpenAI instructions. Capability rules precede dynamic
configuration and document evidence in user prompts so repeated prefixes remain cache-friendly.
Reusable instructions and prompt templates live in versioned Markdown prompt resources. Model-facing
structured-output constraints also live in Pydantic response-schema field descriptions in
`openai_refiner.py`.

Balanced pages needing only compact evidence are collected in source order and greedily packed
under `cloud_batch_characters`, even when full-review pages occur between them. Only full-context
pages remain single-page requests; compact pages with localized high-resolution crops may stay in
packed batches. A compact page that exceeds the rendered evidence-character limit is isolated and
never truncated. Canonical Markdown is rebuilt in source page order after all grounded proposals
are validated. Image tokens and provider tokenization are not inferred from character counts.

## Telemetry

Every usage call may include a `context` object:

| Field | Meaning |
| --- | --- |
| `kind` | Parse, grounded Markdown, repair, reconciliation, checkbox verification, or document-chat context |
| `prompt_characters` | Complete rendered text prompt length |
| `evidence_characters` | Dynamic evidence length used for batching and diagnostics |
| `source_text_characters` | Sum of source OCR text characters represented by Parse/workflow calls; for document chat, retrieved generated-Markdown excerpt characters |
| `block_count` | Evidence block or checkbox-candidate count; for document chat, retrieved excerpt count |
| `compact_pages` | Pages represented with compact OCR rows |
| `full_context_pages` | Pages represented with full OCR/layout rows |
| `batch_kind` | `compact` for a packed compact-evidence request or `full` for an isolated full-review page |
| `batch_index` / `batch_count` | One-based request position and total primary refinement requests |
| `overview_pages` | Pages supplied as low-detail visual overviews |
| `high_resolution_regions` | Grounded high-detail crops, reasons, bounds, and source IDs |
| `high_resolution_region_count` | Number of high-detail crops in the call |

The export manifest records usage calls completed before artifact generation. Document-chat calls
happen afterward, remain in session usage history, and are not retroactively added to an existing
artifact manifest. The Parse Usage & Cost panel exposes context kind, prompt and evidence
characters, block count, compact pages, and full-context pages alongside provider-reported tokens
and costs; `source_text_characters` remains manifest-only. The chat page instead shows summary call,
total-token, and total-cost metrics. Character counts are exact diagnostics but are never presented
as token or price estimates.

## Maintenance rules

- Preserve the column order and update its prompt declaration and tests together.
- Keep dynamic document content after the stable instruction prefix.
- Prefer one combined downstream capability call over duplicate calls.
- Measure serialization changes with deterministic fixtures and verify every source field survives.
- Do not reduce image coverage or evidence fields without an explicit product decision and an
  accuracy evaluation.
