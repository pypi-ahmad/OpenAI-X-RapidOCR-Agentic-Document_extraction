<!-- generated-by: gsd-doc-writer -->
# Glossary

| Term | Meaning in this project |
|---|---|
| ADE | Agentic document extraction: a bounded workflow for parsing, classifying, sectioning, splitting, extracting, validating, and reviewing documents. |
| Abstention | An explicit decision not to claim a classification, refinement, or field value when support is insufficient; it is retained with a reason instead of being treated as verified. |
| Artifact | A selected-page output: Markdown, Parse JSON, annotated PDF, coordinate HTML, manifest, or ZIP bundle. |
| Balanced | Dual-engine mode using compact OCR context, low-detail page overviews, and high-detail crops only for uncertain regions. |
| Block | A page-grounded unit of OCR text with type, raw confidence, and optional geometry; its position in the page's ordered block list represents reconstructed reading order. |
| Canonical Parse result | `LocalParseResult`, the evidence-bearing input to workflow evaluation and artifact generation. |
| Chunk ID | The stable identifier of a layout chunk that groups one or more source blocks and can ground sections or extracted fields. |
| Cloud refinement | Required GPT validation and correction performed after RapidOCR. It is not an independent source of truth. |
| Evidence | A citation to an existing OCR block or chunk, or to a normalized region on an image that GPT received. |
| Grounding | The page, block ID, quote, bounding box, or polygon that connects an output to its source. |
| High Accuracy | Dual-engine mode using full relevant OCR evidence, low-detail page overviews, and high-detail crops only for uncertain regions. |
| HTML | Self-contained viewer with embedded OCR-aligned page rasters and coordinate-positioned raw/refined text. |
| Manifest | Versioned JSON audit metadata describing selected pages, engines, routing, attempts, usage, quality diagnostics, workflow state, refinements, and generated artifacts. |
| Normalized value | A typed conversion accepted by schema-aware normalization, such as a numeric string converted to a number. |
| Quality diagnostics | Per-page measurements and warnings for properties such as DPI, skew, blur, contrast, shadows, compression artifacts, and cropped edges. |
| Raw evidence | RapidOCR output and source geometry retained without GPT or user corrections overwriting it. |
| Reasoning effort | The OpenAI request setting fixed to `medium` for every GPT request. |
| Refinement layer | Auditable GPT or user corrections stored separately from raw OCR evidence. |
| Review required | Terminal state indicating that processing completed but evidence, confidence, or validation needs a person. |
| Review item | Structured audit entry naming the workflow stage, issue code, explanation, affected pages/source IDs, retryability, and attempt number. |
| Selected pages | Inclusive, one-based source pages chosen for processing and export. |
| Source-faithful view | The HTML viewer keeps the page raster as visual truth and overlays selectable text and evidence using canonical Parse coordinates. |
| Workflow event | Audit entry recording state, action, provider, reason, elapsed time, warnings, and token/cost impact. |

See [Architecture](ARCHITECTURE.md) for the runtime flow and
[Domain model](domain-model.md) for the full evidence and workflow contracts.
