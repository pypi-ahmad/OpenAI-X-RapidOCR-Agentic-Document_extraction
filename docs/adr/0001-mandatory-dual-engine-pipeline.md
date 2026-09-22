<!-- generated-by: gsd-doc-writer -->
# ADR 0001: Require OCR, layout, and refinement engines

- Status: Accepted
- Date: 2026-08-30

## Context

RapidOCR supplies local text, confidence, and geometry. PP-DocLayoutV3 supplies document-region
classes, confidence, and reading order. GPT supplies semantic
and layout validation, correction, and structured interpretation. Either engine
alone would violate the product contract implemented by the workflow.

## Decision

Every extraction validates access to OpenAI `gpt-6-sol` before processing
document content. After that preflight succeeds, RapidOCR produces the local first pass, then
PP-DocLayoutV3 runs on every selected page before any GPT refinement request. Every GPT request uses
`gpt-6-sol` with medium reasoning effort. Missing or invalid OpenAI
configuration, RapidOCR or PP-DocLayoutV3 initialization or processing failure, and required
GPT refinement failure all prevent successful completion. The application has
no RapidOCR-only successful mode.

Balanced and High Accuracy remain routing modes. They alter the amount of OCR
context sent to GPT, not whether GPT participates. Both attach every selected
page image for checkbox coverage. Balanced sends compact OCR evidence by
default and full evidence only for routed risk or complexity. High Accuracy
sends full relevant OCR evidence for every selected page.

## Consequences

- Raw geometry is available before semantic refinement.
- The application cannot complete extraction offline.
- Both modes send every selected page at high detail, route locally identified uncertainty
  to additional high-detail crops, and make at least one GPT call.
- Tests prove OpenAI preflight precedes RapidOCR, RapidOCR precedes refinement,
  both modes invoke GPT, and requests use the fixed model settings.
- UI, API, manifests, and documentation describe RapidOCR and gpt-6-sol as the
  mandatory dual engines, with PP-DocLayoutV3 as a required local layout stage.

See [Architecture](../ARCHITECTURE.md) for the runtime flow and
[Configuration](../CONFIGURATION.md) for actionable engine setup requirements.
