<!-- generated-by: gsd-doc-writer -->
# ADR 0002: Preserve raw evidence and audit refinements

- Status: Accepted
- Date: 2026-08-30

## Context

OCR and language-model outputs can be uncertain. Replacing the original OCR
text or geometry would make corrections difficult to explain, validate, or
reverse.

## Decision

RapidOCR output is immutable source evidence. GPT block refinements must cite
valid block or visual-region evidence before they are accepted. Accepted block
refinements are recorded as `RefinementRecord` entries and applied only to
copied blocks used to rebuild derived Markdown. Other workflow proposals use
their stage-specific grounding and validation contracts. User acceptances and
corrections are recorded separately as `FieldCorrection` entries on a copied
workflow result. Unsupported, ungrounded, or schema-invalid fields cannot be
verified.

Annotated PDF, coordinate-positioned HTML, Parse JSON, and workflow validation retain the
same canonical grounding data.

## Consequences

- Every accepted value remains traceable to a selected source page.
- Reviewers can compare raw OCR, GPT refinement, and user correction.
- Result models contain some intentional duplication for auditability.
- Consumers must choose the appropriate layer rather than assuming the newest
  text replaced the original evidence.

See the [domain model](../domain-model.md) for the record contracts and the
[architecture guide](../ARCHITECTURE.md) for the evidence-validation flow.
