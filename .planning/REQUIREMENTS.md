# Requirements: Agentic Document Extractor

**Defined:** 2026-08-30
**Core Value:** Produce evidence-grounded extraction without inventing or discarding source data.

## Phase 1 Requirements

### Context engineering

- [x] **CTX-01**: Compact and full context preserve every required OCR evidence field.
- [x] **CTX-02**: Every selected page image remains attached at high detail in both modes.
- [x] **CTX-03**: Batching uses rendered evidence size and never truncates an oversized page.
- [x] **CTX-04**: Stable instructions precede dynamic configuration and document evidence.
- [x] **CTX-05**: Usage records, Streamlit, and manifests expose exact context diagnostics.
- [x] **CTX-06**: Repository-local guidance preserves product invariants for future agents.
- [x] **CTX-07**: Deep code review, UAT, lint, type, test, artifact, and launch checks pass.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Remove image coverage in Balanced mode | Conflicts with checkbox-accuracy requirement |
| Estimate tokens from character counts | Provider tokenization and image tokens are not equivalent |
| Live OpenAI prompt benchmark | Paid external call needs separate authorization |

## Traceability

| Requirement | Phase | Status |
| --- | --- | --- |
| CTX-01 through CTX-07 | Phase 1 | Complete |

**Coverage:** 7 requirements mapped to Phase 1; 0 unmapped.

---
*Requirements defined: 2026-08-30*
