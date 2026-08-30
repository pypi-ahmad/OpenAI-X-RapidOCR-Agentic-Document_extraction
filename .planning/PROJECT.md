# Agentic Document Extractor

## What This Is

A local-machine, LandingAI-ADE-inspired document extraction application for PDFs and common image
formats. RapidOCR creates grounded local evidence and GPT-5.6-luna validates and refines it into
auditable Markdown, workflows, structured fields, and downloadable artifacts.

## Core Value

Produce evidence-grounded document extraction without silently inventing or discarding source data.

## Requirements

### Validated

- ✓ Mandatory RapidOCR followed by GPT-5.6-luna with medium reasoning — existing application
- ✓ Local Streamlit UI on port 8841 and optional local FastAPI layer — existing application
- ✓ Evidence-bearing Parse, agentic workflows, artifacts, usage, and cost reporting — existing application

### Active

- [ ] Deep-review the lossless context-engineering implementation.
- [ ] Resolve auto-fixable review and UAT findings with verified atomic commits.

### Out of Scope

- Public hosting, accounts, authentication, and multi-tenancy — the product is local-machine only.
- RapidOCR-only or GPT-only successful extraction — both engines are mandatory.
- Paid live OpenAI evaluation — requires separate explicit authorization.

## Context

Phase 1 is a retrospective audit of commit `f030fa8`, which added deterministic compact context
rows, rendered-evidence batching, stable prompt prefixes, context telemetry, and repository-local
agent guidance. The implementation passed 132 tests with 91.74% coverage before audit.

## Constraints

- **Accuracy**: Preserve every selected page image and evidence-bearing OCR field.
- **Model**: Every OpenAI request uses `gpt-5.6-luna` with medium reasoning effort.
- **Auditability**: Raw OCR evidence is immutable; corrections remain a separate refinement layer.
- **Environment**: Windows 11, Python 3.13, `uv`, Streamlit port 8841.

## Key Decisions

| Decision | Rationale | Outcome |
| --- | --- | --- |
| Keep every selected image at high detail | Checkbox accuracy takes priority over image-token savings | ✓ Good |
| Compact repeated evidence keys into declared rows | Reduce context without dropping evidence | — Pending audit |
| Use explicit Phase 1 review scope | The project predates GSD planning artifacts | ✓ Good |

---
*Last updated: 2026-08-30 for retrospective Phase 1 audit initialization*
