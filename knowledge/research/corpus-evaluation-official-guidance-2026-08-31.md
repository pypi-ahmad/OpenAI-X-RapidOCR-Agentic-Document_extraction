# Corpus evaluation: official guidance and a repository-specific metric plan

**Date:** 2026-08-31  
**Scope:** all-page PDF outputs compared with paired Markdown and structured JSON ground truth

## Executive recommendation

Use a layered, deterministic evaluator. Gate first on file pairing, JSON validity/schema, page coverage, grounding-range validity, and required semantic objects. Then report transcription distance, Markdown structure, field/value accuracy, object detection, and geometry separately. Do not collapse these into one headline score until each submetric and its weighting are visible.

This follows OpenAI's guidance to define the objective, dataset, and task-specific metrics before running comparisons; to combine metrics with human judgment; and to calibrate automated scoring with human feedback.[^openai-eval] OpenAI also distinguishes exact string checks, text-similarity graders, model graders, and executable Python graders, and supports combining field-specific graders when different fields need different tolerances.[^openai-graders]

## What the repository already measures

`src/agentic_extractor/oracle_eval.py` currently checks top-level shape, selected pages, Markdown character/checkbox/table counts, node-type counts and exact type order, grounding-range validity, checkbox-state counts, and one-to-one special-object matches by page/type with a fixed IoU threshold. `tools/run_corpus_validation.py` enforces a paired `.parse.json` and `.parse.md` for every source PDF and sends every PDF page to the real validator.

Those checks should remain. Counts are good diagnostics, but equal length or equal counts do not prove equal content. Likewise, exact node-type order is a useful strict signal but cannot explain whether the error is text, hierarchy, geometry, or harmless serialization.

## Recommended metric stack

### 1. Corpus integrity and hard gates

Fail the run, rather than assigning a low quality score, when any PDF lacks either oracle file, any page is unprocessed, JSON cannot be parsed, the candidate violates its required schema, a grounding range falls outside canonical Markdown, or a required output artifact is absent. Validate structured output with a versioned JSON Schema: the JSON Schema validation specification defines assertion keywords for structural validation and distinguishes them from non-assertive annotations.[^json-schema]

Record the exact source/oracle/candidate hashes, application revision, prompt version, model identifier, reasoning effort, run mode, page list, and evaluator version. This makes cycle-to-cycle results attributable rather than silently comparing different inputs or graders.

### 2. Markdown transcription fidelity

Compute two views per page and for the corpus:

- **Strict text:** normalize line endings only, then compute character edit counts and `CER = (substitutions + deletions + insertions) / oracle characters`.
- **Canonical text:** apply Unicode NFC, normalize line endings, and optionally collapse only whitespace that the acceptance contract declares insignificant; compute CER and token/word edit rate again.

Unicode normalization is necessary because canonically equivalent strings can otherwise have different binary representations; Unicode Standard Annex #15 defines the normalization forms and states that normalized equivalent strings have a unique binary representation.[^unicode]

Do **not** silently lowercase, strip punctuation, use compatibility normalization, or canonicalize dates/numbers in the strict view: those transformations can hide OCR errors in identifiers, decimal punctuation, units, checkbox states, and signatures. Report normalization rules and both raw and canonical results.

Report micro averages from summed edit counts and macro averages across documents/pages. Micro scores expose total corpus error; macro scores stop long documents from hiding failures in short ones. Also report exact-match page rate and the worst pages by normalized edit distance.

### 3. Markdown structure fidelity

Parse oracle and candidate Markdown with the same pinned CommonMark/GFM-capable parser and compare syntax trees after discarding source-position metadata. CommonMark supplies a versioned syntax specification and conformance examples, making parser-based comparison more defensible than regex-only Markdown comparison.[^commonmark]

Report heading sequence/levels, paragraph and list counts, list-item order, HTML table count, row/cell counts, links, emphasis, and checkbox label/state tuples. For tables, compare a normalized cell matrix plus `rowspan`/`colspan`; score table detection precision/recall/F1 separately from cell exact match. Keep rendered-structure scores separate from plain-text CER so a table flattened into prose cannot receive a deceptively strong transcription score.

### 4. Structured JSON fidelity

First compare the contract schema, then compare meaning:

- recursively compare field presence, JSON types, and values;
- treat object key order as irrelevant and array order as significant unless the field contract explicitly says otherwise;
- use exact match for IDs, codes, checkbox states, dates, amounts, units, page numbers, spans, and enumerations;
- use declared field-specific canonicalizers only where the product contract permits equivalent representations;
- score nullable/abstained fields explicitly rather than converting missing, `null`, empty string, and unsupported guesses into one value.

Report field-level true/false positives and false negatives, precision/recall/F1, exact-record match, and path-level error samples. OpenAI's grader examples explicitly use exact matching for fields where exactness matters and fuzzy similarity for a name where misspellings are tolerated; that supports field-specific rules rather than one fuzzy JSON score.[^openai-graders]

### 5. Semantic objects and grounding

For tables, figures, logos, attestations, marginalia, and controls, match one oracle object to at most one candidate object on the same page and type. Report detection precision/recall/F1 and unmatched objects. For matched objects, report IoU distribution and thresholded recall at declared thresholds; do not rely only on a single permissive cutoff. The existing evaluator's greedy one-to-one matching is a reasonable diagnostic baseline, but matching policy and thresholds must be versioned and visible.

For grounding, additionally check page identity, normalized box validity, Markdown-range validity, range-text agreement, and evidence-link completeness. Geometry correctness and text correctness should remain separate because a correctly transcribed value can be grounded to the wrong region.

### 6. Model-based review only for residual ambiguity

Use deterministic graders for schema, exact fields, edit distance, tables, controls, and geometry. Reserve a model grader for narrowly defined semantic questions that deterministic comparison cannot decide, and require a rubric plus representative human-graded examples. OpenAI warns that model graders need iterative calibration against trusted expert grades and that grader hacking can be detected by disagreement between model-grader and expert-human evaluations.[^openai-graders]

Do not let the same model that produced the candidate be the sole acceptance judge. Store the model judgment, rubric version, and deterministic evidence independently; a model score must not override an exact mismatch in a critical identifier.

## OpenAI vision constraints relevant to the real run

OpenAI documents that image inputs consume tokens and that vision models can err on small text, precise spatial localization, and counting; it recommends enlarging small text and using `detail: "original"` for OCR or coordinate-sensitive work when supported.[^openai-vision] For GPT-5.6, OpenAI states that `auto` uses `original` sizing behavior, while `high` is standard high-fidelity understanding and `original` is preferred for large, dense, coordinate-sensitive OCR/localization tasks; larger inputs can increase tokens and latency.[^openai-deploy]

Therefore the evaluator should not treat a vision model's object count or coordinates as ground truth. Use the supplied paired artifacts as the oracle, compute deterministic counts/geometry locally, and log image dimensions, detail level, image tokens, latency, and cost for each run. If the product invariant requires `high`, evaluate that exact production configuration rather than changing detail during the benchmark.

## Threshold and iteration policy

1. Freeze the evaluator and oracle hashes before a repair cycle.
2. Run every PDF and every page; never average away a failed document.
3. Require zero hard-gate failures and zero regression on critical exact fields.
4. Compare each cycle with the same per-document table, including absolute errors and deltas.
5. Inspect the worst pages and every critical-field mismatch; add a focused regression fixture for each confirmed mechanism.
6. Keep a held-out subset if prompts or heuristics are repeatedly tuned to this corpus. OpenAI recommends evaluation on real-world distributions, early and continuous evaluation, and growing the eval set from observed failures.[^openai-eval]

Suggested acceptance dashboard: completion rate; strict/canonical CER; page exact-match rate; Markdown AST/type/order metrics; table detection and cell/span accuracy; checkbox label/state accuracy; JSON schema pass rate; critical-field exact match; semantic-object precision/recall/F1; grounding validity and IoU; per-document latency, tokens, and cost.

## Sources

[^openai-eval]: OpenAI, [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices), accessed 2026-08-31.
[^openai-graders]: OpenAI, [Graders](https://developers.openai.com/api/docs/guides/graders), accessed 2026-08-31.
[^openai-vision]: OpenAI, [Images and vision](https://developers.openai.com/api/docs/guides/images-vision), accessed 2026-08-31.
[^openai-deploy]: OpenAI, [API deployment checklist: Set image detail intentionally](https://developers.openai.com/api/docs/guides/deployment-checklist#set-image-detail-intentionally), accessed 2026-08-31.
[^unicode]: Unicode Consortium, [Unicode Standard Annex #15: Unicode Normalization Forms](https://www.unicode.org/reports/tr15/), Version 17.0.0, 2025-07-30.
[^json-schema]: JSON Schema, [JSON Schema Validation: A Vocabulary for Structural Validation of JSON, Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-validation), 2022-06-10.
[^commonmark]: John MacFarlane et al., [CommonMark Specification 0.31.2](https://spec.commonmark.org/0.31.2/), 2024-01-28.
