# LandingAI ADE vs. Our App: Output Comparison

**Date:** 2026-08-31  
**Scope:** The two supplied output folders for `Masked_Amerigroup_RealSolutions_1.pdf`  
**LandingAI snapshot:** `dpt-3-pro-20260710`, 11 pages  
**Our snapshot:** canonical Parse contract v7, Balanced mode, 11 pages  
**Evaluation oracle:** the supplied LandingAI Markdown and Parse JSON are treated as ground truth at the user's direction

## Executive summary

LandingAI's supplied output is materially better at **semantic document reconstruction**. It groups the page into a small number of meaningful elements, reconstructs nine tables with cell spans and reading order, and represents the form checkboxes conservatively. Our supplied snapshot is stronger at **local auditability and deliverable breadth**: it retains raw OCR geometry and confidence, GPT refinements, quality diagnostics, usage/cost records, an annotated PDF, coordinate-positioned HTML, checkbox crops, and a checksummed manifest.

The largest quality gap is not basic OCR. Normalized character-sequence agreement with the LandingAI ground truth is high on most pages (0.801 to 0.977 on 9 of 11 pages). The gap is **structure**:

- LandingAI emits 9 HTML tables, 51 rows, and 146 grounded cells.
- Our snapshot emits no HTML tables. Twelve local table candidates were marked invalid; two were recorded as GPT-corrected in review metadata, but no table structure reached the exported Markdown.
- Page 4 is the clearest failure: LandingAI preserves two four-column infusion tables; ours flattens and scrambles the same values and inserts false checkbox/list syntax.
- LandingAI's Markdown contains 28 checkbox markers. Ours contains 67 markers/candidates, many on pages with no checkbox form. This adds unsupported semantics even though our checkbox records correctly mark agreement as incomplete or review-required.

LandingAI is also much faster in the supplied telemetry: **13.339 seconds** for the whole document. Our manifest records **77.43 cumulative layout seconds**, **17.24 OCR seconds**, **22.54 table-structure seconds**, and **354.19 cumulative GPT seconds across 13 calls**. Our artifact does not record one authoritative end-to-end wall time, so an exact latency ratio would be misleading.

For this report, the supplied LandingAI output is the reference truth: every deviation in our output is scored as an error unless explicitly identified as a harmless representation difference. This supports reproducible **LandingAI-agreement** measurements for this document, not an independently verified absolute accuracy claim. LandingAI officially reports 99.16% on DocVQA, but that separate benchmark must not be substituted for our agreement score on this PDF.[^landing-overview]

## Methodology

### Local evidence inspected

LandingAI:

- `LandingAI Output/Masked_Amerigroup_RealSolutions_1.parse.md`
- `LandingAI Output/Masked_Amerigroup_RealSolutions_1.parse.json`

Our app:

- `Our App Output/document.md`
- `Our App Output/parse-result.json`
- `Our App Output/manifest.json`
- `Our App Output/document.html`
- `Our App Output/annotated.pdf`
- 67 JPEG checkbox crops under `Our App Output/checkboxes/`

The comparison counted structural elements directly from JSON and markup, inspected page-level output, and calculated normalized sequence agreement after removing HTML tags and collapsing text to lowercase alphanumeric tokens. LandingAI is treated as correct when assigning direction to each difference. The sequence score remains a similarity measure rather than CER or WER.

Official behavior claims were checked only against LandingAI's own documentation and product pages. Observations from the local files are labeled separately from those claims.

### Important snapshot caveat

`Our App Output/document.md` predates the current code change that serializes accepted tables as HTML. This report evaluates the supplied artifact, not an inferred future rerun. Current serialization code cannot recover tables that the upstream table detector rejects, so rerunning may change the markup format without closing the core detection and grouping gap.

## Quantitative comparison

| Dimension | LandingAI supplied output | Our supplied output | Interpretation |
| --- | ---: | ---: | --- |
| Pages | 11, all `ok` | 11, all `completed` | Both completed every page |
| Markdown characters | 22,856 in JSON; 22,871 file bytes | 21,204 characters; 21,212 bytes | Similar output volume; not equivalent completeness |
| Markdown words | 3,876 | 3,467 | Ours contains about 10.6% fewer word tokens |
| Semantic/grounded structure | 329 nodes including document/pages | 601 OCR blocks, 424 chunks, 1,025 grounding entries | Ours is much more granular, but granularity includes fragmentation |
| Text-like types | 84 text, 74 marginalia | 296 paragraphs, 272 key-value OCR blocks | LandingAI groups lines; ours often keeps line-level evidence |
| Specialized types | 9 tables, 2 logos, 2 attestations | 5 heading blocks; layout regions stored separately | LandingAI's exported hierarchy carries richer semantic types |
| Tables in Markdown | 9 | 0 | Largest functional quality gap |
| Table rows / cells | 51 / 146 | 0 / 0 in Markdown | Ours loses row/column relationships |
| Table candidates/reviews | Not exposed as failures | 12 candidates: 10 unresolved, 2 review records accepted | Our abstention avoids fabricated grids but produces no usable tables |
| Checkbox markers in Markdown | 28: 5 checked, 23 unchecked | 67: 34 checked, 32 unchecked, 1 uncertain | Our checkbox detector is over-sensitive |
| OCR blocks with polygons/bboxes | Atomic grounding exists under semantic nodes | 601/601 have polygons and normalized bboxes | Both ground content; ours exposes raw OCR geometry explicitly |
| Raw OCR confidence | Not included in supplied result | 601 scores; mean 0.9758, minimum 0.5162 | Our artifact is better for OCR-level audit and thresholding |
| Warnings | No failed pages; no warning list in supplied result | 28 warnings | Ours exposes uncertainty, but downstream output still contains noise |
| Processing telemetry | 13,339 ms, 22.5 credits | local stage timings plus 13 GPT calls; 420,092 input and 44,231 output tokens | Ours is far heavier |
| Reported cost | 22.5 credits | $0.14534061 exact GPT API cost; local API cost $0 | Different billing units and scopes |
| Files supplied | Markdown + parse JSON | Markdown, parse JSON, manifest, HTML, annotated PDF, checkbox crops | Our bundle is substantially more operationally auditable |

LandingAI's current public Explore/Team list price is $0.01 per credit, so 22.5 credits would equal **$0.225** under those plans.[^landing-pricing] That is about $0.080 more than our reported GPT cost. This is not a full economic comparison: our $0.145 excludes local GPU/CPU electricity, hardware, and operator time, while LandingAI's credit price bundles hosted inference and infrastructure. Enterprise pricing may differ.

## Page-level agreement with LandingAI ground truth

Normalized sequence agreement between the LandingAI reference and ours:

| Page | Agreement | Main observed issue |
| ---: | ---: | --- |
| 1 | 0.885 | Mostly formatting and reading-order differences |
| 2 | 0.871 | Form tables and checkbox semantics differ |
| 3 | 0.896 | Table grouping differs |
| 4 | 0.801 | Two infusion tables flattened; false checkbox/list syntax |
| 5 | 0.884 | Structured pump/catheter tables lost |
| 6 | 0.940 | Strong text agreement; both contain noisy header/logo OCR |
| 7 | 0.977 | Closest match; mostly prose |
| 8 | 0.928 | Mostly prose; layout differences |
| 9 | 0.675 | Largest text/order divergence |
| 10 | 0.944 | Strong text agreement; our false checkbox markers remain |
| 11 | 0.910 | Signature/attestation representation differs |

These scores demonstrate that RapidOCR captures much of the LandingAI reference text. They may be summarized only as LandingAI-reference agreement, not as absolute document accuracy.

## Detailed findings

### 1. Text recognition: broadly competitive, but noisier in difficult regions

Both outputs recover the main fax content, patient/report narrative, identifiers, dates, dosages, and most repeated headers and footers. Our 601 OCR blocks all carry a polygon, normalized bounding box, source ID, raw score, and reading-order index. This is excellent low-level evidence retention.

LandingAI more often converts those signals into clean semantic groups. Its hierarchy has only 84 `text` nodes and 74 `marginalia` nodes, with atomic grounding beneath them. Our output has 601 line-like blocks and 424 chunks. The difference is visible in phrases such as `*BeginningofSessioni`, `wCurrentSettingst`, and fragmented table values on page 4. More blocks do not mean more information; here they increase ordering and classification opportunities for error.

LandingAI's official docs describe Parse as producing structured Markdown plus element-level page and coordinate references.[^landing-parse] The supplied LandingAI JSON substantively matches that claim: 328 of 329 nodes have a range and normalized box, with the ungrounded node being the document root.

### 2. Reading order and semantic grouping: LandingAI is cleaner

LandingAI separates repeated fax headers/footers as `marginalia`, logos as `logo`, signatures as `attestation`, tables as `table`, and content as `text`. Our canonical contract stores richer raw and layout evidence, but the final Markdown often follows fine OCR line order rather than document-level relationships. Six pages explicitly warn that reading order is ambiguous.

The result is most damaging inside multi-column or grid content. Values are usually present but can be separated from their labels, reordered, or converted into bullets. For downstream RAG, this changes meaning: retrieving `0.75 mcg` without its step, time range, drug column, and session state is much less useful than retrieving a table row.

LandingAI officially defines chunks as discrete semantic elements and documents specific types including text, tables, figures, marginalia, logos, and attestations.[^landing-chunks] The supplied hierarchy uses that ontology consistently. Our output has paragraph/key-value/list/heading blocks and form/list chunks, but its semantic categories did not compensate for failed table structure.

### 3. Tables: the decisive quality gap

LandingAI reconstructs nine HTML tables with 51 rows and 146 cells. The output preserves `rowspan` and `colspan`, and its JSON grounds tables and individual cells with row/column/span indices. This directly follows LandingAI's documented format: table chunks use HTML, and individual cells may have IDs, coordinates, row/column positions, and spans.[^landing-markdown][^landing-json]

Our snapshot detects 12 possible tables across pages 1 to 5, but all 12 page-level structures are stored as invalid with zero cells. Review metadata reports two page-3 tables as accepted/corrected and ten as unresolved/abstained, yet the accepted corrections are not materialized into `document.md`; the exported Markdown contains zero `<table>` elements.

Page 4 shows the user impact:

- LandingAI creates separate “Beginning of Session” and “Current Settings” tables.
- Each has four columns: Step, Duration, FENTANYL, BACLOFEN.
- Multi-line doses stay in the correct cells.
- Our output interleaves headers, values, times, bullets, stray characters, and false checked boxes.

Abstaining from invalid local geometry is safer than inventing a grid, but the current system treats abstention as the endpoint. The repair path needs to generate a validated table from image evidence and OCR block assignments, then persist the accepted structure into the canonical output.

### 4. Checkboxes: our high-recall strategy produces false semantics

LandingAI's page-2 form contains 28 checkbox markers in Markdown: 5 `[x]` and 23 `[ ]`. Our Markdown contains 67: 34 `[x]`, 32 `[ ]`, and one uncertain marker. Our canonical JSON likewise contains 67 checkbox records, including repeated page-wide candidates on pages 3 to 11 where the content is predominantly reports and prose rather than checkbox forms.

Our records often acknowledge the weakness: the sample checkbox has `decision_status: review_required`, `agreement: incomplete`, and no local-vision state. However, these candidates are still rendered as semantic checkbox Markdown. This violates the intended policy boundary: uncertain detection is being displayed as document content rather than only as review metadata.

On page 4, ordinary headers and numeric values become checked boxes, such as BACLOFEN, FENTANYL, and dosage entries. This is not a minor formatting issue; `[x]` asserts a source state absent from the supplied comparison output.

The remedy is precision-first publication: only emit `[x]`/`[ ]` when control geometry, a unique grounded label, local visual classification, and Luna agree. Keep all other candidates in diagnostics as `REVIEW_REQUIRED`, never in canonical Markdown.

### 5. Grounding and auditability: both are strong in different ways

LandingAI:

- Grounds semantic regions and 146 table cells.
- Links every structure node to a Markdown character range.
- Provides both aggregate and atomic boxes for text-like nodes.
- Produces a much smaller 375.8 KB JSON result.

Our app:

- Grounds every raw OCR block with pixel polygon and normalized bounding box.
- Grounds derived chunks back to source block IDs.
- Retains raw evidence, 27 GPT refinements, engine/model provenance, timings, diagnostics, and review reasons.
- Produces 1,025 grounding entries, a 4.86 MB parse result, and a 2.42 MB manifest.
- Adds a source-faithful coordinate HTML view and annotated PDF.

Our evidence chain is more inspectable for debugging. LandingAI's chain is more useful at the semantic unit that downstream users care about. The best target is not fewer raw records; it is a two-layer contract: immutable fine evidence plus compact, accepted semantic structures whose grounding resolves back to that evidence.

### 6. Artifact breadth: our app wins the supplied-folder comparison

LandingAI's supplied folder contains only Markdown and parse JSON. Our folder contains:

- canonical Markdown;
- canonical Parse JSON;
- checksummed JSON manifest;
- coordinate-positioned standalone HTML backed by page imagery;
- annotated PDF;
- 67 checkbox review crops.

The manifest identifies source hash, selected pages, engine versions/devices, warnings, failures, quality diagnostics, visual routing, prompts, usage, costs, and artifact SHA-256 values. This is materially better for reproducibility and local review.

This does **not** prove LandingAI cannot produce other views or artifacts; it only describes the two folders supplied for comparison.

### 7. Latency: our pipeline does much more work and sends too much context

LandingAI reports 13,339 ms for the 11-page Parse. Our manifest records:

- RapidOCR: 17.24 seconds;
- PP-DocLayoutV3 on CPU: 77.43 cumulative seconds;
- table structure: 22.54 cumulative seconds;
- checkbox detection: 0.60 seconds;
- GPT: 354.19 cumulative seconds across 13 calls.

The GPT calls comprise 11 page refinement calls, one checkbox-verification call, and one full-document repair call. They use 420,092 input tokens, including 55,468 cached tokens, plus 44,231 output tokens. Every page was routed with a high-resolution crop covering roughly 94% to 99% of the page, so “uncertain-region” routing behaved almost like another full-page pass.

Three implications follow:

1. The CPU layout fallback alone exceeds LandingAI's complete reported duration by 5.8×.
2. One GPT call per page plus a 200,047-input-token repair call dominates latency and cost.
3. Large, noisy evidence payloads can reduce accuracy as well as speed by asking Luna to reconcile too many low-value regions.

The recorded stage values are cumulative timings and may overlap. Because our manifest lacks a single end-to-end wall-clock duration, they must not be summed and presented as exact user wait time.

### 8. Cost: locally cheaper by API line item, but less efficient per useful structure

Our exact GPT API cost is $0.14534061. LandingAI reports 22.5 credits, which corresponds to $0.225 at the documented Explore/Team rate of $0.01 per credit.[^landing-pricing] On that narrow basis, ours costs about 35% less.

But cost per page is not the right success metric when the output loses all nine tables and inserts false checkbox semantics. Our pipeline spends 420K input tokens and 13 calls yet still produces a structurally weaker document. The priority should be **cost per accepted grounded semantic element**, especially correct tables and controls.

## Official LandingAI claims vs. this observed sample

| Official first-party statement | What this sample shows | What it does not prove |
| --- | --- | --- |
| Parse returns structured Markdown and hierarchical JSON with exact page/coordinate references.[^landing-overview] | The supplied JSON has a hierarchy, Markdown ranges, pages, and boxes. | Coordinate presence alone does not prove text or grouping correctness. |
| ADE reports 99.16% on DocVQA.[^landing-overview] | Not testable from these outputs. | It does not imply 99.16% Markdown, table, checkbox, or field accuracy on this document. |
| Table chunks use HTML and support cell grounding and spans.[^landing-markdown][^landing-json] | The sample has 9 HTML tables and 146 cell nodes with row/column/span metadata. | The sample has no independent cell-level gold labels. |
| Parse identifies text, tables, images/form fields and other semantic chunks.[^landing-chunks] | The sample includes text, marginalia, tables, logos, and attestations. | It does not show every supported type or every document class. |
| DPT improvements target complex/merged-cell tables, layout detection, richer ontology, and cell/column alignment.[^landing-models] | The sample's form and infusion tables are noticeably coherent. | A single favorable sample cannot establish general superiority across a corpus. |

## Prioritized improvements

### P0: Correct semantic publication boundaries

1. **Never publish uncertain checkboxes as `[x]` or `[ ]`.** Require agreement between control geometry, local vision state, unique label grounding, and Luna. Otherwise retain only a review candidate.
2. **Persist accepted table repairs into the canonical page/chunk/Markdown layers.** A review metadata status of `accepted` is insufficient if the accepted grid is absent from the document.
3. **Validate tables at the semantic level.** Enforce exact block assignment, non-overlapping grid spans, source-text conservation, row/column consistency, and arithmetic checks before publication.

Acceptance measure: reproduce all nine LandingAI reference tables, 51 rows, 146 cells, their spans, and their grounded content; match the LandingAI checkbox states and labels without publishing additional controls.

### P1: Reduce fragmentation before Luna

4. Build deterministic line-to-region grouping using layout regions, whitespace, alignment, font/height proxies, and repeated-margin detection.
5. Send semantic regions rather than hundreds of raw blocks. Preserve the raw block IDs behind each region.
6. Add explicit types for marginalia, logo, attestation, table, and figure in the accepted output layer.

Acceptance measure: reduce semantic chunks on this sample without losing source text or grounding; eliminate fragmented strings and label/value separation in tables/forms.

### P1: Make visual routing genuinely selective

7. Stop treating almost-full-page merged regions as “uncertain regions.” Cap crop expansion and high-resolution area ratio.
8. Use one low-resolution page overview for layout plus high-resolution crops only for unresolved tables, checkboxes, low-confidence text, signatures, and conflicting reading order.
9. Do not perform a full-document repair call with the entire evidence payload. Repair only failed semantic objects with stable source IDs.

Acceptance measure: reduce GPT input well below 420K tokens and avoid any repair call above the largest individual page context, without decreasing gold-set F1.

### P1: Fix the local layout bottleneck

10. Restore PP-DocLayoutV3 GPU execution or replace it with a measured faster layout path on this hardware.
11. Keep layout/table models warm and cache by page image hash and model configuration.
12. Skip table structure and checkbox classifiers when a cheap, high-precision gate finds no credible candidate.

Acceptance measure: record authoritative end-to-end wall time and cold/warm stage timings; target warm local preprocessing below LandingAI's 13.3-second sample time before Luna, or clearly document the local-hardware tradeoff.

### P2: Establish a real evaluation harness

13. Version the supplied LandingAI Markdown and Parse JSON as the regression oracle for text, reading order, table cells/spans, checkbox state/label, marginalia, and attestations.
14. Report CER/WER, reading-order accuracy, table TEDS or cell F1, checkbox precision/recall/state accuracy, grounding IoU, abstention precision, latency, and cost.
15. Run multiple document families. Do not optimize solely against this medical fax.

Acceptance measure: versioned LandingAI-reference evaluation data plus reproducible before/after metrics. Label any overall score explicitly as LandingAI agreement and define its metric and sample scope.

## Limitations

- LandingAI is the designated ground truth for this comparison by user decision. It was not independently human-verified, so resulting scores measure conformity to that oracle.
- The original source PDF is not in either output folder, so visual claims are limited to the embedded page imagery in our HTML/annotated artifacts and cross-output consistency.
- The systems may have used different rendering resolution, preprocessing, model dates, prompts, or privacy/redaction settings.
- LandingAI's JSON uses a newer `structure` shape in this snapshot, while some current public documentation still describes `chunks`, `splits`, and a grounding map. The report compares the actual artifact and uses docs for behavioral intent, not schema identity.
- Our current code has changed after this artifact was generated. Only a fresh paired run can measure the current build.
- LandingAI's `duration_ms` is authoritative for its request. Our stage timing fields are cumulative and do not provide one comparable end-to-end wall time.
- Local hardware cost is absent from our $0.145 GPT figure; LandingAI's plan/tier for this run is not stated beyond `service_tier: priority`.

## Conclusion

Our app is not yet output-equivalent to LandingAI ADE on this sample. It has a stronger local evidence ledger and a better artifact package, but LandingAI produces the more useful canonical document because it converts visual layout into compact, grounded semantic structure (particularly tables) without flooding the Markdown with uncertain checkbox states.

The shortest path to improvement is not adding more detectors or more GPT context. It is tightening publication rules, persisting validated table repairs, grouping OCR evidence before refinement, and measuring against the versioned LandingAI reference tables and controls. Those changes should improve agreement, latency, and cost together.

## Sources

[^landing-overview]: LandingAI, [ADE Overview](https://docs.landing.ai/ade/ade-overview), accessed 2026-08-31. Describes Parse-first workflows, structured Markdown/hierarchical JSON, exact page/coordinate references, chunk detection, visual grounding, and the 99.16% DocVQA claim.
[^landing-parse]: LandingAI, [Parse Documents](https://docs.landing.ai/ade/parse), accessed 2026-08-31. Defines Parse output as structured Markdown, chunks, metadata, and page/coordinate references.
[^landing-json]: LandingAI, [JSON Response for Parsing](https://docs.landing.ai/ade/ade-json-response), accessed 2026-08-31. Documents chunks, tables/cells, IDs, splits, Markdown ranges, page numbers, normalized bounding boxes, and row/column/span metadata.
[^landing-markdown]: LandingAI, [Markdown Response](https://docs.landing.ai/ade/ade-markdown-response), accessed 2026-08-31. Documents HTML table markup, table/cell IDs, and `rowspan`/`colspan` behavior.
[^landing-chunks]: LandingAI, [Chunk Types](https://docs.landing.ai/ade/ade-chunk-types), accessed 2026-08-31. Defines text, table, marginalia, figure, logo, attestation, and other chunk classes.
[^landing-models]: LandingAI, [Parse Model Versions](https://docs.landing.ai/ade/ade-parse-models), accessed 2026-08-31. Describes table fidelity, layout detection, ontology, cell parsing, alignment, and table-boundary improvements.
[^landing-pricing]: LandingAI, [Plans & Billing](https://docs.landing.ai/ade/ade-pricing), accessed 2026-08-31. States that Explore and Team credits cost $0.01 and that total cost is credits consumed multiplied by credit price.
