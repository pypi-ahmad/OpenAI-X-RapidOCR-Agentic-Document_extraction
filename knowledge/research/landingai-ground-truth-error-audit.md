# LandingAI-Ground-Truth Error Audit

**Audit date:** 2026-08-31  
**Document:** `Masked_Amerigroup_RealSolutions_1.pdf`, 11 pages  
**Oracle:** the supplied files under `LandingAI Output/`  
**System under test:** the saved files under `Our App Output/`

## Executive verdict

Under the user's explicit decision to treat the supplied LandingAI output as authoritative ground truth, our saved output is **not equivalent**. Its prose transcription is often close, but it fails badly on tables and checkboxes and contains smaller OCR, reading-order, formatting, and attestation errors.

The highest-impact discrepancies are:

1. LandingAI contains **9 HTML tables, 51 rows, and 146 cells**. Our saved Markdown contains **0 HTML tables**. It has only two pipe-table approximations on page 3; seven oracle tables are fully flattened.
2. LandingAI contains **28 checkbox markers, all on page 2**. Ours contains **67 markers across pages 2 to 11**. At least 34 markers on pages 3 to 11 are definite false additions. Only one of our 67 checkbox records is marked automated, yet all records, including 66 review-required records, are published into Markdown.
3. Page 4 is the worst structural failure. Two four-column infusion tables become interleaved text, bullet lists, and nine false checkbox markers.
4. Page 11 loses the oracle's explicit signature semantics (`[E-SIGNED]`, `[SIGNED]`, `[ILLEGIBLE_SIGNATURE]`) and adds an unsupported heading, `Bhal m`.
5. Our Parse JSON is internally inconsistent for repaired tables: page-3 review metadata says two tables were accepted, and Markdown contains two pipe grids, but `$.pages[2].table_structures[*]` still says `invalid` with empty cells.

The saved snapshot predates the current HTML table serializer, so pipe-table formatting is already changed in source. However, rerendering alone will not recover the seven missing tables, fix checkbox publication, or make accepted table structure persist in canonical JSON.

## Methodology and oracle rule

### Oracle rule

For this audit only, every semantic element, string, ordering decision, table cell/span, checkbox state, and formatting decision in:

- `LandingAI Output/Masked_Amerigroup_RealSolutions_1.parse.md`; and
- `LandingAI Output/Masked_Amerigroup_RealSolutions_1.parse.json`

is treated as correct. A difference in our output is therefore classified as our error, even where an independent human review might prefer our representation.

This is a product acceptance oracle, not a claim that LandingAI is universally infallible.

### Procedure

1. Split both Markdown files into the same 11 source pages.
2. Count tables, rows, cells, checkbox markers, words, and semantic/grounding nodes.
3. Compare every page manually at semantic-element level.
4. Verify representative differences with Markdown line locations and JSON paths.
5. Inspect current source to separate snapshot defects from already-modified behavior and active defects.
6. Use LandingAI first-party documentation only to explain the expected output contract.

The audit does not use raw edit distance as accuracy. Markup, spacing, tables, and repeated headers distort it. Counts are reported only when their meaning is defensible.

### Error classes

| Class | Meaning in this audit |
| --- | --- |
| Omission | Oracle content or semantic element is absent from ours |
| Addition | Ours introduces content or semantics absent from the oracle |
| Substitution | Ours replaces oracle text/value with a different value |
| Reading order | Correct pieces appear in the wrong sequence or association |
| Structure | Semantic grouping/hierarchy differs from the oracle |
| Table | Table boundary, row, column, cell, or span differs |
| Checkbox | Control existence, label, state, duplication, or placement differs |
| Grounding | A semantic item is absent from grounding, linked to the wrong source, or represented inconsistently |
| Formatting | Markdown/HTML semantics differ without necessarily changing text |

## Exact artifact and schema comparison

### Supplied artifacts

| Artifact | LandingAI | Our app |
| --- | --- | --- |
| Canonical Markdown | Yes, 22,871 bytes | Yes, 21,212 bytes |
| Parse JSON | Yes, 375,800 bytes | Yes, 4,862,359 bytes |
| Manifest | No supplied file | Yes, 2,416,095 bytes |
| Coordinate HTML | No supplied file | Yes, 6,665,088 bytes |
| Annotated PDF | No supplied file | Yes, 3,472,838 bytes |
| Checkbox crops | No supplied files | 67 JPEGs |

This table describes the supplied folders only. It does not claim that LandingAI's service cannot provide other visualization or download features.

### Parse schemas

LandingAI `parse.json` has three top-level fields:

- `$.markdown`
- `$.metadata`
- `$.structure`

Its structure is a semantic tree: document → 11 pages → semantic elements. It contains 329 total nodes:

| Oracle node type | Count |
| --- | ---: |
| document | 1 |
| page | 11 |
| text | 84 |
| marginalia | 74 |
| table | 9 |
| table_cell | 146 |
| logo | 2 |
| attestation | 2 |

All 328 non-root nodes have page, Markdown character range, and normalized box grounding. Text-like nodes also carry `atomic_grounding`. Tables contain explicit child cells with row, column, `rowspan`, and `colspan`.

Our `parse-result.json` exposes a broader evidence/audit schema: contract and document metadata, selected pages, Markdown, OCR/layout engines, timings, warnings, failures, OCR/GPT attempts, refinement layer, checkboxes, pages, chunks, coordinate spaces, and a grounding map. It contains:

- 601 OCR blocks, all with polygon and normalized bounding box;
- 424 derived chunks;
- 1,025 block/chunk grounding entries;
- 210 layout regions;
- 27 accepted/abstained/rejected refinement records;
- 12 local table candidates;
- 67 checkbox records; and
- 28 warnings.

Our schema is richer for diagnostics, but the oracle is richer where downstream meaning matters: tables, marginalia, logos, and attestations are part of the accepted semantic hierarchy rather than parallel diagnostic evidence.

### Canonical-format discrepancies

- LandingAI uses HTML tables in Markdown, as its official format documentation specifies.[^markdown]
- Our saved page-3 tables use pipe-table syntax (`Our App Output/document.md:343-356`); the other seven oracle tables are absent.
- LandingAI connects semantic nodes to exact Markdown ranges. Our block/chunk grounding has no equivalent accepted-document character range.
- Our semantic Markdown publishes review-required checkbox records. The JSON correctly labels their uncertainty, but the canonical text does not preserve that boundary.
- Our page-3 table Markdown and table review audit disagree with our serialized page structures.

LandingAI's current documentation describes Parse as structured Markdown, chunks/structure, and exact page/coordinate references; it also documents table- and cell-level IDs and grounding.[^parse][^json]

## Quantified discrepancy summary

### Tables

| Page | Oracle tables | Our recognizable grids | Oracle cells | Result |
| ---: | ---: | ---: | ---: | --- |
| 2 | 3 | 0 | 38 | Three complete omissions |
| 3 | 2 | 2 pipe grids | 17 | Both format-incompatible and contain text/row errors |
| 4 | 2 | 0 | 54 | Two complete omissions; severe order corruption |
| 5 | 2 | 0 | 37 | Two complete omissions; label/value association lost |
| **Total** | **9** | **2** | **146** | 7/9 absent; recognizable-grid recall 22.2%; HTML-table recall 0% |

“Recognizable-grid recall” is intentionally lenient: it counts page-3 pipe tables as detected despite oracle-format and content differences.

### Checkboxes

| Page | Oracle markers | Our markers | Net excess | Oracle rule result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 0 | 0 | Pass |
| 2 | 28 | 33 | +5 | State, duplication, label, and order errors |
| 3 | 0 | 2 | +2 | Both false additions |
| 4 | 0 | 9 | +9 | All false additions |
| 5 | 0 | 1 | +1 | False addition |
| 6 | 0 | 4 | +4 | All false additions |
| 7 | 0 | 3 | +3 | All false additions |
| 8 | 0 | 4 | +4 | All false additions |
| 9 | 0 | 4 | +4 | All false additions |
| 10 | 0 | 4 | +4 | All false additions |
| 11 | 0 | 3 | +3 | All false additions |
| **Total** | **28** | **67** | **+39** | At least 34 definite false positives outside page 2 |

On page 2, the marker totals alone understate errors. Ours marks Planned inpatient, Diagnostic study, and Home as checked, duplicates options, publishes raw label `p2-cv20`, and misses the oracle's exact table association for the servicing-provider `X`.

`$.checkboxes[21]` is the only record marked `automated`; it claims a checked **Home** control with three-engine consensus. The oracle says Home is unchecked (`LandingAI ...parse.md:129`). This is a false consensus, not merely an unreviewed candidate.

### Completion and telemetry

Both systems completed 11 pages without failed pages. LandingAI reports `$.metadata.duration_ms = 13339`, model `dpt-3-pro-20260710`, and 22.5 credits. Our manifest reports 13 GPT calls, 420,092 input tokens, 44,231 output tokens, and $0.14534061 GPT cost, plus cumulative local stage timings. Our snapshot has no single authoritative end-to-end wall time, so no exact latency ratio is asserted.

## Page-by-page error inventory

Line spans below refer to the two Markdown files.

### Page 1

**Oracle:** LandingAI lines 1 to 45. **Ours:** lines 1 to 58.

- **Reading order / structure:** Oracle keeps `San Antonio / TX 78209`; ours orders `San Antonio / 78209 / TX` (`LandingAI:28-29`; ours `19-23`).
- **Addition / formatting:** Ours emits isolated duplicate `Phone` and `Fax` labels and values (`ours:25-35`) instead of the oracle's compact sender block (`LandingAI:23-31`).
- **Formatting:** Oracle represents the note with `**Note:**`; ours leaves `Note:` unstyled and separates the asterisk-delimited alert into thematic breaks (`ours:39-51`).
- **Substitution/order:** Token alignment moves phone/fax values relative to labels; the content mostly survives, but associations are weaker.

Severity: **medium**, because core fax content is present but key-value relationships are degraded.

### Page 2

**Oracle:** lines 47 to 148. **Ours:** lines 59 to 286.

- **Table / structure omission:** Oracle has three provider/facility tables at lines 83 to 109, 38 cells total. Ours flattens all three into free text and checkbox bullets; corresponding JSON candidates `$.pages[1].table_structures[0:5]` are invalid and empty.
- **Checkbox substitution:** Oracle marks referring Nonparticipating and facility Nonparticipating; servicing-provider selection is represented by a separate table-cell `X` (`LandingAI:84,93,102`). Ours turns provider rows into repeated list controls (`ours:101-149`).
- **Checkbox false additions:** Ours asserts `[x] TIN`, `[x] Planned inpatient`, `[x] Diagnostic study`, and two `[x] Home` markers (`ours:139,183,203,227,248`). Oracle has TIN as ordinary text, Planned inpatient/Diagnostic study/Home unchecked (`LandingAI:95,114,120,129`).
- **Addition:** `- [ ] p2-cv20` exposes an internal candidate ID as document content (`ours:252`).
- **Duplication / reading order:** Options are repeated and reordered. For example Planned inpatient appears three times (`ours:179-221`) but once in the oracle (`LandingAI:114`).
- **Grounding:** The only automated checkbox, `$.checkboxes[21]`, links a checked control to `Home`; this contradicts the oracle state despite recorded OpenCV/RapidOCR/Luna consensus.

Severity: **critical**. This page contains authorization form decisions; false checked values alter meaning.

### Page 3

**Oracle:** lines 150 to 233. **Ours:** lines 287 to 365.

- **Substitution:** `Birthdate : 3/31/1983` becomes `Birthdate: 3|31983` (`LandingAI:179`; ours `310`).
- **Table / formatting:** Oracle has two HTML tables (`LandingAI:211-224`). Ours has two pipe tables (`ours:343-356`). Current source has already changed generic table serialization to HTML, so this formatting defect is snapshot-specific.
- **Table omission:** Oracle's first table includes a `RESERVOIR` colspan row; ours renders `RESERVOIR` as separate text, so the table hierarchy differs.
- **Substitution:** `2,000.0 mcg/mL` becomes both `2,000.0 mċg/mL` and `2,000:0 mcg/mL` (`LandingAI:222`; ours `356`).
- **Checkbox addition:** `Current Settings` and `BACLOFEN...` become checked controls (`ours:349,358`), while the oracle has no checkbox on this page.
- **Grounding inconsistency:** `$.document_metadata.table_reviews` marks `p3-l18-table` and `p3-l20-table` accepted, but `$.pages[2].table_structures[0:2]` remains invalid with empty cells.

Severity: **high** because drug concentration punctuation and table roles matter.

### Page 4

**Oracle:** lines 235 to 304. **Ours:** lines 366 to 541.

- **Table omission:** Oracle has two four-column infusion tables with 54 cells (`LandingAI:265-290`). Ours has no table.
- **Structure / reading order:** Column headers, times, fentanyl values, baclofen values, and session states are interleaved. `Beginning of Session` becomes `*BeginningofSessioni`; `Current Settings` becomes `wCurrentSettingst` (`ours:390,458`).
- **Checkbox additions:** Nine false controls are attached to BACLOFEN, FENTANYL, Monday through Sunday, dose values, alarm text, and `Page 2` (`ours:392-538`).
- **Substitution:** Table values lose cell association; a dose such as `0.75 mcg` is emitted as both a bullet and a checked checkbox (`ours:488-496`).
- **Formatting:** `ALARMS SETTINGS` loses the oracle heading semantics and an alarm interval becomes an unchecked control (`ours:525-531`).

Severity: **critical**. Values exist but their drug/time/session relationships are unreliable.

### Page 5

**Oracle:** lines 306 to 360. **Ours:** lines 542 to 607.

- **Omission:** Oracle `[REDACTED]` patient placeholder is absent (`LandingAI:323`).
- **Table omission:** Oracle's pump and catheter tables contain 37 cells (`LandingAI:330-350`); ours flattens both.
- **Reading order / structure:** `Calibration Constant`, `Reservoir Max Volume`, replacement date, and pump time are split and interleaved (`ours:561-578`). Paired Pump Segment/Tip Segment rows lose column ownership (`ours:585-598`).
- **Substitution / formatting:** `8637-40 NGV749621H` is concatenated as `8637-40NGV749621H` (`ours:559`).
- **Checkbox addition:** Calibration becomes `[x] Calibration` (`ours:600`); the oracle has no controls on page 5.

Severity: **critical** for structured device data.

### Page 6

**Oracle:** lines 362 to 518. **Ours:** lines 608 to 694.

- **Omission:** Oracle `Patient Name: [REDACTED]` and `DOB: [REDACTED]` become empty labels (`LandingAI:380-381`; ours `625-626`).
- **Substitution:** `(Years)` becomes `(ears)` (`ours:617`).
- **Structure / substitution:** Oracle's vertically OCR'd logo noise is normalized into `## CPM Consultants in Pain Medicine, PA` (`ours:633`). Under the chosen oracle rule this is a substitution, even if human review might prefer ours.
- **Checkbox additions:** Page header and patient-header text become controls; internal ID `p6-cv4` is published (`ours:621,629-631,658`).
- **Formatting:** Oracle bolds `ROS`; ours renders it as plain `ROS:`.

Severity: **high** because redaction placeholders and false controls are semantically important; narrative prose otherwise tracks the oracle closely.

### Page 7

**Oracle:** lines 520 to 601. **Ours:** lines 695 to 815.

- **Omission:** Oracle redaction placeholders in header/name/DOB are absent (`LandingAI:536-538`; ours `709-717`).
- **Substitution:** `musculoskeletal` becomes `musculoskelatal` (`ours:719`).
- **Substitution:** ICD codes `I95.1-458.0` and `I73.9-443.9` become `195.1-458.0` and `173.9-443.9` (`LandingAI:559,565`; ours `746,758`). These replace capital `I` with digit `1` and change medical identifiers.
- **Checkbox additions:** Page and DOB header fragments become three controls (`ours:708,722,724`).
- **Formatting:** Oracle section labels are bold; ours flattens most labels into plain text.

Severity: **critical** for identifier fidelity.

### Page 8

**Oracle:** lines 603 to 680. **Ours:** lines 816 to 925.

- **Omission:** `[REDACTED]` name and DOB placeholders are absent (`LandingAI:621-622`; ours `831-832`).
- **Checkbox additions:** Four false controls attach to page/header text and internal ID `p8-cv4` (`ours:829,837,839,877`).
- **Formatting/substitution:** `Alcohol history: Never drinks alcohol` becomes `Alcohol historyNever drinks alcohol` (`LandingAI:625`; ours `841`); `* NO KNOWN...` spacing is inconsistent.
- **Structure:** Oracle uses bold section labels for medications and physical-exam subsections; ours largely emits disconnected plain lines.

Severity: **high**, dominated by false controls and redaction omission.

### Page 9

**Oracle:** lines 682 to 760. **Ours:** lines 926 to 1026.

- **Omission:** Oracle redaction placeholders are absent (`LandingAI:700-701`; ours `941-942`).
- **Checkbox additions:** Four false controls attach to page/header text and `p9-cv4` (`ours:939,948,950,1001`).
- **Substitution / formatting:** Oracle `≥` becomes `>/=` for systolic and diastolic codes (`LandingAI:747-748`; ours `1017-1019`).
- **Substitution / omission:** Complete fax footer becomes `PG:: 0::/0-5` (`LandingAI:755`; ours `1025`).
- **Addition:** A stray `)` appears before procedure codes (`ours:1009`).

Severity: **high**. Main medication/dose content survives, but footer grounding and symbols are corrupted.

### Page 10

**Oracle:** lines 762 to 846. **Ours:** lines 1027 to 1126.

- **Reading order:** Our fax receipt footer appears at the top of the page (`ours:1029`), while the oracle places it at the bottom.
- **Substitution:** Oracle header `UGTAM0001/F DOB` becomes `JGTAM0001/F DOB` (`LandingAI:778`; ours `1037`).
- **Omission:** Oracle `[REDACTED]` name/DOB placeholders become empty (`LandingAI:783-784`; ours `1044-1045`).
- **Checkbox additions:** Four false controls include two header duplicates, page number, and uncertain internal ID `p10-cv4` (`ours:1041,1050,1052,1102`).
- **Formatting:** Oracle bolds Plan; ours uses plain `Plan:`.

Severity: **high**; narrative prose is otherwise close.

### Page 11

**Oracle:** lines 848 to end. **Ours:** lines 1127 to end.

- **Substitution:** `HUGTAM0001/F DOB (61 Years)` becomes `UGTAM0001/F DOP` plus a detached `Years)` (`LandingAI:864`; ours `1136-1138`).
- **Omission:** `[E-SIGNED]`, `[SIGNED]`, and `[ILLEGIBLE_SIGNATURE]` are absent (`LandingAI:874,878-879`).
- **Addition / structure:** Ours creates unsupported heading `## Bhal m` (`ours:1155`) instead of an attestation/signature element.
- **Checkbox additions:** Page/header fragments become three false controls (`ours:1140,1151,1153`).
- **Grounding:** LandingAI has two `attestation` nodes in its hierarchy; ours has no accepted attestation semantic type.

Severity: **critical** for signature and attestation workflows.

## Root-cause mapping to current source

### A. Review-required checkboxes are intentionally published

This is an **active current-code defect**, not only a stale snapshot issue.

1. `workflow._checkbox_candidates` unions Luna and OpenCV proposals, including OpenCV-only candidates (`src/agentic_extractor/workflow.py:1183-1227`).
2. `_checkboxes` correctly computes risks and marks non-consensus records `decision_status="review_required"`, but still appends every record (`workflow.py:1246-1353`).
3. `document_markdown_with_checkboxes` loops over every checkbox without filtering `decision_status` and inserts its state and label into canonical Markdown (`src/agentic_extractor/parse.py:123-146`).

That explains why 66 review-required records appear as facts. It also exposes internal fallback labels such as `p6-cv4` when no source label exists.

The page-2 false consensus has a second cause: local geometry-to-label association is too permissive. `_nearest_label` accepts a control up to 0.45 normalized page width left of a block and uses a loose vertical condition (`checkbox_vision.py:128-146`). The resulting `Home` record has control and label boxes at different vertical positions, yet IoU-based detector matching plus Luna agreement promotes it.

### B. Accepted table repairs do not persist in the canonical page model

This is also an **active current-code consistency defect**.

- `_apply_refinements` deep-copies raw pages, applies table reviews to the copy, builds Markdown, and returns only Markdown/refinement records/warnings (`pipeline.py:371-442`).
- `refine_local_parse` separately applies reviews to another audit copy only to obtain metadata (`pipeline.py:183-188`).
- Neither path replaces `local.pages` with the accepted table structures.

Therefore page-3 accepted reviews can produce Markdown tables while exported `pages[].table_structures` remains invalid and empty. This breaks audit traversal from semantic table back to canonical cells.

### C. Invalid local table geometry is now rejected, but recovery is insufficient

Current `normalize_table_result` rejects count mismatches, out-of-parent cells, overlaps, duplicate assignments, and unassigned OCR blocks. Current `apply_table_reviews` also requires non-overlapping logical slots and exact one-time source-block coverage (`table_structure.py:50-149,159-267`). These are good, already-changed safeguards.

However, the repair contract is all-or-nothing: every OCR block whose center lies in the table region must be assigned exactly once. False layout boundaries, marginalia inside a region, or missing OCR can make a valid visual table impossible to accept. The saved result has ten unresolved tables and seven fully missing oracle tables.

### D. Generic HTML table serialization is already fixed

The saved page-3 pipe tables are stale behavior. Current `_chunk_markdown` serializes generic table chunks as escaped HTML (`parse.py:244-263`). This should fix format for newly generated accepted tables, but it does not solve table discovery, repair, spans, or canonical persistence.

### E. Simple line heuristics cause semantic fragmentation

`reconstruct_layout` assigns types from punctuation and relative line height, then selects either basic column-major or top-to-bottom/left-to-right sorting (`parse.py:149-195`). `build_layout_chunks` groups only consecutive blocks of the same primitive type (`parse.py:202-241`).

These heuristics explain detached labels, repeated line fragments, lost marginalia/attestation semantics, and table values divorced from their columns. PP-DocLayout regions exist, but they are not the authoritative semantic hierarchy in the final contract.

### F. High-resolution routing can still expand to nearly a full page

Recent code stopped routing every dense form/table row automatically and targets invalid tables more precisely. That is an improvement after this snapshot. But `ambiguous_reading_order` still envelopes all grounded blocks into one padded candidate (`visual_routing.py:105-126`), and merging candidates can create near-page crops. The snapshot records 94% to 99% high-resolution area on every page, increasing cost and making Luna reconcile excessive evidence.

### G. High-confidence OCR errors evade correction

Examples such as `195.1-458.0`, `JGTAM0001/F DOB`, and `UGTAM0001/F DOP` have RapidOCR scores around 0.97 to 0.99 in `$.pages[*].blocks[*]`. Confidence-only routing therefore treats them as safe. A domain-aware identifier validator or Luna visual comparison must override raw confidence for risky patterns.

## Prioritized fixes and oracle-based acceptance tests

### P0. Publish only accepted checkbox facts

**Fix:** Canonical Markdown, accepted chunks, business-rule evaluation, and ordinary overlays must include only `automated` or explicitly `user_verified` controls. Review-required candidates stay in diagnostics/crops.

**Acceptance against this oracle:**

- pages 1 and 3 to 11: zero checkbox markers;
- page 2: exactly 28 markers with oracle labels/order/states;
- `Home`, Planned inpatient, and Diagnostic study are unchecked;
- no internal candidate ID appears in Markdown;
- a review-required record can never change canonical Markdown.

### P0. Correct control-label association

**Fix:** Require same-row geometric overlap, small horizontal gap, form-region membership, unique OCR label, and source-image crop evidence. Match by control geometry before trusting a semantic label. Do not count three models as independent when Luna sees an OpenCV-seeded crop/label.

**Acceptance:** `$.checkboxes[21]` must no longer resolve to checked Home. A labeled page-2 gold set must achieve 100% precision before recall improvements are accepted.

### P0. Persist accepted table repair as canonical data

**Fix:** Return refined pages/table structures from `_apply_refinements`, or apply reviews once to the owned canonical derived layer. Use the same accepted object to generate Markdown, JSON, HTML, annotated PDF, and manifest.

**Acceptance:** for every accepted table ID, all artifacts expose the same status, cells, source block IDs, row/column/spans, and HTML. No audit may say accepted while `pages[].table_structures` says invalid.

### P0. Recover the nine oracle tables

**Fix:** Treat table detection as semantic object recovery, not only cell-geometry validation. Use the page image, layout boundary, line/whitespace structure, and OCR block graph. Permit grounded exclusion of verified marginalia/noise rather than requiring every region-centered block in a cell. Preserve rejected evidence separately.

**Acceptance:** exactly 3/2/2/2 tables on pages 2/3/4/5, with all 146 oracle cells, correct row/column positions and spans, and no tables on other pages. Compare normalized cell text and spans to LandingAI JSON paths under `$.structure.children[*].children[*].children[*]`.

### P1. Add semantic-region validation

**Fix:** Validate high-risk identifiers, dates, units, redaction placeholders, footers, and attestations independently of OCR confidence.

**Acceptance examples:**

- `3/31/1983`, not `3|31983`;
- `I95.1` and `I73.9`, not `195.1`/`173.9`;
- `2,000.0 mcg/mL`, not `mċg` or `2,000:0`;
- `≥`, not `>/=`;
- oracle redaction placeholders on pages 5 to 10;
- page-11 E-signed/signed/illegible-signature semantics, with no `Bhal m`.

### P1. Build one accepted semantic hierarchy

**Fix:** Keep immutable OCR blocks as the evidence layer, but make a compact hierarchy of marginalia, text, tables, logos, and attestations the only source for canonical Markdown and downstream workflows.

**Acceptance:** element counts/types match the oracle for this document; each accepted element resolves to page coordinates and source evidence; repeated fax headers/footers are typed as marginalia.

### P1. Make routing object-specific

**Fix:** Do not expand ambiguous reading order into an all-block page envelope. Review the conflicting columns/regions only. Batch small semantic objects, not one page plus duplicated raw/layout context.

**Acceptance:** no routine crop covers more than the object plus bounded padding; GPT input falls materially below 420,092 tokens; all oracle discrepancies above still pass.

### P2. Turn this oracle into a regression fixture

**Fix:** Store normalized expected Markdown semantics (not protected or private source content in ordinary public tests unless authorized) as a page/table/control error manifest. Build evaluators for text substitutions, element presence/order, tables, checkboxes, grounding consistency, latency, and cost.

**Acceptance dashboard:**

- exact checkbox precision/recall/state accuracy;
- table detection recall, cell text accuracy, span accuracy, and TEDS;
- semantic-element type/order F1;
- identifier/date/unit exact match;
- grounding IoU and evidence-link completeness;
- end-to-end wall latency and GPT cost.

## Official format and behavior context

LandingAI says Parse converts documents into structured Markdown with hierarchical JSON and exact page/coordinate references.[^overview] Its documentation defines chunks as discrete elements such as text, tables, figures, marginalia, logos, and attestations.[^chunks] It documents HTML table markup, table/cell identifiers, and `rowspan`/`colspan`, plus cell-level row, column, span, page, and bounding-box grounding.[^markdown][^json]

Those first-party statements are used only to interpret the expected contract. The actual acceptance oracle in this report is the supplied LandingAI artifact.

## Limitations

- This audit intentionally accepts LandingAI output as truth by user decision; it does not independently adjudicate source-image disagreements.
- It covers one 11-page medical fax. Results cannot be generalized to all document classes.
- Our saved output predates some current uncommitted changes. The report explicitly labels known stale behavior, but only a fresh rerun can quantify the current build.
- Exact text accuracy, TEDS, and grounding IoU were not computed because the oracle is a product output rather than a dedicated metric fixture; the proposed regression conversion should do that deterministically.
- Cost units differ. LandingAI reports credits; ours reports GPT API dollars and excludes local hardware cost.

## Sources

[^overview]: LandingAI, [ADE Overview](https://docs.landing.ai/ade/ade-overview), accessed 2026-08-31. Defines the Parse-first workflow, structured Markdown/hierarchical JSON, element detection, and visual grounding.
[^parse]: LandingAI, [Parse Documents](https://docs.landing.ai/ade/parse), accessed 2026-08-31. Defines Parse output as structured Markdown, chunks, metadata, and exact page/coordinate references.
[^json]: LandingAI, [JSON Response for Parsing](https://docs.landing.ai/ade/ade-json-response), accessed 2026-08-31. Documents semantic chunks, Markdown, splits, table/cell identifiers, row/column/span fields, and page/bounding-box grounding.
[^markdown]: LandingAI, [Markdown Response](https://docs.landing.ai/ade/ade-markdown-response), accessed 2026-08-31. Specifies that table chunks use HTML and documents cell IDs plus `rowspan`/`colspan`.
[^chunks]: LandingAI, [Chunk Types](https://docs.landing.ai/ade/ade-chunk-types), accessed 2026-08-31. Defines text, table, marginalia, figure, logo, attestation, and related element types.
