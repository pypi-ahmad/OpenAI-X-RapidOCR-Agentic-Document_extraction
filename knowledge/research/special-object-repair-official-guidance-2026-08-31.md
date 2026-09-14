# Full-corpus special-object parity: official guidance

Research date: 2026-08-31

Scope: GroundTruth JSON/Markdown evaluation, PP-DocLayoutV3 taxonomy, and OpenAI vision/Structured Outputs constraints for the 11-page `Masked_Amerigroup_RealSolutions_1.pdf` parity run.

## Conclusions

1. **PP-DocLayoutV3 explains the logo gap, but not the whole failure.** Its published 25-label model configuration contains `image`, `header_image`, `footer_image`, and `seal`, but no `logo`, `checkbox`, `signature`, or `attestation` label.[1] Treating one of those absent concepts as requiring an exact PP-DocLayoutV3 class guarantees avoidable false negatives.
2. **The checkbox ground truth is not a set of JSON regions.** Inspection of the supplied LandingAI output finds 28 Markdown tokens (`5` checked `[x]`, `23` unchecked `[ ]`) but no top-level `checkbox` structure nodes. Checkbox parity must therefore derive stable objects from Markdown position, containing JSON region, and nearby label; it cannot use direct checkbox-box IoU unless the ground truth is augmented.
3. **The supplied JSON can directly localize logos and attestations.** It declares 11 pages and contains 2 normalized `logo` boxes, 2 normalized `attestation` boxes, 9 `table` boxes, 84 `text` boxes, and 74 `marginalia` boxes. Those types should be paired one-to-one by page and class before their content is scored.
4. **A valid JSON response is not evidence of correct extraction.** OpenAI Structured Outputs guarantees conformance to the requested JSON Schema.[2] OpenAI separately documents that vision can err on small text, precise spatial localization, and object counts.[3] Pixel/OCR/region validation is still required.
5. **“100%” needs an explicit metric contract.** Count equality alone can hide duplicates and misses. Freeze the GroundTruth interpretation, pair each prediction to at most one reference object, and report localization, classification/state, and content scores separately.

## GroundTruth contract for this corpus

The LandingAI JSON represents a document tree. Each page and child region has a page number, Markdown character range, and normalized box. The companion Markdown holds the content addressed by those ranges. This permits a two-layer evaluator:

1. **Region layer:** match JSON structure nodes one-to-one within the same page and canonical class.
2. **Content layer:** score only the content attached to the matched pair by slicing the canonical Markdown range or using the equivalent app field.

Checkboxes are the exception in this corpus: they occur as Markdown tokens inside text/table content, not as independent grounded nodes. Materialize each as a derived reference record such as `(page, containing_region_id, ordinal, normalized_label, state)`. If pixel localization is required, annotate it separately rather than inventing a box from the token.

Do not compare the app Markdown to GroundTruth as one undifferentiated string. OmniDocBench's official evaluation similarly represents localized components with recognition text and reading order, and evaluates text, tables, layout, and reading order with distinct metrics.[4]

## One-to-one matching and metrics

COCO's official evaluator operates per image and category, computes region IoU, prevents repeated matching of a non-crowd ground-truth object, and accumulates precision and recall. Its standard detection thresholds span IoU 0.50 through 0.95.[5] Adapt that principle to pages and document classes; do not infer correctness from `predicted_count == expected_count`.

Recommended deterministic sequence:

1. Normalize all boxes to the same page coordinate system and reject invalid/out-of-page boxes.
2. Partition by page and canonical class. Map known upstream aliases before matching, not afterward.
3. Build eligible edges using a documented overlap rule. Resolve them globally or deterministically so each prediction and reference appears in at most one pair.
4. Mark matched pairs as true positives, unmatched predictions as false positives, and unmatched references as false negatives.
5. Score type-specific semantics only on matched pairs. Preserve localization and semantic failures as separate fields.
6. Aggregate over all 11 pages. Report both micro totals and per-page results so one dense page cannot hide a failed page.

| Type | Region pairing | Matched-pair content score | Required parity report |
| --- | --- | --- | --- |
| Text | Same page/class + box overlap | Unicode-normalized edit distance; optionally exact text | TP/FP/FN, precision/recall/F1, normalized edit distance, exact-match rate |
| Table | Same page + table-box overlap | TEDS on normalized HTML plus normalized cell-text edit distance | Table localization P/R/F1, TEDS, cell-text score |
| Checkbox | Derived containing region + nearby normalized label/ordinal | Exact `checked`/`unchecked` state; `uncertain` is not a correct match | Control P/R/F1 and state confusion matrix |
| Logo | Same page + final canonical `logo` class + overlap | Presence/class correctness; optional normalized wordmark text as a separate diagnostic | Logo P/R/F1 and per-reference match ledger |
| Attestation | Same page + agreed annotation unit + overlap | Normalized associated Markdown/text plus subtype if GroundTruth defines one | Attestation P/R/F1, content score, grouping errors |

OmniDocBench's official configuration uses edit distance for text blocks and reading order, and TEDS plus edit distance for tables.[4] PubTabNet introduced TEDS specifically to compare predicted and reference HTML table trees, with cell content incorporated into node similarity.[6] Therefore TEDS is appropriate for table structure, but it does not replace table-region localization or an auditable cell-text comparison.

For attestations, first freeze the unit represented by the two supplied nodes. A clause, signature mark, and signed-name line can be one grouped attestation or multiple objects; changing that unit between repair cycles makes `1 / 2` uninterpretable.

## PP-DocLayoutV3 implications

The official model artifact lists exactly these 25 labels: `abstract`, `algorithm`, `aside_text`, `chart`, `content`, `display_formula`, `doc_title`, `figure_title`, `footer`, `footer_image`, `footnote`, `formula_number`, `header`, `header_image`, `image`, `inline_formula`, `number`, `paragraph_title`, `reference`, `reference_content`, `seal`, `table`, `text`, `vertical_text`, and `vision_footnote`.[1]

Paddle's official layout documentation describes outputs as class ID, label, confidence, and coordinates, and exposes per-class thresholds, NMS, box expansion, and overlap-merge controls.[7] These are proposal controls, not a promise that an absent semantic class will be produced.

Consequences:

- Do not require PP-DocLayoutV3 label `image` before accepting independently grounded logo evidence. Consider `image`, `header_image`, `footer_image`, text-like proposals, and a dedicated visual candidate path.
- Do not expect PP-DocLayoutV3 to produce checkbox or attestation objects. Checkbox geometry/state and attestation grouping need downstream detectors or semantic refinement.
- Preserve the PP box/score/label as provenance. Final canonical class may differ, but unsupported semantic output must remain reviewable.
- Threshold or merge-mode tuning can recover suppressed proposals; it cannot add missing taxonomy classes.

## Checkbox recovery

OpenCV documents adaptive thresholding for locally varying illumination, Otsu thresholding as a data-derived global alternative, and morphology whose effect depends on kernel shape and size.[8][9] Its shape API provides contour hierarchy, polygon approximation, and connected-component bounding statistics.[10]

Use those primitives to detect small near-square controls independently of OCR, deduplicate candidates spatially, and then classify the interior as checked, unchecked, or uncertain. Recover broken borders with scale-relative morphology. Associate nearby OCR/table text after visual detection; missing OCR must not erase a grounded control. This separation directly supports the evaluator's independent localization and state metrics.

## OpenAI vision and Structured Outputs implications

OpenAI documents four image detail levels. For GPT-5.6 Luna, `high` fits images within 2048 × 2048 pixels and a 2,500-patch budget; `original` has materially larger limits. The same guide says `high` is standard high-fidelity understanding when precise original coordinates are not required, while fine detail, OCR, and small-object detection benefit from `original` when supported.[3]

This project's fixed full-page `detail="high"` contract means tiny page objects may be reduced before inference. A compatible repair is to keep the required high-detail full page and add high-detail candidate crops so a checkbox, logo, or signature occupies more pixels. Returned crop coordinates must be mapped back to page coordinates and validated locally.

OpenAI also states that vision may struggle with small text and precise spatial localization, may return approximate object counts, may resize images, and may generate incorrect descriptions.[3] Therefore:

- Ask for grounded proposals, not authoritative truth.
- Include page, normalized coordinates, canonical class, class-specific attributes, and an evidence note in the schema.
- Permit `uncertain`/abstention.
- Validate every proposal against page bounds, crop pixels, raw OCR, and overlap with accepted objects.
- Validate siblings independently so one malformed proposal does not discard valid objects.

Structured Outputs makes those fields structurally reliable, but its guarantee is schema adherence.[2] It does not remove the documented vision limitations or prove that an object exists.

## Iteration gate

OpenAI's eval guidance requires representative test inputs and ground-truth outputs, then evaluates generated samples against explicit testing criteria.[11] For every real repair cycle, keep the same PDF, all 11 pages, the same normalized GroundTruth records, and the same thresholds. Persist a match ledger containing reference ID, prediction ID, page, overlap, semantic score, and failure reason. Stop only when every required class has zero FP/FN and every required matched-pair semantic score meets its frozen acceptance threshold.

## Primary sources

1. PaddlePaddle, PP-DocLayoutV3 official model artifact, `inference.yml`: https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/blob/main/inference.yml
2. OpenAI, "Structured model outputs": https://developers.openai.com/api/docs/guides/structured-outputs
3. OpenAI, "Images and vision" (detail levels, model sizing, and limitations): https://developers.openai.com/api/docs/guides/images-vision
4. OpenDataLab, OmniDocBench official repository and end-to-end configuration: https://github.com/opendatalab/OmniDocBench and https://github.com/opendatalab/OmniDocBench/blob/main/configs/end2end.yaml
5. COCO API, official `COCOeval` implementation: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
6. IBM Research, PubTabNet and TEDS: https://github.com/ibm-aur-nlp/PubTabNet and https://arxiv.org/abs/1911.10683
7. PaddleX, "Layout Detection Module Tutorial": https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html
8. OpenCV, "Image Thresholding": https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
9. OpenCV, "Morphological Transformations": https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
10. OpenCV, "Structural Analysis and Shape Descriptors": https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
11. OpenAI, "Working with evals": https://developers.openai.com/api/docs/guides/evals
