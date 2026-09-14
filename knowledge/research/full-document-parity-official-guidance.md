# Official guidance for full-document Parse parity

Research date: 2026-08-31

Scope: first-party OpenAI, PaddlePaddle/PaddleOCR, and RapidAI sources relevant to improving full-document Parse accuracy, latency, and token efficiency. This note does not claim that any individual technique will match LandingAI; each change still needs measurement against the project's labeled and LandingAI-derived oracle data.

## Executive findings

1. **Keep the three responsibilities separate.** RapidOCR's official result is line-level OCR evidence: polygons, text, recognition scores, optional word results, and stage timings. PP-DocLayoutV3 supplies document regions and reading order. Luna should validate and refine those grounded objects. RapidOCR alone is not a document-layout or table-structure system. [RapidOCR output source](https://github.com/RapidAI/RapidOCR/blob/e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7/python/rapidocr/utils/output.py) [PP-DocLayoutV3 model card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
2. **Use one warm instance per local model configuration.** RapidOCR lazily initializes detection, classification, and recognition models behind locks on its instance. Recreating the instance discards that warm state. [RapidOCR implementation](https://github.com/RapidAI/RapidOCR/blob/e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7/python/rapidocr/main.py)
3. **Use PP-DocLayoutV3's order and geometry rather than rebuilding global order from OCR lines.** Its first-party model card says it predicts multi-point layout boxes and logical reading order for skewed and curved documents in one forward pass, specifically to reduce cascading errors. [PP-DocLayoutV3 model card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
4. **Treat tables as detected objects with a specialized pipeline.** PP-StructureV3 separates layout detection, table classification, cell detection, wired/wireless structure recognition, OCR, and matching. Its troubleshooting guide maps each failure class to the module that owns it. This supports object-specific repair rather than full-page or full-document regeneration. [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/pipeline_usage/PP-StructureV3.en.md) [Table Recognition v2](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/pipeline_usage/table_recognition_v2.en.md)
5. **Batch page evidence inside bounded synchronous Responses requests, not through OpenAI's Batch API.** The Responses API accepts multiple image inputs; the Batch API has a `24h` completion window and is intended for asynchronous jobs, so it is not an interactive latency optimization. [Images and vision](https://developers.openai.com/api/docs/guides/images-vision) [Batch API](https://developers.openai.com/api/docs/guides/batch)
6. **Make the reusable Luna prefix byte-for-byte stable.** OpenAI prompt caching matches the rendered prefix. Stable developer instructions, schemas, and shared definitions belong first; page-specific OCR/layout evidence and images belong later. GPT-5.6 has a 1,024-visible-token minimum cacheable prefix and supports explicit cache breakpoints. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
7. **Use Structured Outputs for shape, then validate evidence in application code.** Strict schema adherence prevents malformed result objects but does not prove that a value is grounded. Every accepted correction still needs source-ID, page, and geometry checks against immutable OCR/layout evidence. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

## OpenAI Responses API

### Structured Outputs

Official behavior:

- Structured Outputs is preferred over JSON mode because it enforces the supplied schema. The Python SDK supports `client.responses.parse(..., text_format=YourPydanticModel)` and returns `output_parsed`. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- Under strict schemas, all object properties must be required, every object must set `additionalProperties: false`, and optional values are represented by a union with `null`. Unsupported or incomplete JSON Schema is rejected before generation. [Supported schemas](https://developers.openai.com/api/docs/guides/structured-outputs#all-fields-must-be-required) [Strict mode](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)
- A schema-constrained response still requires application validation for domain suitability. Schema correctness and evidence correctness are different checks. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

Project implication:

- Keep a single versioned response schema for page/object refinements. Include page number, source IDs, operation type, replacement content, confidence/status, and abstention/review reason as required fields; use nullable types rather than omitting fields.
- Reject any returned object whose source IDs do not exist, whose page differs from its evidence, or whose geometry is outside the source page. Do not let valid JSON bypass grounding validation.
- Keep table, checkbox, and text corrections object-scoped. A small, typed repair surface constrains accidental changes to unrelated page content.

### Image inputs and request grouping

Official behavior:

- Responses can receive several `input_image` content items in one request. Images may be URLs, Base64 data URLs, or uploaded file IDs. The documented request limits are up to 1,500 images and 512 MB total, while images and text must still fit the model context window. [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)
- For `gpt-5.6-luna`, `low` fits within 512×512, `high` fits within 2048×2048 and 2,500 patches, and `original` has no patch-budget limit within its dimension limit. OpenAI describes `high` as standard high-fidelity vision and `original` as appropriate for dense, spatially sensitive, OCR, or precise-coordinate tasks. [Image detail levels](https://developers.openai.com/api/docs/guides/images-vision#choose-an-image-detail-level)
- OpenAI's Batch API requires an uploaded JSONL file and currently uses a `24h` completion window. Output order is not guaranteed; `custom_id` is required for correlation. [Batch API](https://developers.openai.com/api/docs/guides/batch)

Project implication:

- Preserve the product invariant that every selected page is visually reviewed at `high` detail. Group a small bounded number of pages per synchronous response and give every page an explicit page ID. Determine the group size from measured context tokens, latency, and schema reliability; the official maximum image count is not an operating target.
- Send `original`-detail crops only for evidence objects whose tiny marks, dense cells, or precise spatial relationships cannot be resolved at page-level `high`. Map crop coordinates back to page coordinates before accepting a correction.
- Do not migrate the interactive Parse path to the Batch API. It can be considered only for explicitly asynchronous/offline evaluations.

### Prompt caching and token telemetry

Official behavior:

- Prompt caching is enabled by default on supported models and reuses only an identical rendered prefix. That rendered context includes developer instructions, schema/tool definitions, conversation content, images, and documents. A change before a cache breakpoint prevents reuse after that point. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- For GPT-5.6 and later, the minimum cacheable prefix is 1,024 visible input tokens. Explicit mode uses `prompt_cache_options.mode: "explicit"` and `prompt_cache_breakpoint` on a supported content block; top-level `instructions` cannot contain an explicit breakpoint. The current TTL value is `30m`. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- OpenAI recommends putting stable developer instructions and shared reference material first and changing content later. Model, tools, `text.format`, reasoning effort, and other settings participate in the rendered prefix. [Prompt caching best practices](https://developers.openai.com/api/docs/guides/prompt-caching#best-practices)
- GPT-5.6 cache writes cost 1.25× uncached input and reads cost 0.1× uncached input. Cached usage must therefore distinguish uncached input, cache writes, and cache reads rather than labeling all input tokens identically. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)

Project implication:

- Put the stable prompt resource, invariant instructions, and unchanged strict schema before dynamic document/page evidence. Keep model, `reasoning_effort="medium"`, schema name, schema order, and prompt prefix constant across page groups.
- Add an explicit breakpoint only when the stable prefix is long enough and will be reused. Keep per-document names, page numbers, OCR text, layout objects, and images after it.
- Record API-reported `input_tokens`, cached-input tokens, output tokens, call count, and per-call/page attribution. Do not infer cached tokens from character count.

## PaddleOCR, PP-DocLayoutV3, and PP-StructureV3

### Layout and reading order

Official behavior:

- PP-DocLayoutV3 directly predicts multi-point bounding boxes and logical reading order for non-planar, skewed, and curved documents in one forward pass. It is presented as the layout module used by PaddleOCR-VL 1.5/1.6. [PP-DocLayoutV3 model card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
- PP-StructureV3 improves layout-region detection, table and formula recognition, multi-column reading-order recovery, chart understanding, and Markdown conversion. It combines a layout module, general OCR, and optional preprocessing, table, seal, formula, and chart modules. [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/pipeline_usage/PP-StructureV3.en.md)
- PaddleOCR's published benchmark tables explicitly separate model inference time from preprocessing and postprocessing time. Reported model-only timings cannot be treated as end-to-end Parse latency. [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/pipeline_usage/PP-StructureV3.en.md)

Project implication:

- Persist PP-DocLayoutV3 polygons, scores, labels, and predicted reading order alongside RapidOCR evidence. Build canonical ordering from that order, with deterministic geometry fallback only when order is absent or invalid.
- Measure rendering, preprocessing, layout inference, layout postprocessing, OCR detection/classification/recognition, Luna request/response, validation, and artifact generation separately. Compare like-for-like end-to-end timings, not Paddle's model-only benchmark numbers.
- Avoid running the full PP-StructureV3 OCR path in parallel with mandatory RapidOCR unless an evaluation proves the duplicate OCR adds value. The useful seams here are the layout/order output and candidate-gated specialized object pipelines.

### Table reconstruction

Official behavior:

- PP-StructureV3's table recognition is optional and specialized. The Table Recognition v2 pipeline contains layout detection, table classification, wired and wireless table-structure recognition, wired and wireless cell detection, document preprocessing, and general OCR. [Table Recognition v2](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/pipeline_usage/table_recognition_v2.en.md)
- PaddleOCR's troubleshooting map assigns missed table regions to layout detection, wrong wired/wireless choice to table classification, cell-location errors to cell detection, structure errors to structure recognition, text misses to text detection, and wrong text to text recognition. [Table Recognition v2 secondary development](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/pipeline_usage/table_recognition_v2.en.md#4-secondary-development)

Project implication:

- Gate the expensive table pipeline on credible layout/table evidence. Once gated in, retain the table region, classification, cell boxes, structure output, OCR source IDs, and matching decisions separately so evaluation can identify the failing stage.
- Persist an accepted Luna repair back into the canonical table object, not only rendered Markdown. Validate row/column topology and ensure every cell is supported by OCR/layout evidence before publishing HTML.
- Fine-tune only the module demonstrated by labeled failures; PaddleOCR explicitly recommends diagnosing the failing table stage rather than treating table recognition as one opaque score.

## RapidOCR constraints that affect the canonical contract

Official behavior from the current RapidOCR source:

- `RapidOCROutput` exposes `boxes`, `txts`, `scores`, optional `word_results`, `elapse_list`, and total `elapse`. It does not expose document regions, page reading order, table topology, or page-orientation metadata. [RapidOCR output source](https://github.com/RapidAI/RapidOCR/blob/e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7/python/rapidocr/utils/output.py)
- The engine lazily creates detection, classification, and recognition models under per-model locks. Calls on the same engine reuse those initialized models. [RapidOCR implementation](https://github.com/RapidAI/RapidOCR/blob/e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7/python/rapidocr/main.py)
- The current default configuration uses PP-OCRv6 small detection/recognition, enables detection/classification/recognition and preprocessing, defaults `text_score` to `0.5`, and defaults word boxes off. ONNX Runtime CUDA is disabled unless explicitly configured. [RapidOCR default configuration](https://github.com/RapidAI/RapidOCR/blob/e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7/python/rapidocr/config.yaml)
- `filter_by_text_score` removes lines below the configured recognition threshold from the returned result. Therefore, a downstream uncertainty review cannot recover discarded low-confidence lines from that result. [RapidOCR implementation](https://github.com/RapidAI/RapidOCR/blob/e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7/python/rapidocr/main.py)

Project implication:

- Construct RapidOCR once per stable configuration and reuse it across pages and runs; bound concurrency around the warm engine according to verified thread safety rather than creating one model stack per page.
- Set the OCR acceptance threshold low enough to preserve uncertainty evidence. Apply the product's publish/review threshold after immutable raw evidence capture, not by discarding lines at engine output.
- Enable and store word boxes only where atomic grounding requires them; otherwise line polygons are the cheaper canonical unit. Never synthesize page orientation from RapidOCR's crop classifier because the public combined output does not provide it.
- Treat RapidOCR Markdown helpers as convenience rendering of OCR lines, not as evidence of document semantics. Canonical headings, columns, tables, forms, and reading order must come from the layout/structure/refinement layers.

## Evidence-led implementation order

1. **Instrument first:** verify per-stage timings, per-request image/text tokens, cache reads/writes, object counts, and failures for all 11 pages.
2. **Correct ordering:** consume PP-DocLayoutV3's predicted order and polygons; reject invalid/overlapping regions deterministically.
3. **Correct tables:** run the specialized table path only on credible candidates and evaluate region detection, classification, cells, structure, OCR, and matching separately.
4. **Reduce Luna context without weakening coverage:** group a bounded number of page images at `high`, pair them with compact evidence keyed by page/source ID, and use `original` crops only for unresolved objects.
5. **Stabilize the cacheable prefix:** fixed prompt resource, fixed strict schema, fixed model and medium reasoning before dynamic evidence.
6. **Validate before publishing:** schema validation, source-ID/page/geometry validation, object-specific semantic checks, then canonical persistence. Unsupported corrections abstain or require review.

## What official sources do not establish

- They do not provide a universally optimal number of pages per Luna request. Choose it from this document's real 11-page measurements.
- They do not show that `original` detail improves this document enough to justify its token cost. Use it only after crop-level evaluation.
- They do not establish that PP-StructureV3 end-to-end should replace this app's mandatory RapidOCR-first pipeline. Its layout and table modules are relevant, but duplicate OCR would need evidence.
- Paddle's published accuracy and inference-time values use its own datasets and exclude some end-to-end work; they are not parity evidence against the supplied LandingAI output.

## Primary sources

- OpenAI, [Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- OpenAI, [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)
- OpenAI, [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- OpenAI, [Batch API](https://developers.openai.com/api/docs/guides/batch)
- PaddlePaddle, [PP-DocLayoutV3 model card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
- PaddleOCR, [PP-StructureV3 pipeline](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/pipeline_usage/PP-StructureV3.en.md)
- PaddleOCR, [General Table Recognition v2 pipeline](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/pipeline_usage/table_recognition_v2.en.md)
- RapidAI, [`RapidOCROutput`](https://github.com/RapidAI/RapidOCR/blob/e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7/python/rapidocr/utils/output.py)
- RapidAI, [`RapidOCR` implementation](https://github.com/RapidAI/RapidOCR/blob/e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7/python/rapidocr/main.py)
- RapidAI, [default configuration](https://github.com/RapidAI/RapidOCR/blob/e7edb012372f9ec5cf9e38a1e1e7b6abd489a2e7/python/rapidocr/config.yaml)
