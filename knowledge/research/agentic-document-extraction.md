# Agentic document extraction: ADE, RapidOCR, and ONNX

Research date: 2026-08-30

## Scope and evidence standard

This report examines a hybrid document-extraction system built from LandingAI Agentic Document Extraction (ADE), RapidAI RapidOCR, and ONNX Runtime. "Verified" means the statement appears in first-party documentation or source repositories linked below. "Synthesis" means an architectural or evaluation recommendation derived from those facts; it is not a vendor claim.

No comparative accuracy, latency, or cost claim should be treated as established without a workload-specific benchmark. LandingAI's published accuracy figures are vendor-reported, and RapidOCR's repository describes capabilities rather than proving performance on this project's documents.

## Executive findings

1. **Verified:** ADE is a document-intelligence service whose workflow includes Parse, Extract, Classify, Section, and Split. Parse produces Markdown plus hierarchical JSON, chunks, metadata, and page/coordinate grounding. Extract applies a user-defined schema to parsed Markdown. [LandingAI overview](https://docs.landing.ai/ade/ade-overview) [Parse docs](https://docs.landing.ai/ade/parse) [Extract docs](https://docs.landing.ai/ade/ade-extract)
2. **Verified:** RapidOCR is an open-source, offline-capable OCR deployment project. Its Python quick start uses `rapidocr` with `onnxruntime`; output exposes detected boxes, recognized text, scores, and timing. Its models originated from PaddleOCR conversions, and current configuration separates detection, classification, and recognition. [RapidOCR repository](https://github.com/RapidAI/RapidOCR) [Quick start](https://rapidai.github.io/RapidOCRDocs/main/quickstart/) [Model list](https://rapidai.github.io/RapidOCRDocs/main/model_list/)
3. **Verified:** ONNX is a typed computational-graph format with versioned operator sets. ONNX Runtime validates model conformance, optimizes graphs, and assigns supported nodes or subgraphs to ordered hardware Execution Providers. Application owners remain responsible for validating model accuracy, performance, and suitability. [ONNX concepts](https://onnx.ai/onnx/intro/concepts.html) [ONNX Runtime overview](https://onnxruntime.ai/docs/) [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
4. **Synthesis:** These tools occupy different layers. RapidOCR plus ONNX Runtime is a local text-recognition substrate; ADE is a higher-level parsing and schema-extraction service. The application should preserve this boundary and route documents according to privacy, complexity, confidence, latency, and cost; neither path is universally best.
5. **Synthesis:** Evaluation must score the whole pipeline and each stage separately. Character accuracy alone cannot establish layout fidelity, field correctness, grounding quality, schema compliance, or safe agent behavior.

## What "agentic document extraction" means here

### Verified: LandingAI ADE

LandingAI documents ADE as a platform that converts documents into structured data. Its published API family has distinct operations:

- **Parse:** converts documents into Markdown and hierarchical JSON, identifies text, tables, form fields, images, and other chunks, and associates chunks with page numbers and coordinates. LandingAI calls Parse the required first step for ADE workflows.
- **Extract:** retrieves repeated fields using a caller-defined schema. It accepts ADE Parse Markdown directly or by URL; LandingAI warns that edited or generic Markdown can reduce extraction quality because Parse output includes IDs, anchors, chunk tags, and metadata used by Extract.
- **Classify:** labels pages using caller-defined classes and can run without Parse.
- **Section:** builds a hierarchical table of contents with section levels and chunk references from parsed content.
- **Split:** separates multi-document files after classification and parsing.

Sources: [ADE overview](https://docs.landing.ai/ade/ade-overview), [Parse](https://docs.landing.ai/ade/parse), [Extract](https://docs.landing.ai/ade/ade-extract), [Parse JSON response](https://docs.landing.ai/ade/ade-json-response).

The Parse response documentation lists top-level `markdown`, `chunks`, `splits`, `grounding`, and `metadata` fields. Grounding maps chunk IDs to page numbers and bounding boxes. Metadata includes processing information such as credit usage, duration, filename, job ID, page count, and version. Partial-content responses can identify failed pages. [Parse JSON response](https://docs.landing.ai/ade/ade-json-response)

LandingAI says newer extraction models support semantic alternative names, formatting instructions, large schemas, long content, and cross-page table reconstruction. These are version-specific service capabilities, not guarantees for every document. [Extract](https://docs.landing.ai/ade/ade-extract)

LandingAI reports 99.16% accuracy on DocVQA in its product overview. This is a first-party benchmark claim; the page does not, by itself, establish accuracy for custom schemas, handwriting, multilingual scans, domain-specific tables, or this repository's workload. [ADE overview](https://docs.landing.ai/ade/ade-overview)

### Verified: RapidOCR

RapidOCR describes itself as free, open-source OCR for fast, cross-platform, multilingual, offline deployment. The project originated by converting PaddleOCR models to ONNX for portable inference and exposes implementations across several languages. [RapidOCR repository](https://github.com/RapidAI/RapidOCR)

The documented Python path is:

```python
from rapidocr import RapidOCR

engine = RapidOCR()
result = engine(image)
```

The quick start installs `rapidocr` and `onnxruntime`. `RapidOCROutput` exposes `boxes`, `txts`, `scores`, `word_results`, `elapse_list`, and total `elapse`. This makes geometry, recognition confidence, and stage timing available to downstream code. [RapidOCR quick start](https://rapidai.github.io/RapidOCRDocs/main/quickstart/)

RapidOCR configuration treats detection (`Det`), text-line orientation classification (`Cls`), and recognition (`Rec`) as separate model roles. Official docs show ONNX Runtime as the default engine in a current configuration example, while the repository also documents ONNX Runtime GPU, TensorRT, Paddle, OpenVINO, PyTorch, and MNN development environments. [Model list](https://rapidai.github.io/RapidOCRDocs/main/model_list/) [RapidOCR repository](https://github.com/RapidAI/RapidOCR)

The repository states default Chinese and English recognition, with additional models listed separately. Model selection therefore belongs in explicit configuration and evaluation, not in an assumption that one default recognizes every language equally. [RapidOCR repository](https://github.com/RapidAI/RapidOCR) [Model list](https://rapidai.github.io/RapidOCRDocs/main/model_list/)

### Verified: ONNX and ONNX Runtime

ONNX stores models as typed graphs. Graphs carry operator-set versions, and operators follow the newest definition at or below the graph's declared opset. ONNX does not implicitly cast incompatible tensor types. Shape inference can improve runtime memory planning, though custom operators may limit inference. [ONNX concepts](https://onnx.ai/onnx/intro/concepts.html)

ONNX Runtime applies graph optimizations, then partitions the graph based on available accelerator capabilities. Execution Providers abstract CPU, CUDA, TensorRT, DirectML, OpenVINO, QNN, CoreML, and other hardware backends. Provider order is priority order; unsupported nodes fall through to later providers, such as CPU. [ONNX Runtime overview](https://onnxruntime.ai/docs/) [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)

Graph optimization can run online at session creation or offline, with the optimized graph saved for later use. Offline optimization can reduce repeated startup overhead. [Graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)

ONNX Runtime provides static and dynamic integer quantization tools. Its documentation warns that quantization speedups depend on model and hardware, quantize/dequantize overhead can make performance worse, and accuracy loss must be debugged and measured. GPU INT8 benefits require suitable hardware and backend support. [Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

ONNX Runtime explicitly places responsibility for accuracy, performance, suitability, and untrusted-model safety on the application owner. Runtime format validation is not a security or quality guarantee. [ONNX Runtime overview](https://onnxruntime.ai/docs/)

## Recommended system architecture

The following is synthesis from the verified product boundaries.

```text
Document intake
  -> validate type, size, page count, tenant, and policy
  -> normalize pages and preserve original bytes/hash
  -> route
       local path: RapidOCR -> layout/order reconstruction -> schema extraction
       ADE path:    Parse -> optional Split/Classify/Section -> Extract
  -> normalize both paths into one evidence-bearing result contract
  -> validate schema and business rules
  -> confidence/risk decision
       accept | human review | retry alternate path | reject
  -> audit record and downstream delivery
```

### Routing policy

- Prefer **local RapidOCR** when data cannot leave the environment, documents are layout-simple, OCR output is sufficient, offline operation matters, or predictable local cost/latency dominates.
- Prefer **ADE** when reading order, complex tables, forms, page-level grounding, document splitting, hierarchical sections, or reusable schema extraction are load-bearing requirements.
- Use a **cascade** only when measurement supports it: local OCR first, then ADE for low-confidence or complex pages. Cascades add latency and two-system reconciliation, so complexity detection and thresholds must earn their cost.
- Do not silently send locally processed documents to a cloud fallback. Routing must enforce data-residency and consent policy before content leaves the trust boundary.

### Canonical result contract

Normalize both engines without discarding native evidence:

```json
{
  "document_id": "stable-id",
  "engine": "rapidocr-or-ade",
  "engine_version": "recorded-version",
  "fields": {},
  "blocks": [
    {
      "text": "...",
      "page": 0,
      "polygon_or_box": [],
      "confidence": null,
      "source_id": "native-chunk-or-local-id"
    }
  ],
  "warnings": [],
  "failed_pages": [],
  "timings_ms": {},
  "raw_artifact_ref": "immutable-internal-reference"
}
```

**Synthesis:** Confidence must remain nullable and typed by source. RapidOCR recognition scores and ADE confidence/grounding fields are not automatically calibrated to the same probability scale. Preserve raw scores, model version, and decision threshold used.

### Agent boundary

An agent can choose tools, request a retry, select a schema from an allowlist, and propose corrections. It should not invent fields, rewrite source evidence, broaden document access, select arbitrary URLs, or mark low-confidence output as verified. Structured tool inputs, bounded schemas, deterministic validators, and an auditable state machine matter more than free-form prompting.

## Evaluation plan

### Dataset

Build a versioned, rights-cleared dataset stratified by document type, source quality, language, page count, layout complexity, table type, handwriting, rotation, and sensitive-data class. Keep document-level train/development/test separation so pages from one document never cross splits. Include clean originals, realistic scans, and intentionally hard cases.

Ground truth should include:

- exact field values and types;
- page and polygon/bounding-box evidence for each value;
- reading order and structural hierarchy where relevant;
- table cells, row/column spans, and cross-page continuity;
- document/page class and split boundaries;
- explicit absent fields, illegible regions, and acceptable abstentions.

### Metrics by layer

| Layer | Primary measures | Why |
| --- | --- | --- |
| OCR | CER/WER, normalized edit distance, detection precision/recall/IoU | Separates recognition and localization failures |
| Layout | reading-order accuracy, block-type F1, hierarchy/tree similarity | Text can be correct while structure is wrong |
| Tables | cell value F1, row/column alignment, span accuracy, table exact match | Flat text scores hide structural corruption |
| Extraction | field precision/recall/F1, exact match, numeric/date tolerance, schema-valid rate | Measures business output |
| Grounding | page accuracy, box IoU, evidence coverage, unsupported-field rate | Measures traceability and hallucination risk |
| Classification/split | macro-F1, boundary precision/recall, document reconstruction rate | Measures routing correctness |
| Confidence | risk-coverage curve, calibration error, selective accuracy | Tests whether abstention thresholds are useful |
| System | p50/p95/p99 latency, pages/minute, failure rate, retry rate, cost/page, memory | Measures deployability |
| Agent | invalid tool-call rate, unauthorized-route rate, recovery success, human override rate | Measures orchestration safety |

### Experiments

1. Benchmark RapidOCR and ADE Parse independently on the same frozen pages.
2. Benchmark extraction separately using gold transcription versus each parser's output. This distinguishes extraction errors from upstream OCR/layout errors.
3. Compare RapidOCR CPU and candidate ONNX Runtime Execution Providers on the actual deployment hardware. Record provider assignment and fallbacks.
4. Test FP32 against any optimized or quantized model for both accuracy and performance. Reject optimization when gains do not survive the quality threshold.
5. Evaluate local-only, ADE-only, and cascade routing end to end. Include cloud cost, transfer time, retries, and human-review load.
6. Run perturbation suites for blur, skew, compression, low contrast, rotation, truncation, prompt-like text inside documents, malformed files, and oversized inputs.

### Acceptance gates

Set thresholds from business loss, not generic benchmarks. At minimum:

- zero schema-invalid outputs reaching downstream systems;
- zero silent page failures;
- every accepted critical field has source evidence or an explicit policy exemption;
- measured false-accept rate below the domain's risk limit;
- latency, cost, and memory within deployment budgets on target hardware;
- no cloud route for documents prohibited by policy;
- reproducible results tied to engine, model, schema, and configuration versions.

## Risks and open questions

- **Vendor comparability:** ADE and RapidOCR publish different output contracts and confidence semantics. Direct score comparisons are invalid until calibrated on shared data.
- **Cloud boundary:** ADE processing involves an external service. Security, retention, residency, contractual terms, and regional endpoints require separate current review before production use.
- **Model provenance:** RapidOCR distinguishes engineering-code copyright from OCR model copyright. Confirm every selected model's license and redistribution terms.
- **Fallback opacity:** ONNX Runtime may assign unsupported nodes to a lower-priority provider. Measure actual provider placement instead of assuming full GPU/NPU execution.
- **Document prompt injection:** Text in a document is untrusted data. If an LLM agent consumes parsed content, tool policy must prevent embedded text from changing system instructions or authorization.
- **Ground-truth cost:** Layout and table annotations are expensive but necessary. Field-only labels cannot diagnose whether failures originate in detection, reading order, OCR, or extraction.
- **Version drift:** Hosted ADE models and local RapidOCR/ONNX components can change independently. Pin what can be pinned, record what cannot, and run regression gates before promotion.

## Source register

All sources below are first-party project, vendor, or specification documentation.

### LandingAI

- [ADE documentation home](https://docs.landing.ai/)
- [ADE overview and API roles](https://docs.landing.ai/ade/ade-overview)
- [Parse workflow](https://docs.landing.ai/ade/parse)
- [Parse JSON response](https://docs.landing.ai/ade/ade-json-response)
- [Schema extraction](https://docs.landing.ai/ade/ade-extract)
- [LandingAI description of the OCR-to-ADE evolution](https://landing.ai/blog/ocr-to-agentic-document-extraction-a-look-into-the-evolution-of-document-intelligence)

### RapidAI

- [RapidAI/RapidOCR repository](https://github.com/RapidAI/RapidOCR)
- [RapidOCR raw README](https://raw.githubusercontent.com/RapidAI/RapidOCR/main/README.md)
- [RapidOCR quick start and output contract](https://rapidai.github.io/RapidOCRDocs/main/quickstart/)
- [RapidOCR model list and engine configuration](https://rapidai.github.io/RapidOCRDocs/main/model_list/)

### ONNX

- [ONNX concepts and opsets](https://onnx.ai/onnx/intro/concepts.html)
- [ONNX Runtime overview and validation responsibilities](https://onnxruntime.ai/docs/)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [Graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

## Rerun inputs

```yaml
workflow: research + firecrawl-search + firecrawl-knowledge-base
topic: agentic document extraction, LandingAI ADE, RapidAI RapidOCR, ONNX, ONNX Runtime
goal: reference knowledge base
depth: deep
output: knowledge/research/agentic-document-extraction.md
evidence_policy: primary sources only
```
