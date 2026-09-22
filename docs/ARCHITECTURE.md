# Architecture

## System purpose

Agentic document extractor is a local-machine system for extracting structured, source-grounded content from PDF and image files. The extraction pipeline requires three sequential stages for every successful execution:
1. RapidOCR for optical character recognition, text detection, and polygon extraction.
2. PP-DocLayoutV3 (executing in an isolated worker process) for document region detection, reading order normalization, and table structure parsing.
3. OpenAI `gpt-6-sol` (configured with `reasoning_effort="medium"`) for semantic refinement, error correction, and structured field extraction.

The Streamlit user interface and the local FastAPI service share this canonical pipeline. Neither entry point implements document extraction independently.

## Request and data flow

The following diagram illustrates the request and data flow across real modules during document processing.

```mermaid
flowchart TD
    subgraph Clients["Entry points"]
        direction TB
        StreamlitApp["Streamlit UI (streamlit_app.py / app.py)"]
        FastAPIApp["FastAPI Service (src/agentic_extractor/api.py)"]
    end

    subgraph WorkflowEngine["Workflow and pipeline orchestration"]
        direction TB
        AgentWorkflow["run_agent_workflow (workflow.py)"]
        HybridPipeline["process_hybrid_document (pipeline.py)"]
    end

    subgraph Ingestion["Document ingestion and preprocessing"]
        direction TB
        IngestMod["Ingest and decode (ingest.py)"]
        QualityMod["Quality diagnostics (quality.py)"]
    end

    subgraph LocalEngines["Local inference engines"]
        direction TB
        OCRMod["RapidOCR engine (ocr.py)"]
        LayoutMod["PP-DocLayoutV3 adapter (layout.py)"]
        WorkerSubprocess["Worker subprocess (tools/pp_doclayout/worker.py)"]
        TableMod["Table structure and grounding (table_structure.py)"]
        CheckboxMod["OpenCV checkbox analysis (checkbox_vision.py)"]
        RoutingMod["Visual crop routing (visual_routing.py)"]
    end

    subgraph CloudEngine["Semantic refinement"]
        direction TB
        RefinerMod["OpenAIRefiner (openai_refiner.py)"]
        PromptsMod["Packaged Markdown prompts (prompts/*.md)"]
        OpenAIAPI["OpenAI API (gpt-6-sol)"]
    end

    subgraph ValidationAndAudit["Validation and adjudication"]
        direction TB
        CapabilitiesMod["Capabilities validation (capabilities.py)"]
        AdjudicationMod["Workflow state adjudication (workflow.py)"]
    end

    subgraph Outputs["Artifact generation"]
        direction TB
        ArtifactsMod["LocalArtifacts (artifacts.py)"]
        LandingContractMod["LandingAI parse-result.json (landing_contract.py)"]
        LayoutHTMLMod["Coordinate HTML generator (layout_html.py)"]
    end

    %% Flow connections
    StreamlitApp -->|DocumentRequest| AgentWorkflow
    FastAPIApp -->|DocumentRequest| AgentWorkflow
    FastAPIApp -->|DocumentRequest (post-parse)| AgentWorkflow

    AgentWorkflow --> HybridPipeline
    HybridPipeline --> IngestMod
    IngestMod --> QualityMod

    HybridPipeline --> OCRMod
    OCRMod -->|Raw OCR blocks and scores| LayoutMod

    LayoutMod <-->|JSON IPC| WorkerSubprocess
    LayoutMod --> TableMod
    HybridPipeline --> CheckboxMod
    HybridPipeline --> RoutingMod

    HybridPipeline --> RefinerMod
    RefinerMod --> PromptsMod
    RefinerMod <-->|Responses API| OpenAIAPI

    AgentWorkflow --> CapabilitiesMod
    CapabilitiesMod --> AdjudicationMod

    AgentWorkflow --> ArtifactsMod
    ArtifactsMod --> LandingContractMod
    ArtifactsMod --> LayoutHTMLMod
```

### Execution sequence

1. Preflight configuration check: `workflow.py` validates that `OPENAI_API_KEY` is present and functional via `openai_refiner.py`. If configuration is invalid, execution halts before local inference begins.
2. Ingestion and validation: `ingest.py` inspects file signatures, enforces the 50 MiB, 200-page, and 25-megapixel limits, and renders selected pages to RGB images.
3. Local OCR: `ocr.py` runs RapidOCR over each selected page. CPU mode distributes pages across bounded worker processes (capped by `ADE_OCR_MAX_WORKERS`); GPU mode executes serially. Raw text, recognition confidence scores, and bounding polygons are retained as immutable `Block` structures.
4. Layout and reading order detection: `layout.py` communicates with the isolated `tools/pp_doclayout` worker process via standard I/O JSON IPC. The worker runs PP-DocLayoutV3 to classify document regions and determine reading order.
5. Table analysis: `layout.py` crops detected table regions and invokes table classification (wired vs. wireless) and SLAN structure models in the worker. `table_structure.py` maps predicted table cells to RapidOCR blocks and produces sanitized HTML table representations.
6. Visual routing and checkbox analysis: `checkbox_vision.py` detects square checkboxes and pixel states using OpenCV. `visual_routing.py` identifies high-uncertainty regions (low OCR scores, complex tables, candidate checkboxes) to be passed as high-detail image crops to OpenAI.
7. Semantic refinement: `openai_refiner.py` sends grounded text context, every selected page at high image detail, and routed crops to `gpt-6-sol`. It proposes OCR corrections, semantic groups, visual additions, and document links. Table review and eligible checkbox crop verification use the same model.
8. Downstream workflows: If requested, Classify, Section, Split, and Extract workflows execute using the refined Markdown and grounding index without re-running OCR.
9. Deterministic validation and adjudication: `capabilities.py` and `workflow.py` validate field types, regex patterns, date formats, and business rules, transitioning the job to `ACCEPTED`, `REVIEW_REQUIRED`, or `FAILED`.
10. Artifact creation: Core artifacts (Markdown, `parse-result.json`, `manifest.json`) are finalized immediately. Heavy visual artifacts (`annotated.pdf`, `document.html`, `bundle.zip`) are generated lazily upon download or view.

Steps 7–9 are coordinated by `run_workflow_from_parse`. A fresh Parse enters a loop of at
most two repair rounds. Each round can independently inspect up to eight pending visual
objects, rebuild canonical Markdown, run requested downstream workflows, and attempt scoped
repair of eligible review items. Unchanged review findings stop further rounds. A reused
Parse skips new visual-object crop inspections; downstream workflows do not rerun OCR.
The bound is on repair rounds, not all OpenAI calls: initial page batches, table review,
checkbox verification, and requested downstream operations have their own calls.

`rich_document.py` renders accepted visual additions without inserting raw OCR blocks.
`app_pages/visual_review.py` records human decisions and rebuilds exports without engine calls.

## Main types and state

The primary domain models and application states are defined across the following modules:

| Type / state container | Defined location | Purpose and lifecycle |
|---|---|---|
| `DocumentRequest` | `src/agentic_extractor/models.py` | Command object containing file bytes, file name, selected pages, processing mode (`Balanced` vs. `High Accuracy`), active capabilities, and schema inputs. |
| `Block` | `src/agentic_extractor/models.py` | Immutable representation of an individual OCR text segment, including bounding polygon coordinates, text content, line index, and confidence score. |
| `PageParse` | `src/agentic_extractor/parse.py` | Per-page parsing container holding raw `Block` elements, layout regions, reading order sequences, and rendered page image data. |
| `VisualObject` / `DocumentLink` | `src/agentic_extractor/rich_document.py` | Visual content and relationship proposals; crop decisions and human audits are kept separate from raw OCR. |
| `RequestBudget` | `src/agentic_extractor/budget.py` | Optional token-count-based cost reservations used by the synthetic live runner; not a default UI/API spending limit. |
| `LocalParseResult` | `src/agentic_extractor/ocr.py` | Aggregated result of local ingestion and OCR across all selected pages, containing immutable raw blocks, layout regions, quality metrics, warnings, and engine provenance. |
| `AgentWorkflowResult` | `src/agentic_extractor/workflow.py` | End-to-end workflow container tracking state transitions, audit event logs, extracted fields, field corrections, and human review items. |
| `LocalArtifacts` | `src/agentic_extractor/artifacts.py` | Artifact management container providing cached access to generated Markdown, LandingAI Parse JSON, annotated PDF bytes, HTML viewer bytes, and ZIP archive bundles. |
| `ProcessedMarkdownDocument` | `src/agentic_extractor/document_chat.py` | Restricted session-only document representation used by the chat interface; exposes generated Markdown and page numbers without retaining raw file bytes or OCR tokens. |
| `st.session_state` | Streamlit runtime (`ui_state.py`, `app.py`) | In-memory UI state storing active document IDs, processed document registries, chat histories, and navigation context. Discarded when browser session ends or Reset is clicked. |
| Process caches | `src/agentic_extractor/cache.py` | Four thread-safe in-memory LRU caches (rendered pages, OCR results, layout results, table structures) keyed by exact pixel hashes and model signatures (256 entries / 128 MiB each). |
| `_Job` | `src/agentic_extractor/api.py` | In-memory API job dictionary tracking state (`PROCESSING`, `ACCEPTED`, `REVIEW_REQUIRED`, `FAILED`), raw requests, results, and creation timestamps with 3600-second TTL. |

## External systems

The application interacts with the following external systems and boundaries:

### OpenAI Responses API

- Endpoint: Default `https://api.openai.com/v1` (overridable via `OPENAI_BASE_URL`).
- Model: `gpt-6-sol`.
- Parameters: `reasoning_effort="medium"`.
- Usage: Invoked for Parse page refinement, independent checkbox and visual-object crop verification, dedicated table structure review, downstream agentic capabilities (Classify, Section, Split, Extract), repair, and grounded document chat.
- Request limits: `max_output_tokens=16384`, `store=False`, no external tools, and no automatic SDK retries. Refused, incomplete, or unparseable responses are not accepted; available usage is retained before structured parsing.
- Contract: Communication uses JSON structured outputs. Prompts are loaded from versioned Markdown templates under `src/agentic_extractor/prompts/` and tracked by SHA-256 digests in audit manifests.

### PP-DocLayoutV3 worker subprocess

- Path: `tools/pp_doclayout/worker.py` executed with `tools/pp_doclayout/.venv/Scripts/python.exe`.
- Protocol: Standard input/output pipe exchanging line-delimited JSON messages (`health`, `predict_layout`, `predict_tables`, `shutdown`).
- Isolation rationale: PaddleX 3.4 and PaddlePaddle GPU 3.2 require specific OpenCV package configurations that conflict with the root application's dependencies. The subprocess maintains a dedicated worker lifecycle.
- Fail-safe boundary: All output received across the subprocess pipe is treated as untrusted and validated against strict schemas in `src/agentic_extractor/layout.py`. Any schema violation raises `LayoutContractError`.

### Hardware accelerator runtimes

- NVIDIA CUDA / cuDNN: ONNX Runtime GPU (`onnxruntime-gpu>=1.29.0`) queries CUDA execution providers for RapidOCR inference.
- Paddle CUDA runtime: `paddlepaddle-gpu==3.2.0` in the worker environment uses CUDA 12 packages (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`).

## Document chat architecture

The document chat interface (`app_pages/chat.py` and `src/agentic_extractor/document_chat.py`) operates as an isolated, session-only read model:

```mermaid
flowchart LR
    UserQuery["User question"] --> ScopeFilter["Document scope selection (max 12)"]
    ScopeFilter --> Chunking["Markdown heading & page chunking"]
    Chunking --> LexicalRanker["Lexical keyword retrieval"]
    LexicalRanker --> ContextWindow["Context assembly (max 40k chars, top 12 chunks)"]
    ContextWindow --> SolChat["gpt-6-sol (packaged chat prompt)"]
    SolChat --> CitationValidator["Citation verification against chunk IDs"]
    CitationValidator --> AnswerDisplay["Grounded answer and citations"]
```

1. Source isolation: Chat requests consume only generated Parse Markdown. Uploaded binary files, page images, and raw OCR objects never enter the chat context.
2. Retrieval: Documents are chunked by page boundaries and Markdown headers. If total text is under 40,000 characters, all text is supplied. Otherwise, a local lexical retrieval algorithm scores chunks and selects the top 12 excerpts.
3. Citation validation: The model must return excerpt IDs matching the supplied chunks. If citation verification fails or the model returns unsubstantiated content, the system fails closed with an explicit insufficient-evidence message.
