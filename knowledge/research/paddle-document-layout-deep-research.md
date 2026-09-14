# Deep Research: PaddlePaddle Document Layout Models and PP-StructureV3

**Research date:** 2026-08-30  
**Depth:** Thorough (10 to 15 minute collection target)  
**Evidence policy:** First-party PaddlePaddle/PaddleOCR/PaddleX sources, official Hugging Face model repositories, and the authors' technical report only.

## Executive summary

Paddle's current documentation describes two materially different capabilities that should not be treated as interchangeable. **PP-DocLayout-L** is a conventional 23-class layout detector: it returns class IDs, labels, confidence scores, and axis-aligned boxes. **PP-DocLayoutV3** (called **RT-DocLayout** in its paper) is a newer 25-class layout-analysis model that adds instance-segmentation contours and learned reading order. Neither model performs OCR or produces Markdown by itself. The larger **PP-StructureV3 pipeline** combines layout analysis with OCR and optional table, seal, formula, chart, orientation, and unwarping components, then assembles structured outputs and Markdown. [PP-DocLayout-L model card](https://huggingface.co/PaddlePaddle/PP-DocLayout-L) · [PP-DocLayoutV3 model card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) · [PP-StructureV3 overview](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html)

For this repository, PP-DocLayoutV3 is the more interesting complement to RapidOCR because it can supply region type, polygonal shape, and an explicit reading order before Luna refinement. It should be introduced as an **optional local layout-evidence adapter**, not as a replacement for RapidOCR and not as an additional ADE pipeline. RapidOCR must remain the source of recognized text and line-level geometry; PP-DocLayoutV3 should enrich those blocks; Luna should arbitrate conflicts and create the auditable refinement layer. This preserves the project's required RapidOCR → GPT-5.6-luna sequence and avoids paying Luna to infer layout signals that a local vision model can produce. [PP-DocLayoutV3 module guide](https://paddlepaddle.github.io/PaddleX/3.4/en/module_usage/tutorials/ocr_modules/layout_analysis.html) · [RT-DocLayout paper](https://arxiv.org/html/2606.23344)

The integration is not risk-free. PaddleX/PaddleOCR documentation, model cards, and release code disagree on important details: PP-DocLayoutV3's documented reading order is zero-based, while PaddleX `release/3.4` post-processing rewrites retained regions to one-based order and assigns `None` to skipped labels; a training configuration declares 11 classes despite the inference model's 25-label contract; PP-DocLayout-L prose uses category names that do not exactly match its shipped `inference.yml`. Therefore, an adapter must validate the installed runtime's actual result object and model metadata rather than hard-code prose descriptions. [PaddleX layout-analysis guide](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md) · [PaddleX post-processing source](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/inference/models/layout_analysis/processors.py) · [PP-DocLayoutV3 training configuration](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml)

## 1. Capability map

| Component | What it does | Native evidence | What it does **not** do |
| --- | --- | --- | --- |
| PP-DocLayout-L | 23-class document-region detection using RT-DETR-L | Class, confidence, `[xmin, ymin, xmax, ymax]` | OCR, polygon masks, learned reading order, Markdown |
| PP-DocLayoutV3 / RT-DocLayout | 25-class detection + instance segmentation + reading-order prediction | Class, confidence, box, polygon points, order | Text recognition, semantic field extraction, validation |
| PP-StructureV3 | Orchestrates layout, OCR, and optional specialist sub-pipelines; restores reading order and exports Markdown/other formats | Layout results, OCR results, parsing list, Markdown and specialist outputs | A single model; results and costs cannot be attributed to layout alone |
| Legacy `ppstructure/layout` | Older PicoDet/PaddleDetection training and inference workflow | Coarse box detections for older label sets | The current PaddleX 3.x API or PP-DocLayoutV3 contract |

Sources: [PP-DocLayout-L card](https://huggingface.co/PaddlePaddle/PP-DocLayout-L) · [PP-DocLayoutV3 module guide](https://paddlepaddle.github.io/PaddleX/3.4/en/module_usage/tutorials/ocr_modules/layout_analysis.html) · [PP-StructureV3 pipeline guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html) · [legacy layout README](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md)

The critical distinction is **layout detection versus document parsing**. Detecting a `table`, `text`, or `paragraph_title` region says where a kind of element probably exists. It does not recognize its text, recover table cells, validate totals, generate source-grounded Markdown, or extract fields. PP-StructureV3 performs many of those later operations because it invokes multiple additional models and post-processors. Consequently, PP-DocLayoutV3 alone is not “ADE” and its metrics must not be presented as end-to-end extraction accuracy. [PP-StructureV3 pipeline guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html) · [layout-detection overview](https://paddlepaddle.github.io/PaddleX/3.4/en/module_usage/tutorials/ocr_modules/layout_detection.html)

## 2. PP-DocLayoutV3 / RT-DocLayout

### Architecture and intended behavior

The authors describe RT-DocLayout as a 33-million-parameter, non-autoregressive model built on RT-DETR. It adds a mask head for pixel-level regions and predicts pairwise precedence relationships from Transformer queries. An antisymmetric relation matrix is converted to a global sequence by a voting-based ranking strategy. Classification, bounding-box, generalized-IoU, mask, Dice, and order losses are trained jointly; the paper gives the order loss a much larger weight because it receives fewer and more diluted optimization signals. PaddleX documentation additionally names PPHGNetV2-L as the backbone. [RT-DocLayout sections 3.1 to 3.3](https://arxiv.org/html/2606.23344#S3) · [PaddleX layout-analysis overview](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md)

The training paper reports 38,000 internally curated document images from academic papers, textbooks, financial reports, slides, newspapers, exams, invoices, receipts, and other domains. It says every element was manually annotated with a boundary, category, and absolute reading order, and that physical-spatial augmentations simulate page warping and camera projection. This is useful evidence of intended robustness, but the training corpus is not publicly described sufficiently to establish representativeness for a particular production domain. [RT-DocLayout §4.1.1](https://arxiv.org/html/2606.23344#S4.SS1.SSS1) · [RT-DocLayout §3.4](https://arxiv.org/html/2606.23344#S3.SS4)

The shipped Paddle inference metadata resizes inputs to **800×800**, identifies the architecture as `DETR`, sets a default draw threshold of `0.5`, and enumerates 25 labels. Its TensorRT dynamic-shape metadata includes batches from 1 through 8. These are artifact facts, not promises that all backends or GPUs will sustain batch 8. [PP-DocLayoutV3 `inference.yml`](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/blob/main/inference.yml)

### Exact 25-label inference taxonomy

The model artifact lists these labels in class-ID order:

```text
0 abstract              9 footer_image       18 reference
1 algorithm            10 footnote           19 reference_content
2 aside_text           11 formula_number     20 seal
3 chart                12 header             21 table
4 content              13 header_image       22 text
5 display_formula      14 image              23 vertical_text
6 doc_title            15 inline_formula     24 vision_footnote
7 figure_title         16 number
8 footer               17 paragraph_title
```

Source of truth for the ordering: [PP-DocLayoutV3 `inference.yml`](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/blob/main/inference.yml). The PaddleX guide names the same 25 concepts in prose. [PaddleX model list](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#ii-supported-model-list)

### Actual output contract

The documented Python entry point is:

```python
from paddlex import create_model

model = create_model(model_name="PP-DocLayoutV3")
for result in model.predict("layout.jpg", batch_size=1):
    result.print()
    result.save_to_img(save_path="./output/")
    result.save_to_json(save_path="./output/res.json")
```

The JSON-shaped result contains `input_path`, `page_index`, and `boxes`. Every box contains `cls_id`, `label`, `score`, axis-aligned `coordinate`, `polygon_points`, and `order`. `polygon_points` are contour vertices derived from masks; they are not OCR line polygons. [PaddleX quick integration](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#iii-quick-integration) · [result class](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/inference/models/layout_analysis/result.py) · [post-processing source](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/inference/models/layout_analysis/processors.py)

Runtime parameters include an explicit `device` such as `gpu:0` or `cpu`, input size, global or per-class thresholds, layout NMS, unclip ratio, box-merging mode, and shape mode. Prediction accepts a path/URL, directory, NumPy array, or a list of those, and has a positive integer `batch_size`. Official weights download automatically, with `PADDLE_PDX_MODEL_SOURCE` controlling the preferred Hugging Face, AI Studio, BOS, or ModelScope source. [PaddleX layout-analysis API](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#iii-quick-integration)

### Performance claims and correct interpretation

Two first-party measurements use different protocols:

- The PaddleX module table reports **23.77 ms on an A100** and **126 MB model storage**, explicitly excluding pre- and post-processing. It does not publish a detection AP value in that table. [PaddleX supported model table](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#ii-supported-model-list)
- The paper reports **132.1 FPS** for the 33M-parameter model on an NVIDIA A100 at **batch size 32**, then reports downstream parsing scores when the layout model is coupled to separate recognizers. With PaddleOCR-VL-1.5-0.9B, it reports 94.50 overall on OmniDocBench v1.5 and 92.05 overall on Real5-OmniDocBench. Those “overall” scores combine downstream text, formula, table, and reading-order dimensions and are not standalone layout precision or OCR character accuracy. [RT-DocLayout §4.1.3 and Tables 1 and 2](https://arxiv.org/html/2606.23344#S4.SS1.SSS3) · [RT-DocLayout sections 4.2.1 to 4.2.2](https://arxiv.org/html/2606.23344#S4.SS2)

The paper's ablation shows its strongest gains on warped and skewed subsets after distortion-aware augmentation, and lower reading-order edit distance after coupled order learning. This supports testing the model on camera-captured or distorted pages, but it does not establish equivalent gains on this application's forms, checkboxes, or private-document mix. [RT-DocLayout §4.3](https://arxiv.org/html/2606.23344#S4.SS3)

## 3. PP-DocLayout-L

PP-DocLayout-L is an RT-DETR-L detector trained on a Paddle-built Chinese/English document dataset. The official table reports **90.4 mAP@0.5**, **123.76 MB**, **33.59 ms GPU model inference**, and **503.01/251.08 ms CPU normal/high-performance model inference**. The evaluation set contains only 500 internally built images, and the timing excludes pre- and post-processing; therefore the figures should be used for relative orientation, not as an application latency or general-domain accuracy SLA. [PaddleOCR layout-detection model table and notes](https://www.paddleocr.ai/main/en/version3.x/module_usage/layout_detection.html#2-supported-model-list)

Its shipped metadata uses a fixed **640×640** resize, a `DETR` architecture, default threshold `0.5`, and 23 labels. Unlike V3, its documented result contains only boxes, not masks/polygons or learned order. [PP-DocLayout-L `inference.yml`](https://huggingface.co/PaddlePaddle/PP-DocLayout-L/blob/main/inference.yml) · [PP-DocLayout-L model card](https://huggingface.co/PaddlePaddle/PP-DocLayout-L)

The artifact's label order is:

```text
0 paragraph_title    8 table          16 seal
1 image              9 table_title    17 chart_title
2 text              10 reference      18 chart
3 number            11 doc_title      19 formula_number
4 abstract          12 footnote       20 header_image
5 content           13 header         21 footer_image
6 figure_title      14 algorithm      22 aside_text
7 formula           15 footer
```

Source of truth: [PP-DocLayout-L `inference.yml`](https://huggingface.co/PaddlePaddle/PP-DocLayout-L/blob/main/inference.yml). The model card's prose instead says “page number,” “table of contents,” “figure caption,” and “table caption,” while its example JSON uses the artifact labels. Consumers should bind class IDs to the installed artifact metadata, not translate from that prose list. [PP-DocLayout-L model card](https://huggingface.co/PaddlePaddle/PP-DocLayout-L)

The PaddleOCR 3.x API uses `LayoutDetection(model_name="PP-DocLayout-L")`; `predict()` returns result objects supporting `print`, `save_to_img`, and `save_to_json`. The stable JSON fields are `input_path`, nullable `page_index`, and `boxes`, with each box holding `cls_id`, `label`, `score`, and `[xmin, ymin, xmax, ymax]`. [PaddleOCR layout-detection quick integration](https://www.paddleocr.ai/main/en/version3.x/module_usage/layout_detection.html#3-quick-integration)

## 4. PP-StructureV3: what the full pipeline adds

PP-StructureV3 is an orchestrated pipeline rather than one model. Its documented modules are layout detection, general OCR, optional document preprocessing, optional table recognition, optional seal recognition, optional formula recognition, and chart understanding. It reconstructs reading order and can save JSON, Markdown, Word, HTML, XLSX, and visualizations depending on result content. [PP-StructureV3 algorithm overview](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html) · [pipeline usage guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html)

The `release/3.4` default pipeline configuration uses `PP-DocLayout_plus-L`, not PP-DocLayoutV3 or PP-DocLayout-L. It also enables table and formula recognition, disables the document preprocessor, seal recognition, and chart recognition, and ignores page numbers, headers, footers, footnotes, and aside text in Markdown. Changing only the layout model therefore does not reproduce a documented default pipeline, and copying PP-StructureV3 output policy may silently drop evidence this repository intentionally retains. [PP-StructureV3 `release/3.4` configuration](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/configs/pipelines/PP-StructureV3.yaml)

The structured result exposes `parsing_res_list` in parsed reading order. Each entry includes a `block_bbox`, `block_label`, and nullable `block_order`, while nested OCR results include detected polygons, recognized text, and recognition scores. The pipeline also exposes a `markdown` attribute and `save_to_markdown()`. This is a richer contract than the layout module result because it includes outputs from the other sub-pipelines. [PP-StructureV3 result contract](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html#3-description-of-pipeline-parameters)

Paddle reports PP-StructureV3 document-parsing edit scores on OmniDocBench, including separate English/Chinese overall, text, formula, table, and reading-order metrics. These benchmark values describe the whole configured pipeline and cannot be credited to PP-DocLayout-L/V3 alone or directly compared with this repository's RapidOCR+Luna workflow without running the same dataset, rendering, and metric implementation. [PP-StructureV3 key metrics](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html#key-metrics)

## 5. Installation, GPU execution, and deployment

### Current PaddleX 3.4 path

PaddleX 3.4 documentation supports Python 3.8 to 3.13 and recommends installing PaddlePaddle first, followed by the PaddleX wheel. For inference-only integration it documents `pip install paddlex`, `paddlex[base]`, or the narrower `paddlex[ocr]`; source/plugin installation is intended for retraining or framework modification. [PaddleX installation guide](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/installation/installation.en.md)

GPU use requires a PaddlePaddle GPU build that matches a supported CUDA line, not merely an installed NVIDIA driver. The release/3.4 guide gives CUDA 11.8 and 12.6 wheel indexes and recommends verifying with `import paddle; print(paddle.__version__)`. For Paddle-TensorRT it documents TensorRT 8.6.1.6 for CUDA 11.8; other combinations must follow the compatibility documentation. Windows 50-series GPU support is called out separately with development wheels and known limitations. [PaddlePaddle installation guide for PaddleX 3.4](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/installation/paddlepaddle_install.en.md)

PaddleX's model constructor accepts `device="gpu:0"`; the current PaddleOCR pipeline says it prefers local GPU 0 when available and otherwise uses CPU if no device is supplied. Explicit device selection and a startup smoke inference are safer than assuming the chosen wheel exposes CUDA. [PaddleX layout-analysis API](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#iii-quick-integration) · [PP-StructureV3 parameters](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html#3-description-of-pipeline-parameters)

This repository currently requires Python `>=3.13` and uses NumPy 2.x, OpenCV 5.x, ONNX Runtime GPU, and a Git-pinned RapidOCR, while neither PaddlePaddle nor PaddleX is installed. PaddleX's stated Python support makes an in-process experiment plausible, but dependency compatibility is not established by that statement. A disposable `uv` environment should resolve and smoke-test the exact Paddle wheel before adding it to the main application environment. Source inspected: [`pyproject.toml`](../../pyproject.toml); external compatibility source: [PaddleX installation guide](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/installation/installation.en.md).

### Export and serving

PaddleX provides training, evaluation, prediction, and export modes through its module configuration; the PP-DocLayoutV3 training guide uses a COCO instance-segmentation dataset augmented with a `read_order` field. The same guide documents export from trained weights and use of a local exported model directory in `create_model`. [PP-DocLayoutV3 custom development](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#iv-custom-development)

For ordinary layout detectors, current PaddleOCR additionally documents `paddle_static`, `paddle_dynamic`, `transformers`, and `onnxruntime` engines, plus conversion from Paddle weights to safetensors or ONNX where supported. That engine matrix belongs to the **layout-detection** module and should not be assumed for PP-DocLayoutV3's separate layout-analysis module without a runtime test. [PaddleOCR layout-detection inference engines](https://www.paddleocr.ai/main/en/version3.x/module_usage/layout_detection.html#5-inference-engine)

The project should initially use local in-process inference. PaddleX/PaddleOCR also documents serving and multi-hardware deployment, but those expand operational scope and do not benefit this local-machine Streamlit application unless process isolation becomes necessary. [PP-StructureV3 overview](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html) · [PaddleX installation guide](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/installation/installation.en.md)

## 6. Legacy `ppstructure/layout` is not the current install guide

The requested GitHub README was last materially updated in 2022 and documents a PaddlePaddle `>=2.3` plus cloned PaddleDetection workflow around PicoDet. It covers older English PubLayNet-style categories, Chinese CDLA categories, training, distillation, export, and standalone inference. [legacy README](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md) · [file history](https://github.com/PaddlePaddle/PaddleOCR/commits/main/ppstructure/layout/README.md)

Its installation anchors ([Install](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md#3-install), [Install PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md#31-install-paddlepaddle), and [Install PaddleDetection](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md#32-install-paddledetection)) remain useful only for reproducing that legacy training stack. They should not be combined with modern `paddleocr LayoutDetection`, `paddlex.create_model`, or PP-DocLayoutV3 examples. The modern inference path is the PaddleOCR/PaddleX 3.x wheel API. [current PaddleX layout detection guide](https://paddlepaddle.github.io/PaddleX/3.4/en/module_usage/tutorials/ocr_modules/layout_detection.html)

## 7. Version drift and contract hazards

| Conflict or ambiguity | Evidence | Integration rule |
| --- | --- | --- |
| PP-DocLayoutV3 order base | Guide says order starts at 0; `release/3.4` post-processor initializes at 1 and assigns `None` to skipped labels. | Treat order as an ordinal, normalize explicitly, record raw value and runtime version. |
| Skipped order labels | Source defaults skip titles attached to figures, images, charts, tables, headers/footers, footnotes, seals, and formula numbers. | Do not assume every detected region has an order; retain unordered regions. |
| V3 class count | Inference artifact has 25 labels; the `release/3.4` training YAML says `num_classes: 11`. | Read inference taxonomy from model metadata; require an explicit custom taxonomy for retraining. |
| PP-DocLayout-L label names | Model-card prose and artifact labels differ. | Persist raw `cls_id` + raw artifact `label`; map to canonical labels separately. |
| V3 speed | 23.77 ms module-table latency and 132.1 FPS paper throughput use different conditions. | Benchmark application wall time on the target 8 GB GPU, including preprocessing and post-processing. |
| “Accuracy” | L publishes mAP@0.5 on 500 internal images; V3 paper publishes downstream compound parsing scores. | Never compare or market these as one accuracy measure. |
| Documentation generation | PaddleOCR `main`, PaddleX `release/3.4`, model cards, and the 2022 legacy guide describe different generations. | Pin package/model versions and link to matching source revisions in provenance. |

Sources: [V3 guide](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md) · [V3 processor](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/inference/models/layout_analysis/processors.py) · [V3 config](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml) · [L artifact](https://huggingface.co/PaddlePaddle/PP-DocLayout-L/blob/main/inference.yml) · [RT-DocLayout paper](https://arxiv.org/html/2606.23344)

## 8. Recommended role in the RapidOCR + Luna pipeline

### Recommended data flow

```text
selected page image
  ├─ RapidOCR ──> immutable text lines, OCR polygons, recognition confidence
  └─ PP-DocLayoutV3 (optional) ──> region class, box/polygon, layout confidence, order
          │
          └─ deterministic spatial join
               ├─ OCR lines assigned to containing/intersecting layout regions
               ├─ unmatched OCR lines retained
               ├─ unmatched layout regions retained as non-text evidence
               └─ conflicts/warnings recorded
                        │
                        v
              canonical local Parse evidence
                        │
                        v
          GPT-5.6-luna required refinement (medium effort)
```

This design is a synthesis based on the model contracts above and the repository's inspected Parse implementation (`ocr.py`, `parse.py`, `pipeline.py`, and `openai_refiner.py`). It does **not** adopt PP-StructureV3 as a parallel parser, because doing so would duplicate OCR and Markdown reconstruction, obscure evidence provenance, and violate the canonical pipeline boundary.

### Canonical evidence additions

Add layout evidence without mutating RapidOCR blocks:

- Page provenance: model name, model artifact hash/version, Paddle/PaddleX versions, device, input size, thresholds, elapsed time, and warnings.
- Region: stable local ID, raw class ID/label, score, axis-aligned box, polygon points, raw order, normalized order, and whether order was absent/skipped.
- Spatial links: OCR source IDs with overlap/containment scores and an explicit assignment method.
- Conflicts: overlapping incompatible regions, low-confidence labels, non-monotonic/duplicate orders, OCR outside all regions, and layout regions without OCR.

These fields derive directly from the published result contract and preserve the project's immutable-evidence rule. [V3 output contract](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#iii-quick-integration) · [V3 processor](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/inference/models/layout_analysis/processors.py)

### Expected benefits

1. **Reading order:** use V3 order as a strong local candidate instead of relying only on geometric XY sorting; send conflicts to Luna. The paper's ablations provide evidence that jointly learned order helps on complex and distorted layouts, although project-specific validation remains necessary. [RT-DocLayout §4.3](https://arxiv.org/html/2606.23344#S4.SS3)
2. **Context selection:** region labels can keep headers/footers separate, identify tables/forms/images, and support smaller evidence-bearing Luna prompts. This is an architectural inference; no first-party source measures Luna token savings.
3. **Better crops:** V3 polygons can define tighter uncertain-region crops on curved/skewed pages than axis-aligned OCR unions. The model is specifically designed for mask-level distorted-page localization. [RT-DocLayout sections 3.1 to 3.4](https://arxiv.org/html/2606.23344#S3)
4. **Markdown structure:** deterministic assembly can treat `doc_title`, `paragraph_title`, tables, formulas, lists/content, and footnotes differently before Luna validation. Labels remain hypotheses, never verified semantics.

### What not to do

- Do not replace RapidOCR with PP-DocLayoutV3; V3 has no text-recognition output. [V3 output contract](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#iii-quick-integration)
- Do not run PP-StructureV3 and then call its Markdown “RapidOCR evidence”; that pipeline uses its own OCR and specialist models. [PP-StructureV3 pipeline configuration](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/configs/pipelines/PP-StructureV3.yaml)
- Do not suppress OCR lines because a layout detector missed them; detector recall is not guaranteed.
- Do not map layout confidence to OCR confidence; they measure different predictions.
- Do not accept a table, checkbox, or field value merely because its region label is confident; content still requires OCR/vision agreement and validation.
- Do not expose V3 as mandatory until the Paddle runtime, GPU memory, latency, and quality are measured on this Windows 11 machine.

## 9. Evaluation plan before integration

1. **Environment gate:** in a disposable `uv` environment, install a PaddlePaddle GPU wheel matching the actual driver/CUDA support, then PaddleX 3.4. Verify `paddle.device.is_compiled_with_cuda()`, the selected device, model download, and one inference. Do not infer GPU availability from `nvidia-smi` alone. [PaddlePaddle installation guide](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/installation/paddlepaddle_install.en.md)
2. **Contract fixture:** serialize results for a normal image, scanned PDF page, multi-column page, table, rotated page, warped photo, and empty page. Assert all raw fields and record the actual order base/null behavior.
3. **Join accuracy:** manually label OCR-line-to-region membership and reading order for a small representative corpus. Measure assignment precision/recall and order edit distance before allowing layout evidence to alter Markdown.
4. **End-to-end ablation:** compare current RapidOCR+Luna against RapidOCR+V3+Luna with identical Luna prompts and page selection. Measure Markdown edit/structure metrics, field accuracy, checkbox precision/recall, review rate, Luna tokens/calls, local latency, peak VRAM, and failure rate.
5. **Threshold calibration:** calibrate per-class thresholds on project data. The API supports per-class thresholds; the default `0.5` is not evidence of optimal calibration. [PaddleX V3 API](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#iii-quick-integration)
6. **Acceptance rule:** enable the adapter only if it measurably improves reading order/structure or reduces Luna cost without degrading text, fields, checkbox decisions, or review safety. Retain a clear setup failure if the feature is configured as required but unavailable.

## Contrarian views and risks

- **The additional model may increase, not reduce, latency.** Paddle's quoted V3 time excludes preprocessing/post-processing and uses an A100; an 8 GB consumer GPU plus page rendering, contour extraction, and host transfers can behave very differently. [PaddleX V3 model table](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md#ii-supported-model-list)
- **V3 may be redundant on simple documents.** The current pipeline already has OCR geometry and Luna sees page images. The local model earns its complexity only where order/polygon evidence changes quality or reduces cloud context.
- **Taxonomy mismatch creates false structure.** Neither 23 nor 25 labels contains a dedicated checkbox/control category. Specialized checkbox logic must remain separate.
- **Model benchmarks are author-reported.** The V3 training data is private, and its strongest headline scores depend on downstream recognizers and defined benchmark protocols. Independent application-domain validation is essential. [RT-DocLayout §4](https://arxiv.org/html/2606.23344#S4)
- **Dependency collision is plausible.** The existing app uses a new Python/NumPy/OpenCV/ONNX stack. PaddleX's broad Python support does not prove that a single environment resolves cleanly or shares CUDA safely with ONNX Runtime.
- **Two geometry systems need careful normalization.** RapidOCR text polygons and V3 region polygons can be based on different resizing/post-processing paths. Persist original image dimensions and normalize only in an explicit adapter.

## Open questions

1. Does the released PP-DocLayoutV3 wheel return one-based order with null skipped labels on the exact PaddleX/Paddle versions chosen, despite the guide saying zero-based?
2. What does the V3 `num_classes: 11` training YAML represent when official inference metadata contains 25 labels?
3. Are PP-DocLayoutV3's current Paddle/ONNX/transformers engine options equivalent on Windows, or is Paddle inference the only supported production path?
4. What peak VRAM and wall-clock latency does V3 exhibit at batch sizes 1 to 8 on the user's 8 GB GPU alongside RapidOCR's ONNX Runtime session?
5. Do polygon/order gains remain after Luna's visual review, and do they reduce tokens enough to offset local latency?
6. Which region classes and thresholds are calibrated for the application's actual invoices, forms, claims, and scanned PDFs?

## Knowledge-base source index

### User-specified sources

1. [PP-DocLayoutV3 Hugging Face model repository](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3): model identity, intended non-planar-document behavior, weights, paper link.
2. [PP-StructureV3 algorithm page](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html): pipeline capabilities and end-to-end benchmark tables.
3. [PaddleX 3.4 layout-detection tutorial](https://paddlepaddle.github.io/PaddleX/3.4/en/module_usage/tutorials/ocr_modules/layout_detection.html): modern model API, results, thresholds, models, training.
4. [Legacy PaddleOCR layout README](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md): older PicoDet/PaddleDetection workflow.
5. [Legacy installation section](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md#3-install): legacy install entry point.
6. [Legacy PaddlePaddle install subsection](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md#31-install-paddlepaddle): old Paddle `>=2.3` instructions.
7. [Legacy PaddleDetection install subsection](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md#32-install-paddledetection): cloned PaddleDetection training dependency.
8. [PP-DocLayout-L Hugging Face model repository](https://huggingface.co/PaddlePaddle/PP-DocLayout-L): model description, example API/output, pipeline example.

### Additional first-party sources

9. [PP-DocLayoutV3 inference metadata](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/blob/main/inference.yml): preprocessing, labels, architecture, backend shapes.
10. [PP-DocLayout-L inference metadata](https://huggingface.co/PaddlePaddle/PP-DocLayout-L/blob/main/inference.yml): preprocessing, labels, architecture, backend shapes.
11. [RT-DocLayout technical report](https://arxiv.org/html/2606.23344): architecture, training, benchmarks, ablations, caveats.
12. [PaddleX 3.4 layout-analysis tutorial](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/module_usage/tutorials/ocr_modules/layout_analysis.en.md): PP-DocLayoutV3 API, output, model table, custom development.
13. [PaddleX V3 result implementation](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/inference/models/layout_analysis/result.py): JSON/visualization behavior.
14. [PaddleX V3 post-processing](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/inference/models/layout_analysis/processors.py): polygon extraction, ordering, skipped labels.
15. [PaddleX V3 predictor](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/inference/models/layout_analysis/predictor.py): result construction and configuration.
16. [PP-DocLayoutV3 module config](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml): train/evaluate/export/predict defaults.
17. [PP-StructureV3 pipeline config](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/paddlex/configs/pipelines/PP-StructureV3.yaml): actual submodels and default feature flags.
18. [PaddleOCR PP-StructureV3 usage guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html): APIs, parameters, and structured result contract.
19. [PaddleX 3.4 installation guide](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/installation/installation.en.md): Python versions and package modes.
20. [PaddlePaddle installation guide for PaddleX 3.4](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/docs/installation/paddlepaddle_install.en.md): CUDA wheels, drivers, Windows notes, TensorRT.

## Collection note

The requested Firecrawl workflow was attempted first with `FIRECRAWL_API_KEY` available to the process, but Firecrawl returned **“Insufficient credits to perform this request.”** Collection therefore continued against the same official pages through raw GitHub/Hugging Face endpoints and the authors' arXiv HTML. No secondary sources were substituted. The report is organized as a compact reference knowledge base rather than a scraped documentation mirror so it remains one auditable Markdown artifact.

## Rerun inputs

```yaml
workflow:
  - research
  - firecrawl-deep-research (attempted; blocked by account credits)
  - firecrawl-scrape (attempted; blocked by account credits)
  - firecrawl-knowledge-base (synthesized from first-party fallback collection)
topic: Paddle document layout analysis models and PP-StructureV3
depth: thorough
output: markdown reference knowledge base
output_file: knowledge/research/paddle-document-layout-deep-research.md
retrieved: 2026-08-30
```
