# RapidOCR CUDA and page-orientation findings

Date: 2026-08-30  
Scope: quick investigation of the current Windows machine and installed project environment

## Executive summary

The CUDA message is a false negative from the application's preflight test, not evidence that this machine lacks a usable GPU. The environment has an NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB), ONNX Runtime GPU 1.29.0 reports `CUDAExecutionProvider`, and an initialized RapidOCR detection session reports `CUDAExecutionProvider` followed by `CPUExecutionProvider`. The application's check uses `onnxruntime.get_ep_devices()`, which returns only a CPU device in this environment even though the conventional CUDA execution provider works.

The orientation message is accurate but describes a RapidOCR output-contract limitation. `RapidOCROutput` contains image, boxes, text, recognition scores, word results, and timings; it contains no page-orientation field. RapidOCR's enabled classifier classifies detected text crops as only `0` or `180`, rotates those crops before recognition, and does not copy classifier labels into the combined result. It is therefore not a general page-orientation detector and does not represent 90°/270° page rotation.

## Finding 1: CUDA works on this machine

Observed in the project's `uv` environment:

| Check | Result |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB |
| NVIDIA driver | 616.56 |
| System CUDA toolkit | 13.3 |
| System cuDNN | `cudnn64_9.dll` present |
| Python | 3.13.15 |
| ONNX Runtime | `onnxruntime-gpu` 1.29.0 |
| `ort.get_available_providers()` | TensorRT, CUDA, CPU |
| `ort.get_device()` | GPU |
| Initialized RapidOCR detector providers | CUDA, CPU |
| `ort.get_ep_devices()` | CPU device only |

ONNX Runtime's official compatibility table says 1.29.x is built for CUDA 13.0 and cuDNN 9.x. It also states CUDA 13.0 builds are compatible across CUDA 13.x under NVIDIA minor-version compatibility. This matches CUDA 13.3 and cuDNN 9 on this machine. Starting with ONNX Runtime 1.27, PyPI GPU packages use CUDA 13.0 by default. [ONNX Runtime CUDA Execution Provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

The official API documentation distinguishes the list of compiled/available providers (`get_available_providers`) from providers actually registered on an inference session (`InferenceSession.get_providers`). The latter is the decisive runtime check for a specific model session. [ONNX Runtime Python API summary](https://onnxruntime.ai/docs/api/python/api_summary.html)

### Why the two CUDA messages appear

1. The application currently treats the absence of a CUDA entry from `get_ep_devices()` as proof that no usable CUDA device exists. On this environment, that enumeration contains only a CPU `OrtEpDevice`, while the ordinary CUDA EP is nevertheless available and works in a RapidOCR inference session. That produces the application's false warning: `No usable CUDA execution-provider device`.
2. ONNX Runtime emits `No registered plugin EP device found for 'CUDAExecutionProvider'` while creating the session. Its source emits that message during plugin-EP device lookup. The subsequent RapidOCR session still registers `CUDAExecutionProvider`, so this particular log line is not proof of CUDA fallback. [ONNX Runtime provider-factory source](https://github.com/microsoft/onnxruntime/blob/27e64f961d36fa34b8393fd1743fbc5cf579af16/onnxruntime/python/onnxruntime_pybind_state.cc)

### Exact corrective approach

- Use `CUDAExecutionProvider in ort.get_available_providers()` and `ort.get_device() == "GPU"` only as a cheap eligibility check.
- Request CUDA when constructing RapidOCR.
- After RapidOCR lazily creates its model sessions, inspect each created ONNX session's `get_providers()`. Record CUDA only when `CUDAExecutionProvider` is present there; otherwise report CPU fallback with the initialization exception or provider list.
- Do not use `get_ep_devices()` as a hard gate for the conventional CUDA EP on this environment.
- Keep CPU as an explicit fallback because provider availability does not guarantee every session will initialize successfully.

No CUDA/cuDNN reinstall is presently justified. If session creation later fails with a missing-DLL error, ONNX Runtime 1.21+ officially supports `onnxruntime.preload_dlls()` before session creation and `onnxruntime.print_debug_info()` for diagnosis. The same documentation also permits importing a compatible CUDA-enabled PyTorch first to preload its DLLs. [ONNX Runtime preload DLLs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls)

## Finding 2: combined RapidOCR output has no page orientation

RapidOCR's official `RapidOCROutput` dataclass defines `img`, `boxes`, `txts`, `scores`, `word_results`, `elapse_list`, `elapse`, and visualization state. It defines no orientation or classifier-label field. [RapidOCR output source](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/utils/output.py)

RapidOCR runs detection, then classification and crop rotation, then recognition. In `build_final_output`, the classifier contributes only elapsed time to `RapidOCROutput`; its `cls_res` labels are not exposed in the combined result. [RapidOCR pipeline source](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/main.py)

The packaged classifier configuration has labels `0` and `180`, and the classifier rotates an individual text crop by 180° when that label exceeds the threshold. It does not classify a page as 0°/90°/180°/270°. [RapidOCR classifier source](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/ch_ppocr_cls/main.py)

### Exact corrective approach

- Treat page orientation as `unknown` when it is not established by source metadata or preprocessing. Do not emit the current limitation as a per-page warning; record it as provenance/capability metadata instead.
- Continue applying EXIF orientation for uploaded images. PDF-rendered pages normally provide no useful EXIF orientation.
- If automatic page rotation is required, add a separate whole-page orientation step before final OCR. It must explicitly handle all four rotations. A bounded implementation can score low-resolution 0°/90°/180°/270° candidates using OCR text count, mean recognition confidence, and horizontal-layout evidence, then run final full-resolution OCR only on the winner. Preserve the selected angle, candidate scores, and method in provenance.
- Keep a user rotation override for ambiguous pages. Do not infer a page angle from RapidOCR's crop-level 0°/180° classifier.

The four-candidate method can increase local latency. It should run only when preflight evidence indicates rotation or the first OCR pass produces weak evidence; always trying four full-resolution OCR passes would unnecessarily increase cost and latency.

## Risks and limitations

- A provider name on a session proves registration, not that every operation ran on GPU; ONNX Runtime may assign unsupported nodes to CPU. Profiling is needed for operator-level proof.
- The ONNX Runtime plugin-device warning is noisy but could change meaning in future versions. Continue checking actual session providers after upgrades.
- OCR-confidence-based orientation can choose incorrectly on sparse pages, diagrams, handwriting, or pages dominated by vertical text. Such cases should remain reviewable or use a dedicated orientation model.
- RapidOCR is currently a development build (`3.9.3.dev8`); its output contract may change. Pin and regression-test any private/internal API use.

## Sources

1. [ONNX Runtime CUDA Execution Provider requirements and compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) — official CUDA/cuDNN compatibility table.
2. [ONNX Runtime preload DLLs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls) — official DLL-loading and diagnostic options.
3. [ONNX Runtime Python API summary](https://onnxruntime.ai/docs/api/python/api_summary.html) — official provider/session API.
4. [ONNX Runtime provider-factory source](https://github.com/microsoft/onnxruntime/blob/27e64f961d36fa34b8393fd1743fbc5cf579af16/onnxruntime/python/onnxruntime_pybind_state.cc) — source of the plugin-device warning.
5. [RapidOCR combined-output source](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/utils/output.py) — authoritative output fields.
6. [RapidOCR pipeline source](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/main.py) — detection/classification/recognition flow and combined-output construction.
7. [RapidOCR classifier source](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/ch_ppocr_cls/main.py) — crop classifier and 180° rotation behavior.

## Rerun inputs

```yaml
workflow: firecrawl-deep-research
topic: ONNX Runtime CUDA provider detection and RapidOCR page-orientation output
depth: quick
output: markdown
primary_sources: ONNX Runtime official documentation/source; RapidOCR official source; installed environment
firecrawl_status: attempted; search returned HTTP 402 and scrape returned insufficient credits
```
