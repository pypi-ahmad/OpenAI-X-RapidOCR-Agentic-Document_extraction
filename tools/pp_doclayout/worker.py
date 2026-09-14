"""Isolated PP-DocLayoutV3 NDJSON worker for the local application.

Responsible for: running in an isolated virtual environment (`tools/pp_doclayout/.venv`),
executing PaddleX PP-DocLayoutV3 layout parsing and SLANeXt table structure
extraction, and streaming JSON responses over newline-delimited standard output.

Must not: be imported directly by the main application process (runs strictly as
an external subprocess driven over stdin/stdout), and must not print unformatted
logs to stdout (stdout is reserved exclusively for NDJSON protocol messages).

Next: `src/agentic_extractor/layout.py`, which spawns and manages communication
with this worker.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

MODEL_NAME = "PP-DocLayoutV3"
THRESHOLD = 0.5
TABLE_CLASSIFIER = "PP-LCNet_x1_0_table_cls"
WIRED_STRUCTURE = "SLANeXt_wired"
WIRELESS_STRUCTURE = "SLANet_plus"
_PACKAGED_CUDA_DLLS = (
    ("nvidia-cublas-cu12", "nvidia/cublas/bin"),
    ("nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime/bin"),
    ("nvidia-cuda-nvrtc-cu12", "nvidia/cuda_nvrtc/bin"),
)
_CUDA_DLL_HANDLES: list[Any] = []
_CONFIGURED_CUDA_DIRECTORIES: set[str] = set()


def _write(value: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.__stdout__.flush()


def _configure_packaged_cuda_dlls() -> list[str]:
    """Expose uv-installed CUDA 12 DLLs to Paddle's Windows dynamic loader."""
    if os.name != "nt":
        return []
    directories: list[str] = []
    for distribution_name, relative_path in _PACKAGED_CUDA_DLLS:
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        directory = Path(str(distribution.locate_file(relative_path)))
        if directory.is_dir():
            directories.append(str(directory))
    if directories:
        os.environ["PATH"] = os.pathsep.join([*directories, os.environ.get("PATH", "")])
        add_directory = getattr(os, "add_dll_directory", None)
        if callable(add_directory):
            for directory in directories:
                if directory not in _CONFIGURED_CUDA_DIRECTORIES:
                    _CUDA_DLL_HANDLES.append(add_directory(directory))
                    _CONFIGURED_CUDA_DIRECTORIES.add(directory)
    return directories


def _load_model() -> tuple[Any, str, str | None, bool]:
    _configure_packaged_cuda_dlls()
    with contextlib.redirect_stdout(sys.stderr):
        import paddle
        from paddlex import create_model

        use_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        if use_gpu:
            try:
                model = create_model(model_name=MODEL_NAME, device="gpu:0")
                _smoke(model)
                return model, "GPU", None, True
            except Exception as exc:
                warning = (
                    "PP-DocLayoutV3 GPU initialization failed; CPU fallback is active "
                    f"({type(exc).__name__})."
                )
                model = create_model(model_name=MODEL_NAME, device="cpu")
                _smoke(model)
                return model, "CPU", warning, True
        model = create_model(model_name=MODEL_NAME, device="cpu")
        _smoke(model)
        return (
            model,
            "CPU",
            "PP-DocLayoutV3 cannot access CUDA; CPU fallback is active.",
            False,
        )


def _smoke(model: Any) -> None:
    import numpy as np

    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    list(model.predict([image], batch_size=1, threshold=THRESHOLD))


def _decode(value: str) -> Any:
    import cv2
    import numpy as np

    raw = base64.b64decode(value, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("page image could not be decoded")
    return image


def _result_json(result: Any) -> dict[str, Any]:
    value = result.json
    if not isinstance(value, dict):
        raise TypeError("PaddleX result.json is not an object")
    return value


def _predict(model: Any, pages: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    images = [_decode(str(page["image"])) for page in pages]
    with contextlib.redirect_stdout(sys.stderr):
        results = list(model.predict(images, batch_size=batch_size, threshold=THRESHOLD))
    if len(results) != len(pages):
        raise RuntimeError("PaddleX returned a different number of page results")
    return [
        {"page": int(page["page"]), "result": _result_json(result)}
        for page, result in zip(pages, results, strict=True)
    ]


def _unwrap_result(result: Any) -> dict[str, Any]:
    value = _result_json(result)
    return value["res"] if set(value) == {"res"} and isinstance(value["res"], dict) else value


def _structure_geometry_fits(value: dict[str, Any], *, width: int, height: int) -> bool:
    """Return whether Paddle cell geometry fits the original table crop."""
    boxes = value.get("bbox")
    if not isinstance(boxes, list) or not boxes or width < 1 or height < 1:
        return False
    tolerance_x = max(4.0, width * 0.05)
    tolerance_y = max(4.0, height * 0.05)
    for box in boxes:
        if not isinstance(box, list):
            return False
        if len(box) == 8:
            xs, ys = box[::2], box[1::2]
        elif len(box) == 4 and all(not isinstance(item, list) for item in box):
            xs, ys = [box[0], box[2]], [box[1], box[3]]
        elif len(box) == 4 and all(isinstance(item, list) and len(item) == 2 for item in box):
            xs, ys = [item[0] for item in box], [item[1] for item in box]
        else:
            return False
        try:
            xs = [float(item) for item in xs]
            ys = [float(item) for item in ys]
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(item) for item in [*xs, *ys]):
            return False
        if (
            min(xs) < -tolerance_x
            or min(ys) < -tolerance_y
            or max(xs) > width + tolerance_x
            or max(ys) > height + tolerance_y
            or min(xs) >= max(xs)
            or min(ys) >= max(ys)
        ):
            return False
    return True


def _predict_tables(
    tables: list[dict[str, Any]], device: str, models: dict[str, Any]
) -> list[dict[str, Any]]:
    from paddlex import create_model

    def model(name: str) -> Any:
        if name not in models:
            with contextlib.redirect_stdout(sys.stderr):
                models[name] = create_model(
                    model_name=name, device="gpu:0" if device == "GPU" else "cpu"
                )
        return models[name]

    output: list[dict[str, Any]] = []
    for table in tables:
        image = _decode(str(table["image"]))
        with contextlib.redirect_stdout(sys.stderr):
            classifier_values = list(model(TABLE_CLASSIFIER).predict([image], batch_size=1))
        if len(classifier_values) != 1:
            raise RuntimeError("Table classifier returned an unexpected result count")
        classifier = _unwrap_result(classifier_values[0])
        labels = classifier.get("label_names") or []
        label = str(labels[0]).lower() if labels else ""
        structure_name = (
            WIRED_STRUCTURE if "wired" in label and "wireless" not in label else WIRELESS_STRUCTURE
        )
        with contextlib.redirect_stdout(sys.stderr):
            structure_values = list(model(structure_name).predict([image], batch_size=1))
        if len(structure_values) != 1:
            raise RuntimeError("Table structure model returned an unexpected result count")
        structure = _unwrap_result(structure_values[0])
        fallback = False
        geometry_valid = _structure_geometry_fits(
            structure, width=int(image.shape[1]), height=int(image.shape[0])
        )
        if not geometry_valid:
            alternate_name = (
                WIRELESS_STRUCTURE if structure_name == WIRED_STRUCTURE else WIRED_STRUCTURE
            )
            with contextlib.redirect_stdout(sys.stderr):
                alternate_values = list(model(alternate_name).predict([image], batch_size=1))
            if len(alternate_values) != 1:
                raise RuntimeError(
                    "Alternate table structure model returned an unexpected result count"
                )
            alternate = _unwrap_result(alternate_values[0])
            if _structure_geometry_fits(
                alternate, width=int(image.shape[1]), height=int(image.shape[0])
            ):
                structure_name = alternate_name
                structure = alternate
                fallback = True
                geometry_valid = True
        output.append(
            {
                "id": str(table["id"]),
                "page": int(table["page"]),
                "result": {
                    "classifier": classifier,
                    "structure": structure,
                    "structure_model": structure_name,
                    "structure_fallback": fallback,
                    "structure_geometry_valid": geometry_valid,
                },
            }
        )
    return output


def main() -> None:
    try:
        started = time.perf_counter()
        model, device, warning, gpu_available = _load_model()
        load_seconds = time.perf_counter() - started
        import paddle

        paddlex_version = importlib.metadata.version("paddlex")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

    table_models: dict[str, Any] = {}
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            operation = request.get("op")
            if operation == "health":
                _write(
                    {
                        "request_id": request_id,
                        "ok": True,
                        "model": MODEL_NAME,
                        "device": device,
                        "gpu_available": gpu_available,
                        "warning": warning,
                        "paddlex_version": paddlex_version,
                        "paddle_version": paddle.__version__,
                        "model_load_seconds": load_seconds,
                    }
                )
                continue
            if operation == "predict_tables":
                tables = request.get("tables")
                if not isinstance(tables, list) or not tables:
                    raise ValueError("predict_tables requires at least one table")
                started = time.perf_counter()
                values = _predict_tables(tables, device, table_models)
                _write(
                    {
                        "request_id": request_id,
                        "ok": True,
                        "tables": values,
                        "seconds": time.perf_counter() - started,
                    }
                )
                continue
            if operation != "predict":
                raise ValueError("unsupported worker operation")
            pages = request.get("pages")
            if not isinstance(pages, list) or not pages:
                raise ValueError("predict requires at least one page")
            started = time.perf_counter()
            batch_size = 2 if device == "GPU" else 1
            try:
                values = _predict(model, pages, batch_size)
                retried = False
            except Exception as exc:
                if device != "GPU" or batch_size == 1 or "memory" not in str(exc).lower():
                    raise
                values = _predict(model, pages, 1)
                retried = True
            _write(
                {
                    "request_id": request_id,
                    "ok": True,
                    "pages": values,
                    "seconds": time.perf_counter() - started,
                    "batch_size": 1 if retried else batch_size,
                    "memory_retry": retried,
                }
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _write(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


if __name__ == "__main__":
    main()
