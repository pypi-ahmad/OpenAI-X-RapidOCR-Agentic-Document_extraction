from __future__ import annotations

import runpy
from pathlib import Path

import pytest


def test_table_worker_rejects_square_canvas_geometry_for_a_shallow_crop() -> None:
    worker = runpy.run_path(
        str(Path(__file__).parents[1] / "tools" / "pp_doclayout" / "worker.py"),
        run_name="table_worker_contract",
    )
    geometry_fits = worker["_structure_geometry_fits"]

    assert not geometry_fits(
        {"bbox": [[28, 5, 735, 6, 715, 229, 26, 227]]},
        width=1102,
        height=140,
    )
    assert geometry_fits(
        {"bbox": [[2, 4, 1099, 4, 1099, 132, 2, 132]]},
        width=1102,
        height=140,
    )


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows DLL search behavior")
def test_worker_exposes_packaged_cuda_dll_directories(monkeypatch, tmp_path: Path) -> None:
    worker = runpy.run_path(
        str(Path(__file__).parents[1] / "tools" / "pp_doclayout" / "worker.py"),
        run_name="table_worker_cuda_contract",
    )
    cublas = tmp_path / "cublas"
    runtime = tmp_path / "runtime"
    nvrtc = tmp_path / "nvrtc"
    for directory in (cublas, runtime, nvrtc):
        directory.mkdir()
    directories = iter((cublas, runtime, nvrtc))

    class Distribution:
        def __init__(self, directory: Path) -> None:
            self.directory = directory

        def locate_file(self, _relative_path: str) -> Path:
            return self.directory

    monkeypatch.setattr(
        worker["importlib"].metadata,
        "distribution",
        lambda _name: Distribution(next(directories)),
    )
    monkeypatch.setenv("PATH", "existing")
    added_directories: list[str] = []
    monkeypatch.setattr(
        worker["os"],
        "add_dll_directory",
        lambda directory: added_directories.append(directory) or object(),
    )

    configured = worker["_configure_packaged_cuda_dlls"]()

    assert configured == [str(cublas), str(runtime), str(nvrtc)]
    assert worker["os"].environ["PATH"].split(worker["os"].pathsep)[:3] == configured
    assert added_directories == configured
