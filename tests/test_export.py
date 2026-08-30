import io
import zipfile

from agentic_extractor.export import build_result_zip
from agentic_extractor.models import DocumentResult, Extraction


def test_result_zip_contains_portable_outputs() -> None:
    result = DocumentResult(markdown="# Result", blocks=[], extraction=Extraction(values={"x": 1}))
    bundle = build_result_zip(result)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert {
            "document.md",
            "result.json",
            "blocks.jsonl",
            "metadata.json",
            "extraction.json",
        } <= set(archive.namelist())
