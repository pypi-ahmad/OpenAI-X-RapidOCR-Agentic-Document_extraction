from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_workflow_stages_are_separate_streamlit_pages() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    for page in ("classify", "section", "split", "extract"):
        assert f'"app_pages/{page}.py"' in source
    assert 'position="top"' in source


def test_each_workflow_page_opens_without_a_result() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20).run()
    assert not app.exception

    for path, title in (
        ("app_pages/classify.py", "Classify"),
        ("app_pages/section.py", "Section"),
        ("app_pages/split.py", "Split"),
        ("app_pages/extract.py", "Extract"),
    ):
        app.switch_page(path).run()
        assert not app.exception
        assert app.title[0].value.endswith(title)
