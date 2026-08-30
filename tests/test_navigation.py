from pathlib import Path

from streamlit.testing.v1 import AppTest

from agentic_extractor.document_chat import ProcessedMarkdownDocument


def test_workflow_stages_are_separate_streamlit_pages() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    for page in ("classify", "section", "split", "extract", "chat"):
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
        ("app_pages/chat.py", "Document chat"),
    ):
        app.switch_page(path).run()
        assert not app.exception
        assert app.title[0].value.endswith(title)


def test_chat_selects_processed_markdown_and_clears_history_on_scope_change() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20).run()
    first = ProcessedMarkdownDocument(
        document_id="first",
        display_name="first.pdf",
        markdown="# First",
        selected_pages=[1],
    )
    second = ProcessedMarkdownDocument(
        document_id="second",
        display_name="second.pdf",
        markdown="# Second",
        selected_pages=[2],
    )
    app.session_state["processed_documents"] = {"first": first, "second": second}

    app.switch_page("app_pages/chat.py").run()

    assert not app.exception
    assert app.multiselect[0].value == ["second"]
    assert any("second.pdf" in item.value for item in app.markdown)

    app.session_state["chat_messages"] = [{"role": "user", "content": "old scope"}]
    app.multiselect[0].set_value(["first"]).run()

    assert not app.exception
    assert app.session_state["chat_messages"] == []
