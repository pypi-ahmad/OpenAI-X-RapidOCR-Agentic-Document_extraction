import pytest

from agentic_extractor.schema_input import (
    builder_to_schema,
    markdown_to_schema,
    parse_json_schema,
)


def test_markdown_fields_become_optional_strings() -> None:
    schema = markdown_to_schema(
        "## invoice_number\nThe printed invoice ID\n\n## total\nGrand total"
    )
    assert schema["properties"]["invoice_number"]["description"] == "The printed invoice ID"
    assert schema["properties"]["total"]["type"] == "string"
    assert "required" not in schema


def test_builder_and_raw_json_are_validated() -> None:
    schema = builder_to_schema(
        [{"name": "total", "type": "number", "description": "Total", "required": True}]
    )
    assert schema["required"] == ["total"]
    assert parse_json_schema('{"type":"object","properties":{"id":{"type":"string"}}}') == {
        "type": "object",
        "properties": {"id": {"type": "string"}},
    }


def test_invalid_schema_inputs_fail() -> None:
    with pytest.raises(ValueError, match="at least one"):
        markdown_to_schema("no headings")
    with pytest.raises(ValueError, match="Duplicate Markdown field name 'invoice_id'"):
        markdown_to_schema("## invoice_id\nFirst\n\n## invoice_id\nDuplicate")
    with pytest.raises(ValueError):
        parse_json_schema("[]")
    with pytest.raises(ValueError, match="Duplicate guided field name 'total'"):
        builder_to_schema(
            [
                {"name": "total", "type": "number"},
                {"name": "total", "type": "string"},
            ]
        )
    with pytest.raises(ValueError, match="at least one field"):
        builder_to_schema([])
    with pytest.raises(ValueError, match="object with properties"):
        parse_json_schema('{"type":"string"}')
