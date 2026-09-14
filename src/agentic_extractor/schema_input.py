"""Convert simple user field definitions into JSON Schema.

Responsible for: normalizing the three supported extraction-schema inputs
(Markdown `## field_name` sections, guided builder rows, raw JSON Schema)
into one validated internal schema shape. Must not: allow two of the three
input paths to disagree on what counts as a valid schema — all three funnel
through `validate_schema` so that invariant lives in one place, not three.
Field names must stay unique per input; a silent collision would drop a
field's definition without the caller ever choosing to. Next:
capabilities.py, which validates GPT extraction output against this schema.
"""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator

FIELD_HEADING = re.compile(r"^##\s+([A-Za-z_][A-Za-z0-9_.-]*)\s*$", re.MULTILINE)


def markdown_to_schema(markdown: str) -> dict[str, Any]:
    matches = list(FIELD_HEADING.finditer(markdown))
    if not matches:
        raise ValueError("Markdown schema needs at least one '## field_name' heading.")
    properties: dict[str, Any] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        # Reject rather than overwrite: a later duplicate heading would
        # otherwise silently discard the first field's description.
        if name in properties:
            raise ValueError(f"Duplicate Markdown field name {name!r}.")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        description = markdown[match.end() : end].strip()
        properties[name] = {"type": "string", "description": description}
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    validate_schema(schema)
    return schema


def builder_to_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        if name in properties:
            # Enforce uniqueness identical to markdown_to_schema.
            raise ValueError(f"Duplicate guided field name {name!r}.")
        definition: dict[str, Any] = {
            "type": row.get("type", "string"),
            "description": str(row.get("description", "")),
        }
        for key in ("format", "pattern"):
            if value := str(row.get(key, "")).strip():
                definition[key] = value
        for key in ("minimum", "maximum"):
            value = row.get(key)
            if value not in (None, ""):
                definition[key] = float(value)
        properties[name] = definition
        if row.get("required"):
            required.append(name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    validate_schema(schema)
    return schema


def parse_json_schema(value: str) -> dict[str, Any]:
    schema = json.loads(value)
    if not isinstance(schema, dict):
        raise ValueError("The JSON Schema root must be an object.")
    validate_schema(schema)
    return schema


def validate_schema(schema: dict[str, Any]) -> None:
    # Single funnel for all three input paths (Markdown, guided, raw JSON
    # Schema): whatever shape rules apply to extraction schemas belong here,
    # not duplicated per-caller.
    Draft202012Validator.check_schema(schema)
    properties = schema.get("properties")
    if schema.get("type") != "object" or not isinstance(properties, dict):
        raise ValueError("Extraction schema must be an object with properties.")
    if not properties:
        raise ValueError("Extraction schema must define at least one field.")
