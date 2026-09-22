"""Self-contained coordinate-positioned HTML for grounded Parse results.

Responsible for: generating an interactive, standalone HTML document viewer
containing base64 raster images, selectable transparent/visible text overlays
aligned with OCR coordinates, and interactive SVG geometry layers (layout
regions, table cells, checkboxes).

Must not: load external network scripts, fonts, or assets (strictly enforced
by Content Security Policy: `default-src 'none'; img-src data:; style-src
'unsafe-inline'; script-src 'nonce-layout-viewer'`).

Next: `artifacts.py`, which invokes `build_coordinate_html` lazily on demand.
"""

# ruff: noqa: E501 -- HTML, CSS, and JavaScript stay readable as standalone source lines.

from __future__ import annotations

import base64
import hashlib
import html
import io
import math
from typing import Any

from PIL import Image

from agentic_extractor.models import Block, RefinementRecord
from agentic_extractor.ocr import LocalParseResult
from agentic_extractor.parse import PageParse, is_publishable_checkbox


def build_coordinate_html(result: LocalParseResult) -> str:
    """Render selected Parse pages on their OCR-aligned image coordinate plane."""
    accepted_table_ids = _accepted_table_ids(result.document_metadata)
    pages = "".join(
        _page_html(
            page,
            result.refinements,
            [
                item
                for item in result.checkboxes
                if item.page == page.page and is_publishable_checkbox(item)
            ],
            accepted_table_ids,
        )
        for page in result.pages
    )
    title = html.escape(str(result.document_metadata.get("file_name", "Document")))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'nonce-layout-viewer'; base-uri 'none'; form-action 'none'">
<title>{title} — grounded layout</title>
<style>
:root{{--bg:#101116;--panel:#191a35;--pink:#d72d7a;--cyan:#33d6e8;--text:#f4f1f6;--muted:#a8adbd}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}}
.toolbar{{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:12px 16px;background:#191a35f5;border-bottom:1px solid #303253}}
button,select,input{{font:inherit}} button,select{{border:1px solid #45486b;border-radius:6px;background:#252746;color:var(--text);padding:7px 10px}} button:hover{{border-color:var(--cyan)}}
input[type=search]{{min-width:220px;border:1px solid #45486b;border-radius:6px;background:#101225;color:var(--text);padding:7px 10px}}
label{{display:flex;gap:6px;align-items:center;color:var(--muted)}} #match-count{{color:var(--cyan)}}
.workspace{{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:18px;padding:18px}}
main{{min-width:0;overflow:auto}} .page-shell{{width:min(100%,1000px);margin:0 auto 24px}}
.page-label{{margin:0 0 6px;color:var(--muted)}} .page-canvas{{position:relative;width:100%;overflow:hidden;background:#fff;box-shadow:0 2px 8px #0008;container-type:inline-size}}
.page-image,.text-layer,.geometry-layer{{position:absolute;inset:0;width:100%;height:100%}} .page-image{{display:block;user-select:none}}
.text-run{{position:absolute;z-index:2;overflow:hidden;white-space:pre-wrap;color:transparent;line-height:1;cursor:text;transform-origin:left top}}
.text-run[data-layer=raw]{{display:none}} body.raw-layer .text-run[data-layer=refined]{{display:none}} body.raw-layer .text-run[data-layer=raw]{{display:block}}
body.visible-text .text-run{{color:#111;background:#ffffffe6;outline:1px solid #33d6e888}}
.text-run.match{{outline:3px solid #00bcd4;background:#00bcd455}} .text-run::selection{{color:#111;background:#33d6e8aa}}
.geometry-layer{{z-index:3;display:none;pointer-events:none}} body.show-geometry .geometry-layer{{display:block;pointer-events:auto}}
.region{{fill:#d72d7a18;stroke:var(--pink);stroke-width:2;vector-effect:non-scaling-stroke;cursor:pointer}} .layout-region{{stroke:var(--cyan);fill:#33d6e812}} .table-cell{{stroke:#f0a020;fill:#f0a02012}} .checkbox-region{{stroke:#7bd88f;fill:#7bd88f22}}
aside{{position:sticky;top:76px;align-self:start;max-height:calc(100vh - 94px);overflow:auto;padding:16px;border:1px solid #303253;border-radius:8px;background:var(--panel)}} aside h2{{margin-top:0;font-size:16px}} aside dl{{display:grid;grid-template-columns:auto 1fr;gap:6px 10px}} aside dt{{color:var(--muted)}} aside dd{{margin:0;overflow-wrap:anywhere}}
@media(max-width:850px){{.workspace{{grid-template-columns:1fr}} aside{{position:static;max-height:none}}}}
@media print{{.toolbar,aside{{display:none}}.workspace{{display:block;padding:0}}.page-shell{{width:100%!important;break-after:page;margin:0}}body{{background:#fff}}}}
</style></head><body>
<div class="toolbar" role="toolbar" aria-label="Layout viewer controls">
<button id="zoom-out" type="button" aria-label="Zoom out">−</button><button id="fit" type="button">Fit</button><button id="zoom-in" type="button" aria-label="Zoom in">+</button>
<label>Text <select id="text-layer"><option value="refined">Refined</option><option value="raw">Raw OCR</option></select></label>
<label><input id="visible-text" type="checkbox"> Show text layer</label><label><input id="geometry" type="checkbox"> Show geometry</label>
<input id="search" type="search" placeholder="Search positioned text" aria-label="Search positioned text"><span id="match-count" aria-live="polite"></span>
</div><div class="workspace"><main>{pages or "<p>No completed pages are available.</p>"}</main>
<aside aria-live="polite"><h2>Grounding inspector</h2><p id="inspect-empty">Select a positioned text or geometry region.</p><dl id="inspect"></dl></aside></div>
<script nonce="layout-viewer">
(()=>{{const body=document.body,pages=[...document.querySelectorAll('.page-shell')];let zoom=1;
const resize=()=>pages.forEach(p=>{{p.style.width=zoom===1?'':Math.round(Number(p.dataset.width)*zoom)+'px'}});
document.querySelector('#zoom-in').onclick=()=>{{zoom=Math.min(3,zoom+.25);resize()}};
document.querySelector('#zoom-out').onclick=()=>{{zoom=Math.max(.5,zoom-.25);resize()}};
document.querySelector('#fit').onclick=()=>{{zoom=1;resize()}};
document.querySelector('#text-layer').onchange=e=>body.classList.toggle('raw-layer',e.target.value==='raw');
document.querySelector('#visible-text').onchange=e=>body.classList.toggle('visible-text',e.target.checked);
document.querySelector('#geometry').onchange=e=>body.classList.toggle('show-geometry',e.target.checked);
const search=document.querySelector('#search'),count=document.querySelector('#match-count');
search.oninput=()=>{{const q=search.value.trim().toLowerCase();let n=0,first=null;document.querySelectorAll('.text-run').forEach(el=>{{const active=body.classList.contains('raw-layer')?el.dataset.layer==='raw':el.dataset.layer==='refined';const hit=!!q&&active&&el.textContent.toLowerCase().includes(q);el.classList.toggle('match',hit);if(hit){{n++;first??=el}}}});count.textContent=q?`${{n}} match${{n===1?'':'es'}}`:'';first?.scrollIntoView({{block:'center'}})}};
const inspect=document.querySelector('#inspect'),empty=document.querySelector('#inspect-empty');
document.addEventListener('click',event=>{{const el=event.target.closest?.('.text-run,.region');if(!el)return;empty.hidden=true;inspect.replaceChildren();Object.entries(el.dataset).forEach(([key,value])=>{{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key.replace(/[A-Z]/g,m=>' '+m.toLowerCase());dd.textContent=value;inspect.append(dt,dd)}})}});
}})();
</script></body></html>"""


def coordinate_html_page_images(result: LocalParseResult) -> list[dict[str, Any]]:
    """Describe the exact page rasters embedded in the standalone viewer."""
    return [
        {
            "source_page": page.page,
            "mime_type": "image/png",
            "sha256": hashlib.sha256(_lossless_page_image(page)).hexdigest(),
        }
        for page in result.pages
    ]


def _page_html(
    page: PageParse,
    refinements: list[RefinementRecord],
    checkboxes: list[Any],
    accepted_table_ids: set[str] | None,
) -> str:
    png = _lossless_page_image(page)
    image_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    accepted = _accepted_refinements(page, refinements)
    blocks = _ordered_blocks(page)
    raw = "".join(
        _text_run(page, block, block.text, "raw", order, block.source)
        for order, block in enumerate(blocks, 1)
        if block.source == "rapidocr"
    )
    refined = "".join(
        _text_run(
            page,
            block,
            accepted.get(block.id, block.text),
            "refined",
            order,
            "gpt-6-sol + rapidocr" if block.id in accepted else block.source,
        )
        for order, block in enumerate(blocks, 1)
    )
    geometry = _geometry(page, checkboxes, accepted_table_ids)
    for chunk in page.chunks:
        if chunk.provenance != "gpt-visual":
            continue
        # A derived display block only; immutable page.blocks is not extended.
        derived = Block(
            id=chunk.id,
            page=page.page,
            type=chunk.type,
            text=chunk.text,
            bbox=chunk.bbox,
            source="gpt-visual",
        )
        refined += _text_run(
            page,
            derived,
            chunk.text,
            "refined",
            chunk.reading_order,
            f"gpt-6-sol · {chunk.verification}",
        )
        geometry += _polygon(
            _bbox_polygon(chunk.bbox, page.width, page.height),
            "region block-region",
            {
                "source-id": chunk.id,
                "region-type": chunk.type,
                "verification": str(chunk.verification),
            },
        )
    ratio = page.height / page.width * 100 if page.width else 100
    return (
        f'<section class="page-shell" data-source-page="{page.page}" data-width="{page.width}" '
        f'data-height="{page.height}"><p class="page-label">Page {page.page}</p>'
        f'<div class="page-canvas" style="padding-top:{ratio:.6f}%" '
        f'aria-label="Document page {page.page}"><img class="page-image" src="{image_uri}" '
        f'alt="Source page {page.page}"><div class="text-layer">{raw}{refined}</div>'
        f'<svg class="geometry-layer" viewBox="0 0 {page.width} {page.height}" '
        f'aria-label="Grounding geometry">{geometry}</svg></div></section>'
    )


def _lossless_page_image(page: PageParse) -> bytes:
    if page.layout_image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return page.layout_image_bytes
    source = page.layout_image_bytes or page.image_bytes
    with Image.open(io.BytesIO(source)) as image:
        output = io.BytesIO()
        image.convert("RGB").save(output, "PNG", optimize=True)
        return output.getvalue()


def _accepted_refinements(page: PageParse, refinements: list[RefinementRecord]) -> dict[str, str]:
    block_ids = {block.id for block in page.blocks}
    accepted: dict[str, str] = {}
    for item in refinements:
        if (
            item.page == page.page
            and item.status == "accepted"
            and item.block_id in block_ids
            and item.proposed_text is not None
        ):
            accepted[item.block_id] = item.proposed_text
    return accepted


def _ordered_blocks(page: PageParse) -> list[Block]:
    # Reading order preservation: sequence blocks according to layout reading order,
    # then append any unmapped blocks to ensure no OCR text is omitted.
    by_id = {block.id: block for block in page.blocks}
    ids = page.reading_order_evidence.ordered_block_ids if page.reading_order_evidence else []
    ordered = [by_id[item] for item in ids if item in by_id]
    seen = {block.id for block in ordered}
    return ordered + [block for block in page.blocks if block.id not in seen]


def _text_run(
    page: PageParse, block: Block, text: str, layer: str, order: int, provenance: str
) -> str:
    bbox = _valid_bbox(block.bbox)
    if bbox is None:
        return ""
    left, top, right, bottom = bbox
    style = (
        f"left:{left * 100:.4f}%;top:{top * 100:.4f}%;"
        f"width:{(right - left) * 100:.4f}%;height:{(bottom - top) * 100:.4f}%;"
        f"font-size:{(bottom - top) * page.height / page.width * 85:.4f}cqw;"
    )
    bbox_text = ", ".join(f"{value:.4f}" for value in bbox)
    return (
        f'<span class="text-run" data-layer="{layer}" data-source-id="{html.escape(block.id)}" '
        f'data-page="{block.page}" data-block-type="{html.escape(block.type)}" '
        f'data-reading-order="{order}" data-confidence="{_score(block.ocr_score)}" '
        f'data-provenance="{html.escape(provenance)}" data-bbox="{bbox_text}" style="{style}">'
        f"{html.escape(text)}</span>"
    )


def _geometry(page: PageParse, checkboxes: list[Any], accepted_table_ids: set[str] | None) -> str:
    regions: list[str] = []
    for block in page.blocks:
        points = block.polygon or _bbox_polygon(block.bbox, page.width, page.height)
        if points:
            regions.append(
                _polygon(
                    points,
                    "region block-region",
                    {
                        "source-id": block.id,
                        "region-type": block.type,
                        "confidence": _score(block.ocr_score),
                    },
                )
            )
    for region in page.layout_regions:
        points = region.polygon or _bbox_polygon(region.bbox, page.width, page.height)
        if points:
            regions.append(
                _polygon(
                    points,
                    "region layout-region",
                    {
                        "layout-region-id": region.id,
                        "layout-label": region.label,
                        "confidence": _score(region.score),
                    },
                )
            )
    for table in page.table_structures:
        if (
            table.status != "valid"
            or table.review_required
            or (accepted_table_ids is not None and table.id not in accepted_table_ids)
        ):
            continue
        for cell in table.cells:
            points = _bbox_polygon(cell.bbox, page.width, page.height)
            if points:
                regions.append(
                    _polygon(
                        points,
                        "region table-cell",
                        {
                            "table-cell-id": cell.id,
                            "table-id": table.id,
                            "source-text": cell.source_text,
                        },
                    )
                )
    for checkbox in checkboxes:
        points = _bbox_polygon(checkbox.control_bbox, page.width, page.height)
        if points:
            regions.append(
                _polygon(
                    points,
                    "region checkbox-region",
                    {
                        "checkbox-id": checkbox.id,
                        "checkbox-state": checkbox.state.value,
                        "label": checkbox.label,
                    },
                )
            )
    return "".join(regions)


def _accepted_table_ids(metadata: dict[str, Any]) -> set[str] | None:
    """Return audited table IDs, or None when no Sol audit exists yet."""
    audits = metadata.get("table_reviews")
    if not isinstance(audits, list):
        return None
    return {
        str(item["table_id"])
        for item in audits
        if isinstance(item, dict)
        and item.get("status") == "accepted"
        and item.get("table_id") is not None
    }


def _polygon(points: list[list[float]], css_class: str, data: dict[str, str]) -> str:
    if not points or any(
        len(point) < 2 or not all(math.isfinite(float(v)) for v in point[:2]) for point in points
    ):
        return ""
    point_text = " ".join(f"{float(point[0]):.4f},{float(point[1]):.4f}" for point in points)
    attrs = " ".join(f'data-{key}="{html.escape(str(value))}"' for key, value in data.items())
    return f'<polygon class="{css_class}" points="{point_text}" {attrs}></polygon>'


def _bbox_polygon(bbox: list[float] | None, width: int, height: int) -> list[list[float]]:
    valid = _valid_bbox(bbox)
    if valid is None:
        return []
    left, top, right, bottom = valid
    return [
        [left * width, top * height],
        [right * width, top * height],
        [right * width, bottom * height],
        [left * width, bottom * height],
    ]


def _valid_bbox(value: list[float] | None) -> tuple[float, float, float, float] | None:
    if value is None or len(value) != 4:
        return None
    left, top, right, bottom = (float(item) for item in value)
    if not all(math.isfinite(item) for item in (left, top, right, bottom)):
        return None
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        return None
    return left, top, right, bottom


def _score(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.4f}"
