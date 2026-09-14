<!-- prompt-version: 5 -->

## Checkbox discovery

Inspect only attached page images listed in `CANDIDATE_PAGES` and report visible square checkbox
controls on those pages. Do not inspect or report checkboxes on any other page. A real checkbox
inside a form table or dark header band remains a checkbox; inspect both dark-on-light and
light-on-dark controls. Exclude empty table-cell rectangles, radio buttons, bullets, signatures,
icons, and decorative squares. Determine state only from the control pixels, never from label
meaning:

- `CHECKED`: a deliberate tick, X selection mark, or fill is contained by the control.
- `UNCHECKED`: the control boundary and an empty interior are both clearly visible.
- `INDETERMINATE`: a deliberate dash or partial-fill state is clearly visible.
- `CROSSED_OUT`: a cancellation stroke or scribble crosses the control rather than selecting it.
- `NOT_DETERMINABLE`: the control is clipped, blurred, obscured, malformed, or visually ambiguous.

Return controls in top-to-bottom, then left-to-right order per page. Assign IDs
`p{page}-c{ordinal}` starting at `1`. Include normalized control geometry, nearby visible label and
label geometry, conservative confidence, and RapidOCR label evidence when available. Do not infer a
missing checkbox from text alone.

OpenCV proposals below are untrusted local evidence, not authoritative answers. Inspect the page
pixels independently. Reuse a proposal ID when the visible control overlaps that proposal; report
visible controls missed by OpenCV with the normal `p{page}-c{ordinal}` ID pattern. Do not copy a
local state unless the pixels support it.

<CANDIDATE_PAGES>
$candidate_pages
</CANDIDATE_PAGES>

<LOCAL_CHECKBOX_EVIDENCE>
$local_checkbox_evidence
</LOCAL_CHECKBOX_EVIDENCE>
