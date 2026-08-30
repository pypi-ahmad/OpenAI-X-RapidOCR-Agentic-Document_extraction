<!-- prompt-version: 2 -->

## Checkbox discovery

Inspect every attached page image and report every visible square checkbox control. Exclude radio
buttons, bullets, table cells, signatures, icons, and decorative squares. Determine state only from
the control pixels, never from label meaning:

- `CHECKED`: a deliberate tick, X selection mark, or fill is contained by the control.
- `UNCHECKED`: the control boundary and an empty interior are both clearly visible.
- `INDETERMINATE`: a deliberate dash or partial-fill state is clearly visible.
- `CROSSED_OUT`: a cancellation stroke or scribble crosses the control rather than selecting it.
- `NOT_DETERMINABLE`: the control is clipped, blurred, obscured, malformed, or visually ambiguous.

Return controls in top-to-bottom, then left-to-right order per page. Assign IDs
`p{page}-c{ordinal}` starting at `1`. Include normalized control geometry, nearby visible label and
label geometry, conservative confidence, and RapidOCR label evidence when available. Do not infer a
missing checkbox from text alone.
