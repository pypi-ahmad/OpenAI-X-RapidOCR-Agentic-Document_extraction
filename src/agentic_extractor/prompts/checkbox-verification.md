<!-- prompt-version: 4 -->

# Independent checkbox verification

Inspect each attached crop independently. Match each crop to its stated checkbox ID and page, and
return exactly one verification for every supplied ID—no additions or omissions. Use only control
pixels; label wording is not evidence of state.

First classify `control_status` independently from checkbox state:

- `checkbox`: the crop clearly contains a square checkbox control.
- `not_checkbox`: the proposed shape is a table cell/corner, text glyph, bullet, icon, signature,
  border, or other non-checkbox mark.
- `uncertain`: the crop does not contain enough pixels to decide whether a checkbox exists.

Only use `checkbox` when the control boundary is visibly supported by the crop. Use
`NOT_DETERMINABLE` for `state` whenever `control_status` is `not_checkbox` or `uncertain`.

- `CHECKED`: a deliberate tick, X selection mark, or fill is contained by the control.
- `UNCHECKED`: the boundary and empty interior are clearly visible.
- `INDETERMINATE`: a deliberate dash or partial fill is clearly visible.
- `CROSSED_OUT`: a cancellation stroke or scribble crosses the control.
- `NOT_DETERMINABLE`: the control is clipped, blurred, obscured, malformed, or ambiguous.

When discovery evidence and crop pixels conflict, report the crop-supported control status and state
with a reason. Use `uncertain` and `NOT_DETERMINABLE` rather than guessing, and keep confidence
conservative.

The supplied OpenCV, RapidOCR, and earlier Luna values are context for reconciliation only. Your
returned state must be an independent judgment from the attached crop pixels.

<CHECKBOX_EVIDENCE>
$checkbox_evidence
</CHECKBOX_EVIDENCE>
