<!-- prompt-version: 2 -->

# Independent checkbox verification

Inspect each attached crop independently. Match each crop to its stated checkbox ID and page, and
return exactly one verification for every supplied ID—no additions or omissions. Use only control
pixels; label wording is not evidence of state.

- `CHECKED`: a deliberate tick, X selection mark, or fill is contained by the control.
- `UNCHECKED`: the boundary and empty interior are clearly visible.
- `INDETERMINATE`: a deliberate dash or partial fill is clearly visible.
- `CROSSED_OUT`: a cancellation stroke or scribble crosses the control.
- `NOT_DETERMINABLE`: the control is clipped, blurred, obscured, malformed, or ambiguous.

When discovery evidence and crop pixels conflict, report the crop-supported state with a reason. Use
`NOT_DETERMINABLE` rather than guessing, and keep confidence conservative.

<CHECKBOX_EVIDENCE>
$checkbox_evidence
</CHECKBOX_EVIDENCE>
