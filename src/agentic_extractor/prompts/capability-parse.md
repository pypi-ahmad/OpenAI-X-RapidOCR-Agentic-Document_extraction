<!-- prompt-version: 3 -->

## Parse and refine

- Compare OCR with available visual evidence for text, headings, paragraphs, list items, table
  rows, key-value blocks, and reading order.
- Emit a refinement only when a block needs a correction or structural change.
- `corrected_text` is the complete replacement text for the cited block, never an uncited fragment.
- Use each positive `reading_order` value at most once per page.
- Set `verified=true` only with valid evidence. Otherwise set `abstained=true` and explain why.
- For every OCR block with `requires_gpt_review=true`, emit exactly one refinement outcome.
  Confirm unchanged text with `verified=true`, the original text in `corrected_text`, and grounded
  evidence; otherwise provide a grounded correction or set `abstained=true` with a reason.
