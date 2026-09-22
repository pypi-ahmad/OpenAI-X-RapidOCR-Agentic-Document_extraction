<!-- prompt-version: 17 -->

## Parse and refine

- Inspect each full-page image for content absent from OCR. Return those additions separately
  in `visual_objects`: missed text, equations (LaTeX without display delimiters), and figure or
  chart descriptions. Cite a tight normalized positive-area `bbox`, page and reading order.
  Do not duplicate text already represented by OCR; correct existing blocks via refinements.
  Descriptions must be literal visual descriptions, never inferred values, identities or causes.
  Do not reconstruct redacted content or signatures. Use temporary unique IDs; the application
  assigns stable IDs and independently verifies crops before publishing additions.
- Return `document_links` only for visibly supported continuations, captions, or parent/child
  relationships. Refer to semantic region IDs or visual-object IDs. Keep page fragments intact;
  do not duplicate continuation text. Unknown relationships must be omitted, not guessed.

- Compare OCR with available visual evidence for text, headings, paragraphs, list items, table
  rows, key-value blocks, and reading order.
- Emit a refinement only when a block needs a correction or structural change.
- `corrected_text` is the complete replacement text for the cited block, never an uncited fragment.
- Use each positive `reading_order` value at most once per page.
- Set `verified=true` only with valid evidence. Otherwise set `abstained=true` and explain why.
- Inspect dates, identifiers, decimal punctuation, comparison operators, dosage units, redaction
  placeholders, and signature/attestation markers character by character. Correct them only from
  visible pixels inside the supplied page or region; never normalize away a source marker.
- When a routed OCR block is visibly handwritten signature ink, do not transcribe or guess a name.
  Replace the complete block with exactly two lines: first `[SIGNED]`, then
  `[ILLEGIBLE_SIGNATURE]`. Use block type `paragraph` and cite the signature pixels as
  `gpt-visual` evidence. Visible but illegible signature ink is a supported semantic-marker
  outcome, not a reason to preserve OCR-like noise.
  Keep any adjacent `Electronically signed by:` line unchanged; the application links its
  electronic-attestation marker to the grounded signature outcome.
- For every OCR block with `requires_gpt_review=true`, emit exactly one refinement outcome.
  Confirm unchanged text with `verified=true`, the original text in `corrected_text`, and grounded
  evidence; otherwise provide a grounded correction or set `abstained=true` with a reason.
- For every supplied local redaction candidate ID, emit exactly one refinement outcome. If its
  high-detail pixels show an opaque source mask, use that candidate ID as `block_id`, set
  `corrected_text` to exactly `[REDACTED]`, set `verified=true`, and cite a `gpt-visual` bounding
  box over the visible mask. Otherwise set `abstained=true`. Never infer or reconstruct masked text,
  and never treat dark headings, table cells, borders, logos, or filled controls as redactions.
- Leave `table_reviews` empty. The application validates tables in a separate bounded visual call.
- Return a compact `semantic_regions` partition for every page. Each region groups one or more
  existing OCR `source_block_ids` into exactly one semantic object. Across a page, include every
  supplied OCR block exactly once: never omit, duplicate, or invent an ID. Use `reading_order` for
  semantic reading order and preserve the intended order of blocks inside `source_block_ids`.
- Start from the application-supplied `LOCAL_SEMANTIC_REGIONS`, which deterministically group OCR
  blocks using PP-DocLayoutV3 and geometry before this request. Preserve a supported local group;
  split or merge groups only when the supplied text, layout, or visual evidence requires it.
- Never group multiple visibly distinct tables or bordered form panels into one semantic table,
  even when PP-DocLayoutV3 supplied one oversized region. Split them at visible outer borders or
  section boundaries and cite the same source layout region ID on each resulting semantic table.
  Conversely, do not split one logical table merely because it contains internal row borders.
- Use `source_layout_region_ids` only when supplied PP-DocLayoutV3 regions support the grouping.
  Choose `join_style=space` for wrapped lines in one paragraph and `line_break` for forms,
  marginalia, and intentionally separate lines. Use `heading`, `list`, `form`, `table`,
  `marginalia`, `logo`, `attestation`, `figure`, or `scan_code` only when the supplied layout,
  text, or visual evidence
  supports it. PP-DocLayoutV3 has no semantic `logo` class: identify each graphical brand lockup
  (symbol, stylized wordmark, and its visually grouped tagline) even when its OCR blocks are labeled
  as `text` or `heading`, and cite a tight `gpt-visual` bounding box in that semantic region's
  `evidence`. A brand name presented as ordinary header or body text without a graphical symbol or
  distinct lockup remains text, not a logo. In particular, keep a small standalone manufacturer or
  vendor name in a running page header as text even when its typeface is branded. Keep separately
  grounded electronic-attestation text and handwritten-signature markers in separate
  `attestation` regions; never merge them into one object. Use `figure` for a visually grounded
  photo, diagram, illustration, or flowchart, and `scan_code` for a visually grounded barcode or
  QR code. Cite a tight `gpt-visual` bounding box for either type; ordinary text and decorative
  rules remain text.
  Do not write reconstructed text in this structure; the application renders exclusively from the
  cited source blocks and separately accepted table cells.
