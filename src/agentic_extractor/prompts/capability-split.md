<!-- prompt-version: 2 -->

## Split

- Treat API batches as irrelevant. Segment the selected pages as one ordered document sequence.
- Splits must be contiguous, non-overlapping, and cover every selected page exactly once.
- The first split starts on the first selected page; each later split starts immediately after the
  previous split and cites evidence at or immediately before its boundary.
- Prefer one complete split over an unsupported boundary.
