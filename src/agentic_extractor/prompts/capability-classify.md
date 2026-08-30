<!-- prompt-version: 2 -->

## Classify

- Use only the allowed labels. Use `unknown` when none is supported.
- For page scope, `page_start` must equal `page_end`. Document scope may cover a contiguous range.
- Every classification, including `unknown`, must cite evidence within its declared page range.
- Return no duplicate classification for the same scope and page range.
