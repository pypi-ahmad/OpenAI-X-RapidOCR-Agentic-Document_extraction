# GPT-6 Sol and visual document review

All OpenAI calls use `gpt-6-sol` with medium reasoning, including Parse, table and checkbox
review, visual crop verification, downstream workflows, repair, and document chat. Local
RapidOCR and PP-DocLayoutV3 remain required; they are not alternative OpenAI models.

Rates per million tokens are $2.00 input, $0.20 cached input, $2.50 cache writes and $10.00 output.
The existing long-input multipliers apply above 272,000 input tokens. Missing usage or cache
details remain explicitly unavailable or estimated. No SDK retries or blind malformed-JSON
retries run behind the usage panel. Response usage is captured before structured-output parsing.

## Rich content and review

Parse can propose text missed by OCR, equations, and figure/chart descriptions with page-relative
bounding boxes. These proposals never become raw OCR blocks. The application assigns stable IDs,
independently checks source crops with the same model, and requires exact agreement before
publishing recovered text. Model verification is not human approval. Unverified objects render a
review placeholder; their proposals remain in the audit output. Descriptions are labelled as
descriptions, not transcriptions, and cannot serve as chunk evidence for exact extracted values.

Each workflow invocation has at most two repair rounds, with at most eight visual objects inspected
per round. Identical crop inspections are not repeated, and unchanged review findings stop the
loop. Remaining uncertainty requires human review. PDF inspection can render selected pages at
up to 300 DPI within the existing 25-million-pixel limit. Image inputs retain their source pixels.
Reusing an existing Parse skips new visual-object inspections. The two-round bound is not a cap
on total document API calls: page batches, tables, checkboxes and downstream operations also call Sol.

Continuation, caption, and parent links preserve page fragments. Endpoints, order, duplicates,
and cycles are checked. These links are model proposals, not independently proven semantics.
Existing native table cells and merged-cell HTML remain the table representation.

Streamlit's Visual content review shows the source crop and proposed text. Approve, correct and
reject decisions require a reason and append an audit record bound to the proposal's hash.
They rebuild canonical Markdown and exports without a paid call or an OCR rerun.

The Parse JSON adds `visual_objects`, `visual_audits` and `document_links`. Structural visual
nodes carry source IDs, provenance, verification status, bounding boxes and character ranges in
the actual exported Markdown. Annotated PDF and coordinate HTML include visual additions.

## Validation and limits

Offline tests exercise geometry rejection, abstention, immutable OCR, review decisions,
relationship validation, raw-response billing and budget reservations. These are regression tests,
not evidence of general document accuracy or parity with commercial parsers.

The opt-in runner below generates a two-page synthetic scanned PDF and runs the real pipeline.
It must only be run with explicit paid-test authorization. It does not use private source files.

```powershell
uv run python tools/validate_sol_synthetic.py validation-output/sol-synthetic-run --authorize-up-to-usd 5
```

The runner counts input tokens exactly, reserves the highest input rate plus the 16,384
output-token cap before each generation call, and disables hidden retries. If billing is missing,
it retains the reservation and blocks additional paid requests. Reported estimates are not labelled exact.
The output directory must be new. Current runs persist reservations to `billing.json` before
generation and after settlement. The limit is enforced per runner instance; it is not loaded
from an earlier run or applied to normal UI/API usage. Authorization remains cumulative across
the user's validation session, so prior spending must be considered before a repeat run. Do not
rerun with a fresh budget after an uncertain request. See [Testing](TESTING.md) for checks and limits.

## Research basis

- [OpenAI GPT-6 Sol model reference](https://developers.openai.com/api/docs/models/gpt-6-sol)
- [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
- [Structured outputs and refusals](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Exact input-token counting](https://developers.openai.com/api/docs/guides/token-counting)
- [LlamaParse document parsing](https://developers.llamaindex.ai/llamaparse/parse/)
- [LandingAI ADE grounded JSON](https://docs.landing.ai/ade/ade-json-response)

The last two inform the layout/grounding design; neither is an added service dependency.

## Recorded synthetic run

On 2026-09-23 the two-page fixture completed with no failed pages or unresolved review items.
All six asserted text values survived, and two figure/chart objects passed crop verification.
Five GPT-6 Sol calls reported 31,096 input tokens and 4,171 output tokens, costing $0.1194425
at the configured rates. Artifacts and provider usage are in the ignored local directory
`validation-output/sol-synthetic-20260923/`. This narrow check does not measure general accuracy.
