<!-- prompt-version: 2 -->

# Cross-batch reconciliation

Technical API batches are not document boundaries. Reconcile only classifications, sections,
splits, and extracted fields requested by the listed capabilities into one coherent document result.
Resolve conflicts in favor of stronger valid evidence; otherwise preserve uncertainty or abstain.
Never convert repeated candidate claims into independent corroboration.

Set `reviewed_pages` to every page present in `DOCUMENT_SUMMARIES`, exactly once. Set
`refined_markdown` to an empty string and leave `refinements` and `checkboxes` empty because those
outputs are not reconciled by this call.

<APPLICATION_CONFIGURATION>
Capabilities: $capabilities
Allowed classification labels: $allowed_classes
Extraction JSON Schema: $extraction_schema
</APPLICATION_CONFIGURATION>

$capability_instructions

<BATCH_CANDIDATES>
$batch_candidates
</BATCH_CANDIDATES>

<DOCUMENT_SUMMARIES>
$document_summaries
</DOCUMENT_SUMMARIES>
