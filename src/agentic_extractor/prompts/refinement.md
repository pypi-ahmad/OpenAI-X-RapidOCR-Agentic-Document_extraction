<!-- prompt-version: 3 -->

# Document refinement

Review every page in `DOCUMENT_EVIDENCE` exactly once and list those page numbers in
`reviewed_pages`. Preserve page boundaries and source meaning.

$capability_instructions

$checkbox_instructions

<APPLICATION_CONFIGURATION>
Capabilities: $capabilities
Allowed classification labels: $allowed_classes
Extraction JSON Schema: $extraction_schema
</APPLICATION_CONFIGURATION>

For `refined_markdown`, produce source-faithful Markdown in page order. Apply only supported
corrections and structure; do not silently add content. The application will independently rebuild
and validate the canonical result from grounded proposals.

<DOCUMENT_EVIDENCE>
$document_context
</DOCUMENT_EVIDENCE>
