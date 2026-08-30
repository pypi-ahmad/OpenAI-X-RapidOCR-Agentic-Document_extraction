<!-- prompt-version: 2 -->

$base_prompt

# Bounded repair

Return a complete structured result, but change only proposals directly implicated by
`VALIDATION_ISSUES`. Preserve valid prior proposals exactly. Do not weaken schemas, evidence rules,
confidence thresholds, checkbox rules, or routing. Do not add a missing value unless supplied
evidence supports it; abstain when the issue cannot be repaired from existing evidence.

<VALIDATION_ISSUES>
$validation_issues
</VALIDATION_ISSUES>

<PRIOR_DERIVED_PROPOSAL>
$prior_proposal
</PRIOR_DERIVED_PROPOSAL>
