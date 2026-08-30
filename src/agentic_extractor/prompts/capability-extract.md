<!-- prompt-version: 2 -->

## Extract

- Propose exactly one item for each top-level field declared by the supplied JSON Schema and no
  undeclared fields.
- Use the schema type and constraints. Evidence must support the value itself, not merely the
  presence of a nearby label.
- Return scalar values using their JSON scalar type. For array or object fields, return the complete
  value as a compact JSON string; the application will parse and validate it against the schema.
- Set `status="verified"` only for an unambiguous, grounded value. Otherwise use
  `status="uncertain"`, set `value=null` when unsupported, and provide `abstention_reason`.
- Preserve the source representation in evidence quotes. Do not manufacture values to satisfy
  required fields, arithmetic, patterns, formats, or cross-field consistency.
