"""Versioned prompt resources for the OpenAI document boundary.

This package holds only packaged Markdown prompt text (policy, task, and
context templates), never executable code or f-string-built instructions —
that separation is what lets each prompt's `prompt-version` and content hash
be recorded in usage telemetry. Loading, rendering, and hashing live in
`prompt_resources.py`, not here."""
