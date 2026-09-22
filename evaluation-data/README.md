# Table and checkbox evaluation data

This directory contains versioned object labels for review against the local document corpus.
Model output is not human ground truth.

## Current review pack

`masked-amerigroup-pages-1-2.review.json` covers pages 1 and 2 of the matching source PDF, identified
by SHA-256 rather than by path. It contains:

- 3 page-2 table candidates with normalized boxes and expected grid counts;
- 25 visible page-2 checkbox candidates with normalized boxes, labels, groups, and states;
- 4 LandingAI checkbox-marker candidates rejected because the source image contains no control;
- explicit provenance and review status.

The candidate file is `pending_human_review`. An AI assistant localized the objects and used the supplied LandingAI result to seed table structure. It is not an independently human-labeled gold set.

## Human adjudication

1. Verify the source SHA-256 matches the dataset.
2. Inspect page 2 at sufficient resolution. Review every box, label, state, table boundary, row
   count, column count, and rejected candidate.
3. Correct the candidate values where needed.
4. Change every reviewed object's `review_status` to `human_approved`.
5. Set `approval.status` to `human_approved`, record the reviewer's name, and add an ISO-8601
   `reviewed_at` timestamp.
6. Run the strict validation command below.

```powershell
uv run python -c "from pathlib import Path; from agentic_extractor.evaluation_data import load_evaluation_dataset; load_evaluation_dataset(Path('evaluation-data/masked-amerigroup-pages-1-2.review.json'))"
```

Strict loading fails until the dataset and every annotation have explicit human approval. Tests
and exploratory tools may load the draft with `require_human_approval=False`, but production
accuracy claims must use the strict default.

## Coordinate and privacy contract

Boxes use normalized `[left, top, right, bottom]` coordinates in a top-left origin coordinate
space. The tracked labels omit table cell contents and personal document values. They describe
layout structure and generic checkbox labels only.

