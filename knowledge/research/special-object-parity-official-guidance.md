# Official guidance for checkbox, logo, and attestation parity

Research date: 2026-08-31

Scope: first-party sources only. The recommendations below combine documented OpenCV primitives, the published PP-DocLayoutV3 label contract, and OpenAI vision constraints. No source claims to provide a turnkey checkbox, logo, or signature detector; those compositions must be validated against this project's human-labeled examples.

## Conclusions

1. **PP-DocLayoutV3 does not have `checkbox`, `logo`, `signature`, or `attestation` classes.** Its published inference configuration has 25 labels, including `image`, `header_image`, `footer_image`, `seal`, `table`, and text/layout classes. A detection under `image` is not evidence that the region is a logo, and `seal` is not a signature. [PP-DocLayoutV3 inference configuration](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/blob/main/inference.yml)
2. **Checkboxes need a dedicated local candidate detector.** OpenCV documents the required image primitives—adaptive thresholding for uneven illumination, morphology, contour hierarchy, polygon approximation, and connected-component statistics—but not a universal checkbox classifier. [Thresholding](https://docs.opencv.org/4.13.0/d7/d4d/tutorial_py_thresholding.html) [Morphology](https://docs.opencv.org/4.13.0/d9/d61/tutorial_py_morphological_ops.html) [Shape analysis](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html)
3. **Luna cannot be the sole detector or counter.** OpenAI states that vision models can struggle with small text, rotation, precise spatial localization, and counting, and may produce incorrect descriptions. [Vision limitations](https://developers.openai.com/api/docs/guides/images-vision#limitations)
4. **Use Luna as a candidate verifier on a high-resolution crop.** OpenAI recommends `detail: "original"` for OCR, small-object, dense, or coordinate-sensitive work when supported. If an image is resized, returned coordinates must be mapped back to the original coordinate space. [Image detail levels](https://developers.openai.com/api/docs/guides/images-vision#choose-an-image-detail-level)
5. **Publish only evidence-agreed objects.** A checkbox state, logo identity, or attestation statement must reference a local region and supporting source text. Structured output guarantees response shape, not factual grounding; application code must validate candidate IDs, pages, coordinates, and source IDs. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

## Checkbox detection

### What official image-processing sources support

- OpenCV recommends adaptive thresholding when illumination varies across an image; each pixel's threshold is calculated from a local neighborhood. This is relevant to scans with shadows or uneven backgrounds. [Image thresholding](https://docs.opencv.org/4.13.0/d7/d4d/tutorial_py_thresholding.html)
- Morphological opening removes small foreground noise, while closing fills small holes. The kernel shape and size determine what is preserved, so they must be tuned to the target scan scale. [Morphological transformations](https://docs.opencv.org/4.13.0/d9/d61/tutorial_py_morphological_ops.html)
- `findContours` operates on a binary image. `RETR_TREE` reconstructs the full hierarchy of nested contours, and `approxPolyDP` approximates a curve or polygon. These primitives can identify a box border and its inner region. [Structural analysis and shape descriptors](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html)
- `connectedComponentsWithStats` returns component statistics and centroids for a binary image. These support deterministic size, area, position, and aspect-ratio filters. [Connected components](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f)

### Exact project implications

1. Run checkbox candidate detection on the rendered page pixels, not OCR characters or layout boxes. Keep both a global/Otsu binary view and an adaptive-threshold view; accept a candidate for review only when its geometry survives the scan-appropriate view.
2. Require checkbox-like geometry before semantic review: a compact near-square component or contour, a plausible border, bounded size relative to page DPI, and a stable page-coordinate box. Use contour hierarchy as supporting evidence for bordered boxes; do not require it for filled controls whose inner mark merges with the border.
3. Compute mark evidence only inside an inset region so the printed border does not make every empty box look filled. Preserve the raw crop, thresholded crop, contour/component measurements, and occupancy result as immutable evidence.
4. Associate the candidate with nearby RapidOCR text by spatial proximity and reading order. The OCR text labels the control; it must not create a checkbox candidate by itself. This prevents ordinary words, doses, identifiers, and page numbers from becoming checkbox facts.
5. Send Luna one candidate crop or a small non-overlapping group. Include the checkbox, its immediate label, the candidate ID, page number, original-page coordinates, and compact OCR source records. Use `detail: "original"` for these small-object crops, while retaining normal page-level `high` review elsewhere.
6. Require a structured state of `checked`, `unchecked`, or `uncertain`. Accept `checked`/`unchecked` only when local pixel evidence, nearby OCR grounding, and Luna agree. Publish `uncertain` nowhere in canonical document content; retain it in review diagnostics.
7. Deduplicate overlapping candidates before Luna using deterministic geometry. Validate that every Luna result names exactly one supplied candidate and cannot introduce new coordinates, labels, or checkboxes.

The thresholds for physical size, square tolerance, border strength, inner occupancy, proximity, and agreement are project parameters—not values supplied by OpenCV. Calibrate them from human-labeled positive and negative examples at representative DPI values.

## PP-DocLayoutV3 image and logo handling

### Published contract and limitation

The official `inference.yml` lists these relevant visual labels: `chart`, `footer_image`, `header_image`, `image`, `seal`, and `table`. It does not list `logo`. The model preprocesses to 800×800 and its default draw threshold is 0.5. [PP-DocLayoutV3 inference configuration](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/blob/main/inference.yml)

Paddle's layout module returns a class ID, label, confidence score, and box coordinates, and supports class-specific confidence thresholds plus NMS and bounding-box merge modes. [PaddleOCR layout detection](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/module_usage/layout_detection.en.md)

### Exact project implications

- Persist `image`, `header_image`, and `footer_image` as their actual detected classes. Do not rename them to `logo` without separate evidence.
- A logo candidate may originate from an image-class region near a page header/footer, supported by nearby OCR brand text. Luna may describe or confirm the bounded candidate crop, but the app must abstain from a brand identity when neither pixels nor OCR support it.
- Keep class-specific thresholds. An image-region threshold should be calibrated separately from text/table thresholds; one global threshold cannot express different false-positive costs.
- Apply layout NMS/merge rules before constructing logo/image candidates so nested or overlapping visual regions do not create duplicate semantic objects.
- Because PP-DocLayoutV3 resizes input to 800×800, retain the transform and map detections back to source-page coordinates before attaching OCR source IDs or sending crops.

## Signature and attestation handling

PP-DocLayoutV3 provides no signature or attestation class. Therefore:

- Detect an attestation primarily from grounded text phrases and their layout region. A nearby visual mark may support presence of a signature, but it cannot prove signer identity or intent.
- Treat `seal` as a seal only. Do not map it to `signature`.
- Crop the smallest region that contains the attestation text, signature line/mark, and necessary label context. Large page crops increase unrelated evidence and make object association less reliable.
- Ask Luna to classify only the supplied object: attestation text present/absent, signature-like mark present/absent/uncertain, and exact supporting source IDs. Do not ask Luna to infer a person's identity from handwriting.
- Persist uncertain signature-like marks for review and keep them out of verified Markdown statements.

## OpenAI vision constraints for special-object review

Official behavior:

- `gpt-5.6-luna` supports `low`, `high`, `original`, and `auto`. `high` fits within 2048×2048 and 2,500 patches; `original` supports dense/spatially sensitive images without the `high` patch budget. [Model sizing behavior](https://developers.openai.com/api/docs/guides/images-vision#model-sizing-behavior)
- OpenAI explicitly recommends `original` for OCR, small-object detection, or precise coordinates. Original inputs may still be resized at model limits, so coordinate mapping remains the caller's responsibility. [Choose an image detail level](https://developers.openai.com/api/docs/guides/images-vision#choose-an-image-detail-level)
- Vision limitations include small text, rotated content, variable line styles, precise spatial localization, approximate object counts, and incorrect descriptions. [Vision limitations](https://developers.openai.com/api/docs/guides/images-vision#limitations)

Exact project implications:

- Keep every page's required `high`-detail semantic review, then add `original` only for bounded checkbox/logo/attestation crops that remain unresolved. Do not resend a full page at `original` merely because one small object is uncertain.
- Give each crop a stable candidate ID and original-page box. Luna returns the ID and classification, not replacement coordinates. This avoids relying on a known spatial-localization weakness.
- Never use Luna's page-level count as the canonical checkbox count. The local candidate index defines cardinality; Luna verifies supplied candidates.
- Rotate/deskew the crop for review when local orientation evidence supports the transform, but retain and cite the original geometry.
- Limit the prompt to the object definition, allowed states, supplied evidence, and abstention rule. Document text remains untrusted content and cannot create new candidates or change routing.

## Acceptance evidence to collect

- Checkbox precision, recall, and state accuracy by page and DPI, with false positives categorized by text, punctuation, table cells, logos, page furniture, and scan artifacts.
- Candidate-stage counts: threshold components → square/border candidates → OCR-associated candidates → Luna-reviewed candidates → published controls.
- Logo/image region precision, including whether a published logo identity had both a bounded visual region and supporting OCR/vision evidence.
- Attestation-text recall and signature-like-mark precision, while separately tracking abstentions.
- Per-object crop dimensions, detail level, input tokens, latency, and cache usage. Compare these with the former full-page repair path.
- Invariants: no unsupported candidate IDs, no out-of-page coordinates, no review-required checkbox in canonical Markdown, and no `image`/`seal` automatically promoted to `logo`/`signature`.

## Primary sources

- PaddlePaddle, [PP-DocLayoutV3 model card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
- PaddlePaddle, [PP-DocLayoutV3 inference configuration and label list](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/blob/main/inference.yml)
- PaddleOCR, [Layout detection module](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/module_usage/layout_detection.en.md)
- OpenAI, [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)
- OpenAI, [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- OpenCV, [Image thresholding](https://docs.opencv.org/4.13.0/d7/d4d/tutorial_py_thresholding.html)
- OpenCV, [Morphological transformations](https://docs.opencv.org/4.13.0/d9/d61/tutorial_py_morphological_ops.html)
- OpenCV, [Structural analysis and shape descriptors](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html)
