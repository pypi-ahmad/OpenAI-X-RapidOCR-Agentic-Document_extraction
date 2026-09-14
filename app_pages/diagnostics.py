"""Runtime, engine, and extraction diagnostics.

Responsible for: displaying engine readiness (`SETTINGS.openai_configured`,
RapidOCR version, PP-DocLayoutV3 runtime), processing warnings, failed pages,
and wall-clock stage bottleneck analysis for the current session.

Must not: execute parsing, trigger LLM calls, or mutate session state.

Next: `streamlit_app.py` to trigger document parsing.
"""

from collections import Counter

import streamlit as st

from agentic_extractor.config import SETTINGS
from agentic_extractor.layout import pp_doclayout_runtime_installed
from agentic_extractor.ocr import rapidocr_package_version
from agentic_extractor.timing import stage_timing_summary

st.title(":material/monitoring: Diagnostics")
st.caption("Engine readiness and processing warnings for the current session.")

st.subheader("Engine status")
if SETTINGS.openai_configured:
    st.success("OpenAI API configured · gpt-5.6-luna")
else:
    st.error(
        "OpenAI is not configured. Add OPENAI_API_KEY to the launcher environment, "
        "restart the app, and retry."
    )

rapidocr_version = rapidocr_package_version()
if rapidocr_version:
    st.success(f"RapidOCR installed · {rapidocr_version}")
else:
    st.error("RapidOCR is unavailable. Run 'uv sync --all-groups' and verify ONNX Runtime setup.")

if pp_doclayout_runtime_installed():
    st.success("PP-DocLayoutV3 runtime installed · GPU preferred, CPU fallback")
else:
    st.error(
        "PP-DocLayoutV3 is unavailable. Run "
        "'uv sync --project tools/pp_doclayout --locked' and restart the app."
    )

result = st.session_state.get("result")
if result is None:
    st.info("No processing diagnostics yet. Process a document on the Parse page first.")
    st.stop()

st.subheader("Processing warnings")
warnings = list(getattr(result, "warnings", []))
if not warnings:
    st.success("No processing warnings were recorded.")
else:
    counts = Counter(warnings)
    for warning in dict.fromkeys(warnings):
        repeats = counts[warning]
        suffix = f" (recorded {repeats} times)" if repeats > 1 else ""
        st.warning(f"{warning}{suffix}")

failed_pages = list(getattr(result, "failed_pages", []))
if failed_pages:
    st.subheader("Failed pages")
    st.error(", ".join(str(page) for page in failed_pages))

timing_analysis = stage_timing_summary(getattr(result, "timings", {}))
bottleneck = timing_analysis["bottleneck"]
st.subheader("Stage timing")
if bottleneck is None:
    st.info("No per-stage timing is available for this result.")
else:
    timing_row = st.container(horizontal=True, border=True)
    timing_row.metric("Slowest stage", bottleneck["stage"])
    timing_row.metric("Stage time", f"{bottleneck['seconds']:.3f}s")
    timing_row.metric("Measured share", f"{bottleneck['share_percent']:.1f}%")
    st.dataframe(timing_analysis["stages"], width="stretch", hide_index=True)

with st.expander("Runtime details"):
    engine = getattr(result, "engine", None)
    st.json(
        {
            "engine": (
                {
                    "name": engine.name,
                    "version": engine.version,
                    "device": engine.device,
                    "model_metadata": engine.model_metadata,
                }
                if engine is not None
                else None
            ),
            "timings": getattr(result, "timings", {}),
        }
    )
