from __future__ import annotations

import streamlit as st

from core.schemas import ExplainationResponce, GeneratedSolution, PipelineTrace


def render_trace(trace: PipelineTrace) -> None:
    """Surfaces the pipeline's internal decisions -- route, retrieval verdict,
    generation source -- so the user (and reviewer) can see *why* an answer
    looks the way it does, not just the final output."""
    
    with st.expander("Routing & retrieval details", expanded=False):
        st.write(f"**Intent:** {trace.intent.value} (confidence {trace.router_confidence:.2f})")
        if trace.retrieval_used:
            verdict = trace.retrieval_verdict.value if trace.retrieval_verdict else "n/a"
            st.write(f"**Retrieval verdict:** {verdict}")
            st.write(f"**Reasoning:** {trace.retrieval_reasoning}")
        if trace.generation_source:
            st.write(f"**Generation source:** {trace.generation_source.value}")
        st.write(f"**Ingested into knowledge base:** {trace.ingested}")
        st.write(f"**Latency:** {trace.latency_ms:.0f} ms")


def render_solution(solution: GeneratedSolution) -> None:
    st.code(solution.code, language=solution.language)
    st.markdown(solution.explaination)
    if solution.dependencies:
        st.caption(f"Dependencies: {', '.join(solution.dependencies)}")


def render_explanation(explanation: ExplainationResponce) -> None:
    st.markdown(explanation.explaination)
    if explanation.complexity_analysis:
        st.markdown(f"**Complexity:** {explanation.complexity_analysis}")
    if explanation.best_practices:
        st.markdown("**Best practices / notes:**")
        for item in explanation.best_practices:
            st.markdown(f"- {item}")
