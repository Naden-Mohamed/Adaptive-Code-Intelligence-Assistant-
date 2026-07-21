from __future__ import annotations
import streamlit as st

from UI.components import render_explanation, render_solution, render_trace
from core.executor import run_code
from core.pipeline import AdaptiveCodeAssistantPipeline
from core.schemas import ExplainationResponce, GeneratedSolution

st.set_page_config(page_title="Adaptive Code Intelligence Assistant", page_icon="🧠", layout="wide")


@st.cache_resource
def get_pipeline() -> AdaptiveCodeAssistantPipeline:
    return AdaptiveCodeAssistantPipeline()


def main() -> None:
    st.title("Adaptive Code Intelligence Assistant")
    st.caption(
        "Routes between code explanation and RAG-backed code generation, "
        "and learns from every new solution it produces."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Upload code (optional)")
        uploaded = st.file_uploader(
            "Attach a source file to ask about",
            type=["py", "js", "ts", "java", "go", "cpp", "c"],
        )
        code_input = uploaded.read().decode("utf-8") if uploaded else None
        if code_input:
            st.code(code_input, language="python")
        st.divider()
        st.caption("Knowledge base grows automatically from novel solutions the assistant generates.")

    query = st.chat_input("Ask a coding question, or request new code...")

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    if query:
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pipeline = get_pipeline()
                result, trace = pipeline.run(query, code=code_input)

            render_trace(trace)

            if isinstance(result, ExplainationResponce):
                render_explanation(result)
                st.session_state.history.append({"role": "assistant", "content": result.explaination})
            elif isinstance(result, GeneratedSolution):
                render_solution(result)
                st.session_state.history.append({"role": "assistant", "content": result.explaination})

                if st.button("▶ Run Code", key=f"run_{len(st.session_state.history)}"):
                    with st.spinner("Executing in sandbox..."):
                        exec_result = run_code(result.code, result.language)
                    st.subheader("Execution Result")
                    st.write(f"Success: {exec_result.success} · {exec_result.execution_time_ms:.0f} ms")
                    if exec_result.stdout:
                        st.code(exec_result.stdout, language="text")
                    if exec_result.stderr:
                        st.error(exec_result.stderr)


if __name__ == "__main__":
    main()
