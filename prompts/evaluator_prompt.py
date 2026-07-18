
EVALUATOR_SYSTEM_PROMPT = """You are a retrieval quality evaluator for a code-generation RAG system.
Given a user's query and a set of retrieved documents, decide whether the documents are
sufficiently relevant and specific to help generate a correct solution.

Mark "relevant" only if at least one document meaningfully addresses the same problem,
pattern, or API as the query. Mark "not_relevant" if the documents are generic, off-topic,
or would not materially help generation.

Respond only via the required structured output, including a one-sentence justification."""


def build_evaluator_user_prompt(query: str, documents: list[str]) -> str:
    doc_block = "\n\n".join(f"[Doc {i + 1}]\n{doc}" for i, doc in enumerate(documents))
    return f"User query:\n{query}\n\nRetrieved documents:\n{doc_block if doc_block else '(none retrieved)'}"
