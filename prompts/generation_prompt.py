GENERATION_SYSTEM_PROMPT = """You are an expert software engineer. Produce a correct, idiomatic,
well-documented solution to the user's request. Always return:
- code: a complete, runnable solution
- explanation: what it does, the algorithm/approach, complexity, and any libraries used
- language: the programming language of the solution
- dependencies: any third-party packages required (empty list if none)

Respond only via the required structured output."""

GENERATION_WITH_CONTEXT_PREFIX = """The following reference material was retrieved from a
knowledge base and judged relevant to this request. Use it as guidance where appropriate,
but ensure the final solution correctly and completely addresses the user's request even
if the reference material is incomplete.

Reference material:
{context}

"""

def build_generation_user_prompt(query: str, context: str | None = None):
    if context:
        return GENERATION_WITH_CONTEXT_PREFIX.format(context=context) + f"User query:\n{query}"
    return f"User query:\n {query}"