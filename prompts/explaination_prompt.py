EXPLANATION_SYSTEM_PROMPT = """You are an expert software engineer explaining code to another
developer. Given code (and/or a question about it), produce:
- explanation: a clear, structured, line-by-line or section-by-section walkthrough
- complexity_analysis: time/space complexity discussion, if applicable
- best_practices: a short list of relevant best-practice notes or potential issues

Respond only via the required structured output."""

def build_explaination_user_prompt(query:str, code:str | None):
    parts = [f"User question: \n{query}"]
    if code:
        parts.append(f"\nCode:\n```\n{code}\n```")
    return "\n".join(parts)