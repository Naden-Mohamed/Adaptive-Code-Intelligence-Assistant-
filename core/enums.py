from enum import Enum

class UserIntent(str, Enum):
    EXPLANATION = "explanation"
    GENERATION = "generation"

class RetrievalRelevance(str, Enum):
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"

class GenerationSource(str, Enum):
    RAG = "rag"
    LLM_ONLY = "llm_only"
