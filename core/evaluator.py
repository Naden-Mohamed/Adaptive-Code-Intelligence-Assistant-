
import logging

from core.config import get_settings
from core.llm_client import LLMClient
from core.schemas import EvaluatorOutput, RetrievedDocument
from .enums import RetrievalRelevance
from prompts.evaluator_prompt import EVALUATOR_SYSTEM_PROMPT,build_evaluator_user_prompt

logger = logging.getLogger(__name__)

class Evaluator:
    """LLM-as-judge that decides whether retrieved documents are usable context."""
    def __init__(self) -> None:
        settings = get_settings()
        self.llm_client = LLMClient(model_name=settings.model_id, temperature=settings.temperature)

    def evaluate(self,query:str, documents: list[RetrievedDocument]) -> EvaluatorOutput:
        if not documents:
            return EvaluatorOutput(retrieval_relevance= RetrievalRelevance.NOT_RELEVANT, reasoning="No documents retrieved.")
        try:
            user_prompt = build_evaluator_user_prompt(query=query, documents=[d.content for d in documents])
            return self.llm_client.structured_call(
                user_prompt=user_prompt,
                system_prompt=EVALUATOR_SYSTEM_PROMPT,
                output_model=EvaluatorOutput
            )
        except Exception as exc:
            logger.warning("Evaluator LLM call failed (%s); defaulting to not_relevant", exc)
            return  EvaluatorOutput(retrieval_relevance= RetrievalRelevance.NOT_RELEVANT, reasoning=f"Evaluator unavailable: {exc}")