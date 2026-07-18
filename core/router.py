
import logging

from core.config import get_settings
from core.llm_client import LLMClient
from core.schemas import UserIntent, RouterOutput
from prompts.router_prompt import ROUTER_SYSTEM_PROMPT, build_router_user_prompt

logger = logging.getLogger(__name__)

class TaskRouter:
    def __init__(self) -> None:
        settings = get_settings()
        self.llm_client = LLMClient(model_name=settings.model_id, temperature=settings.temperature)
        self.min_confidence = settings.router_min_confidence

    def route(self, query:str, code:str | None) -> RouterOutput:
        """
        Classifies a user request as 'explanation' or 'generation'.

        Falls back to a deterministic heuristic if the LLM call fails or returns
        low-confidence output, so the pipeline degrades gracefully instead of
        crashing (or silently misrouting) on a router failure.
        """
        try:
            router_prompt = build_router_user_prompt(query=query, code=code)
            result = self.llm_client.structured_call(
                user_prompt=router_prompt,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                output_model= RouterOutput
            )
            if result.confidence < self.min_confidence:
                logger.info(
                    "Router confidence %.2f below threshold %.2f; using heuristic fallback",
                    result.confidence,
                    self.min_confidence,
                )
                return self._heuristic_fallback(query, code)
            return result
        except Exception as exc:  
            logger.warning("Router LLM call failed (%s); using heuristic fallback", exc)
            return self._heuristic_fallback(query, code)                


    @staticmethod
    def _heuristic_fallback(query: str, code: str | None) -> RouterOutput:
        """
        Simple deterministic rule used only when the LLM router is unavailable
        or unconfident. Not a replacement for the LLM router's nuance -- just
        a safety net so the pipeline never hard-fails on routing.
        """
        explanation_keywords = ("explain", "what does", "why does", "how does", "review", "debug", "fix")
        generation_keywords = ("write", "generate", "create", "implement", "build")

        text = query.lower()
        if code and not any(k in text for k in generation_keywords):
            return RouterOutput(
                intent= UserIntent.EXPLANATION, confidence= 0.5 , reasoning="heuristic: code attached, no generation verb"
            )

        if any(k in text for k in explanation_keywords) and not any(k in text for k in generation_keywords):
            return RouterOutput(
                intent= UserIntent.EXPLANATION, confidence= 0.5 , reasoning="heuristic: explanation keyword match"
            )
        
        return RouterOutput(
                intent= UserIntent.GENERATION, confidence= 0.5 , reasoning="heuristic: default to generation"
            )
