import logging
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from core.config import get_settings
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel) # Means T can be any class that inherits from BaseModel

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name 
        self.temperature = temperature
        self._model = None
    
    def _get_model(self):
        from groq import Groq
        settings = get_settings()
        if not self._model:
            self._model = Groq(
                api_key=settings.groq_api_key,
            )
        return self._model
    """
    Heavy dependencies are imported lazily inside
    _get_model() so that modules depending on LLMClient can be unit
    tested -- and the app can start up -- without the package installed
    until an actual LLM call is made.
    """
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def structured_call(self, user_prompt: str, system_prompt:str, output_model: Type[T]) -> T:
        settings = get_settings()
        model = self._get_model()
        try:
            result = model.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:  
            logger.warning("LLM structured call failed: %s", exc)
            raise

        if not isinstance(result, output_model):
            result = output_model.model_validate(result)

        return result