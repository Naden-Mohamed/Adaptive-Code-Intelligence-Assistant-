import os
from pydantic import BaseModel
import pytest

from core.config import get_settings

get_settings.cache_clear()

class MockLLMClient():
    def __init__(self, structured_response: BaseModel, raise_error: bool = False) -> None:
        self.structured_response =structured_response
        self.raise_error = raise_error
        self.calls = []

    def structured_call(self, system_prompt, user_prompt, output_model):
        self.calls.append((system_prompt, user_prompt, output_model))
        if self.raise_error:
            raise RuntimeError("Simulated LLM failure")
        return self.structured_call

@pytest.fixture
def mock_llm_client():
    def _factory(**kwargs):
        return MockLLMClient(**kwargs)
    return _factory