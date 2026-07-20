import pytest
from pydantic import ValidationError
from ..core.schemas import GeneratedSolution,GenerationResponce, RouterOutput,ExplainationResponce
from ..core.enums import UserIntent, GenerationSource
def test_router_output_valid():
    router_output = RouterOutput(intent= UserIntent.GENERATION, confidence=0.9)
    assert router_output == UserIntent.GENERATION

def test_router_confidence_bound():
    with pytest.raises(ValidationError):
        RouterOutput(intent= UserIntent.GENERATION, confidence=1.9)

def test_generated_response_wraps_generated_code():
    response = GenerationResponce(code="print(1)", explaination=ExplainationResponce(explaination="prints 1"), language="python", dependencies=[])
    solution = GeneratedSolution(**response.model_dump(), source=GenerationSource.RAG)
    assert solution.source == GenerationSource.RAG
    assert solution.code == "print(1)"

