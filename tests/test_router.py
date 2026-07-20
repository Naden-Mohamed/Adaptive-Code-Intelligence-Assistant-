from core.router import TaskRouter
from core.schemas import UserIntent, RouterOutput

def test_router_return_llm_result_when_confident(mock_llm_client):
    llm_client = mock_llm_client(structured_response = RouterOutput(intent=UserIntent.EXPLANATION, confidence=0.8, reasoning=" "))
    router = TaskRouter(llm_client=llm_client)
    result = router.route(query=" explain the recursion to me",code= "")
    assert result.intent == UserIntent.EXPLANATION
    assert llm_client.calls # confirms the LLM path was actually exercised

def test_router_uses_heuristic_fallback_with_low_confidence(mock_llm_client):
    llm_client = mock_llm_client(structured_response = RouterOutput(intent=UserIntent.EXPLANATION, confidence=0.1, reasoning=" "))
    router = TaskRouter(llm_client=llm_client)
    result = router.route(query="generate a function to get duplicates of 2",code= "")
    assert result.intent == UserIntent.GENERATION

def test_router_fallback_when_llm_raises_error(mock_llm_client):
    llm_client = mock_llm_client(raise_error = True)
    router = TaskRouter(llm_client=llm_client)
    result = router.route(query="generate a function to get duplicates of 2",code= "")
    assert result.intent == UserIntent.GENERATION

def test_router_heuristic_fallback():
    result = TaskRouter._heuristic_fallback(query="generate a function to get duplicates of 2", code="")
    assert result.intent == UserIntent.GENERATION
