from unittest.mock import MagicMock
from core.evaluator import Evaluator
from core.schemas import RetrievedDocument, EvaluatorOutput
from core.enums import RetrievalRelevance

def test_router_when_no_documents(mock_llm_client):
    llm_client = mock_llm_client()
    evaluator = Evaluator(llm_client=llm_client)
    evaluation = evaluator.evaluate(query="implement coding agent", documents=[])
    assert evaluation.retrieval_relevance == RetrievalRelevance.NOT_RELEVANT
    assert not llm_client.calls

def test_router_with_error(mock_llm_client):
    llm_client = mock_llm_client(raise_error = True)
    evaluator = Evaluator(llm_client=llm_client)
    docs = [RetrievedDocument(content="some code", metadata={}, score=0.8)]
    evaluation = evaluator.evaluate(query="implement coding agent", documents=docs)
    assert evaluation.retrieval_relevance == RetrievalRelevance.NOT_RELEVANT

def test_router_with_relevant_docs(mock_llm_client):
    llm_client = mock_llm_client(
        structured_response = EvaluatorOutput(retrieval_relevance=RetrievalRelevance.RELEVANT, reasoning=" ")
        )
    evaluator = Evaluator(llm_client=llm_client)
    docs = [RetrievedDocument(content="some code", metadata={}, score=0.8)]
    evaluation = evaluator.evaluate(query="implement coding agent", documents=docs)
    assert evaluation.retrieval_relevance == RetrievalRelevance.RELEVANT
