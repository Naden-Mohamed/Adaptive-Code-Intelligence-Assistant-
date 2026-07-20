from unittest.mock import MagicMock
from core.ingestion import SelfLearningIngestor
from core.schemas import GeneratedSolution, GenerationSource,ExplainationResponce, RetrievedDocument

def _make_solution():
    return GeneratedSolution(
        code="def add(a, b): return a + b",
        explaination=ExplainationResponce(explaination="Adds two numbers."),
        language="python",
        dependencies=[],
        source=GenerationSource.LLM_ONLY,
    )

def test_ingest_skips_near_duplicate():
    retriever = MagicMock()
    retriever.search.return_value = [RetrievedDocument(content="Adds two numbers.", metadata={},score = 0.9)]
    ingestor = SelfLearningIngestor(retreiver=retriever)
    inserted = ingestor.ingest("Add 2 values", _make_solution())
    assert inserted is False
    retriever.insert.assert_not_called()

def test_ingest_inserts_new_content():
    retriever = MagicMock()
    retriever.search.return_value = [RetrievedDocument(content="unrelated", metadata={},score = 0.2)]
    ingestor = SelfLearningIngestor(retreiver=retriever)
    inserted = ingestor.ingest("subtract 4 from 8", _make_solution())
    assert inserted is True
    retriever.insert.assert_called_once()

def test_ingest_inserts_when_store_empty():
    retriever = MagicMock()
    retriever.search.return_value = []
    ingestor = SelfLearningIngestor(retreiver=retriever)
    inserted = ingestor.ingest("subtract 4 from 8", _make_solution())
    assert inserted is True

