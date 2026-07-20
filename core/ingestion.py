import logging
import hashlib
from .retriever import DataRetrieval
from .schemas import GenerationResponce, GeneratedSolution
from .config import get_settings
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class SelfLearningIngestor:
    def __init__(self, retreiver) -> None:
        self.settings = get_settings()
        self.retreiver = retreiver or DataRetrieval()
        self.similarity_threshold = self.settings.similarity_threshold

    def ingest(self, query: str, solution: GeneratedSolution):
        document_text = self._format_document(query=query, solution=solution)
        is_near_duplicate = self._is_near_duplicate(query=query)
        if is_near_duplicate:
            logger.info("Skipping ingestion: near-duplicate entry already present in the knowledge base.")
            return False
        metadatas = [
            {
                "source": "generated",
                "language": solution.language,
                "topic": query[:120],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # "chunk_index": i,
                "doc_hash": self._hash(document_text),
            }
        
        ]
        self.retreiver.insert(texts=[document_text], metadatas=metadatas)
    def _is_near_duplicate(self, query:str):
        result = self.retreiver.search(query=query, k=self.settings.top_k)

        if not result:
            return False
        
        return result[0].score >= self.settings.similarity_threshold

    @staticmethod
    def _format_document(query: str, solution: GenerationResponce) -> str:
        return (
            f"# Query\n{query}\n\n"
            f"# Code ({solution.language})\n{solution.code}\n\n"
            f"# Explanation\n{solution.explaination}"
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
