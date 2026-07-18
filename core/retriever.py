import logging
from langchain_chroma import Chroma
from .config import get_settings
from .schemas import RetrievedDocument
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class DataRetrieval():
    def __init__(self):
        self.settings = get_settings()
        self.persist_directory = self.settings.vector_db_path
        self.top_k = self.settings.top_k

    def _get_vectordb(self):
        embedding = HuggingFaceEmbeddings(model_name= self.settings.embedding_model, encode_kwargs={"normalize_embeddings": True},)
        return Chroma(
            embedding_function= embedding,
            persist_directory= self.persist_directory,
            collection_name="coding_assistant_db"
        )
    
    def search(self, query:str, k: int | None)->list[RetrievedDocument]:
        vector_db = self._get_vectordb()
        k = k if k else self.top_k

        try:
            results = vector_db.similarity_search_with_relevance_scores(query=query, k=k)
        except Exception as exc:
            logger.error(f"Search Operation didn't complete{exc}")
            return []
        return [
            RetrievedDocument(content=doc.page_content, metadata=doc.metadata, score=score)
            for doc, score in results
        ]
    
    def insert(self, texts: list[str], metadatas: list[dict]):
        vector_db = self._get_vectordb()
        vector_db.add_texts(texts=texts, metadatas=metadatas)
