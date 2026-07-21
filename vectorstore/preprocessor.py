from core.config import get_settings
import json
from core.retriever import DataRetrieval
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self):
        settings = get_settings()
        self.dataset_path = settings.dataset_path
        self.retrieval = DataRetrieval()
    def _read_file_content(self):
        if not self.dataset_path:
            logger.error("dataset path is not set")

        with open(self.dataset_path, "r", encoding="utf-8") as file:
            data = []
            for line in file:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _preprocess_file_content(self, data):
        if not data:
            logger.error("No data were found to preprocess")

        return [{
            "task_id": str(row["task_id"]),
            "prompt": str(row["prompt"]).strip(),
            "entry_point": str(row["entry_point"]).strip(),
            "solution":  str(row["canonical_solution"]).strip(),
            "test": str(row["test"]).strip()}
            for row in data
            ]

    def insert_into_vectordb(self, preprocessed_data: list[dict]):
        if not preprocessed_data:
            return "No data were found"
        
        for row in preprocessed_data:
            text_to_embed = f"Function: {row["entry_point"]}\nDescription and Examples:\n{row["prompt"]}"

            self.retrieval.insert(
                texts=[text_to_embed],
                metadatas=[{
                "entry_point": row["entry_point"],
                "prompt": row["prompt"],        # Kept as raw text for LLM reference
                "solution": row["solution"]     
            }]
            )

if __name__ == "__main__":
    preprocessor = Preprocessor()
    dataset = preprocessor._read_file_content()
    data = preprocessor._preprocess_file_content(data=dataset)
    preprocessor.insert_into_vectordb(data)

    print(data[0])