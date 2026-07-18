from prompts.explaination_prompt import EXPLANATION_SYSTEM_PROMPT, build_explaination_user_prompt
from prompts.generation_prompt import GENERATION_SYSTEM_PROMPT, build_generation_user_prompt
from core.schemas import ExplainationResponce, GenerationResponce, GenerationSource, RetrievedDocument, GeneratedSolution
from .enums import GenerationSource
from .llm_client import LLMClient
from .config import get_settings

class Generator:
    def __init__(self) -> None:
        settings = get_settings()
        self.llm_client = LLMClient(model_name=settings.model_id, temperature=settings.temperature)

    def generate_with_context(self, query:str, documents: list[RetrievedDocument])->GeneratedSolution:
        context = "\n\n".join(d.content for d in documents)
        user_prompt = build_generation_user_prompt(query=query, context=context)
        responce = self._generate(user_prompt=user_prompt)
        return GeneratedSolution(**responce.model_dump(), source = GenerationSource.RAG)

    def generate_without_context(self, query:str)->GeneratedSolution:
        user_prompt = build_generation_user_prompt(query=query)
        responce = self._generate(user_prompt=user_prompt)
        return GeneratedSolution(**responce.model_dump(), source = GenerationSource.LLM_ONLY)

    def _generate(self, user_prompt:str)-> GenerationResponce:
        return self.llm_client.structured_call(
            user_prompt=user_prompt,
            system_prompt=GENERATION_SYSTEM_PROMPT,
            output_model=GenerationResponce
        )
    def explain(self, query: str, code: str | None) -> ExplainationResponce:
        user_prompt = build_explaination_user_prompt(query=query,code=code)
        return self.llm_client.structured_call(
            user_prompt=user_prompt,
            system_prompt=EXPLANATION_SYSTEM_PROMPT,
            output_model=ExplainationResponce
        )