import logging
import time
from datetime import datetime, timezone
from typing import Union

from .evaluator import Evaluator
from .generator import Generator
from .ingestion import SelfLearningIngestor
from .retriever import DataRetrieval
from .router import TaskRouter

from .schemas import (
    ExplainationResponce,
    GeneratedSolution,
    GenerationSource,
    UserIntent,
    PipelineTrace,
    RetrievalRelevance,
)
from .tracing import write_trace

logger = logging.getLogger(__name__)

PipelineResult = Union[GeneratedSolution, ExplainationResponce]

class AdaptiveCodeAssistantPipeline:
    """
    Orchestrates the full request lifecycle: routing, retrieval, evaluation,
    generation, and self-learning ingestion. This is the single entry point
    the UI and the evaluation harness should call -- neither should talk to
    router/retriever/evaluator/generator directly.
    """

    def __init__(self,
                router: TaskRouter | None = None, evaluator: Evaluator | None = None,
                generator: Generator| None = None, retriever: DataRetrieval| None = None,
                ingestor: SelfLearningIngestor | None = None) -> None:
        self.router = router or TaskRouter()
        self.evaluator = evaluator or Evaluator()
        self.generator = generator or Generator()
        self.retriever = retriever or DataRetrieval()
        self.ingestor = ingestor or SelfLearningIngestor()

    def run(self, query: str, code: str | None = None) -> tuple[PipelineResult, PipelineTrace]:
        start = time.perf_counter()
        router_output = self.router.route(query, code)

        trace = PipelineTrace(
            query=query,
            intent=UserIntent(router_output.intent),
            router_confidence=router_output.confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if router_output.intent == UserIntent.EXPLANATION:
            result = self.generator.explain(query, code)
            trace.latency_ms = (time.perf_counter() - start) * 1000
            write_trace(trace)
            return result, trace

        result, trace = self._run_generation(query, trace)
        trace.latency_ms = (time.perf_counter() - start) * 1000
        write_trace(trace)
        return result, trace
    def _run_generation(self, query: str, trace: PipelineTrace) -> tuple[GeneratedSolution, PipelineTrace]:
        documents = self.retriever.search(query, 3)
        trace.retrieval_used = bool(documents)

        evaluator_output = self.evaluator.evaluate(query=query, documents=documents)
        trace.retrieval_verdict = evaluator_output.retrieval_relevance
        trace.retrieval_reasoning = evaluator_output.reasoning

        if evaluator_output.retrieval_relevance == RetrievalRelevance.RELEVANT:
            solution = self.generator.generate_with_context(query, documents)
        else:
            solution = self.generator.generate_without_context(query)

        trace.generation_source = solution.source

        if solution.source == GenerationSource.LLM_ONLY:
            trace.ingested = bool(self.ingestor.ingest(query, solution))

        return solution, trace