from typing import Optional
from pydantic import BaseModel, Field
from .enums import RetrievalRelevance, UserIntent, GenerationSource

class RouterOutput(BaseModel):
    intent: str
    confidence: float = Field(ge=0, le=1, default=1.0)
    reasoning: Optional[str] = ""

class ExplainationResponce(BaseModel):
    explaination: str
    complexity_analysis: Optional[str] = None
    best_practices: list[str] = Field(default_factory=list)

class GenerationResponce(BaseModel):
    code: str
    explaination: ExplainationResponce
    language: str = "python"
    dependencies: list[str] = Field(default_factory=list)

class GeneratedSolution(GenerationResponce):
    source: GenerationSource

class RetrievedDocument(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    score: float = 0.0

class EvaluatorOutput(BaseModel):
    retrieval_relevance: RetrievalRelevance
    reasoning: str


class ExecutionResult(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0


class PipelineTrace(BaseModel):
    query: str
    intent: UserIntent
    router_confidence: float
    retrieval_used: bool = False
    retrieval_verdict: Optional[RetrievalRelevance] = None
    retrieval_reasoning: Optional[str] = None
    generation_source: Optional[GenerationSource] = None
    ingested: bool = False
    latency_ms: float = 0.0
    timestamp: str
