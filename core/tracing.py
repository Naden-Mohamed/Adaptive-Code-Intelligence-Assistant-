import json
import logging
from pathlib import Path

from core.config import get_settings
from core.schemas import PipelineTrace

logger = logging.getLogger(__name__)


def write_trace(trace: PipelineTrace) -> None:
    """
    Appends a structured JSON trace of the pipeline run to a log file.
    This is the raw material for both debugging and the offline/online
    evaluation reports -- every routing decision, retrieval verdict, and
    generation source is recorded, not just the final answer.
    """
    settings = get_settings()
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline_trace.jsonl"
    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace.model_dump(mode="json")) + "\n")
    except OSError as exc:
        logger.warning("Failed to write trace log: %s", exc)
