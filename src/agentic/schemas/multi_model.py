from typing import Annotated, TypedDict, Dict, List, Optional, Any
from pydantic import BaseModel, Field
from agentic.schemas.output_schema_cv import OutputSchemaCV
from agentic.schemas.output_schema_cover_letter import OutputSchemaCoverLetter
from agentic.schemas.output_schema_cv_critic import CVCriticFeedback
import utilities.global_const_var as gcv

def merge_model_runs(existing: dict, incoming: dict) -> dict:
    """
    Thread-safe dictionary reducer for LangGraph.
    Merges independent parallel LLM channel updates without data overwrites.
    """
    # Initialize state if empty
    if existing is None:
        existing = {}
    if incoming is None:
        return existing
        
    # Merge per model key, so a partial workspace update does not wipe the other fields
    # (e.g. target_language, company_research) previously written for the same model
    merged = dict(existing)
    for model_key, workspace in incoming.items():
        if isinstance(workspace, dict) and isinstance(merged.get(model_key), dict):
            merged[model_key] = {**merged[model_key], **workspace}
        else:
            merged[model_key] = workspace
    return merged

class ReflectionIterationCV(TypedDict):
    """Tracks a single pass through a drafting/criticism loop."""
    draft_cv: OutputSchemaCV
    feedback_cv: CVCriticFeedback
    approved_cv: bool
    no_more_iterations: bool
    skip_cover_letter: bool

class ReflectionIterationCoverLetter(TypedDict):
    """Tracks a single pass through a drafting/criticism loop."""
    draft_cover_letter: OutputSchemaCoverLetter
    feedback_cover_letter: str
    approved_cover_letter: bool
    no_more_iterations: bool

class ModelPipelineWorkspace(TypedDict, total=False):
    """Isolated runtime memory for a specific LLM channel (e.g. OpenAI, Anthropic, Ollama)."""
    target_language: str
    company_research: str
    labels: List

    # Company research structured data
    is_fake_company: bool
    is_headhunter: bool
    employment_type: str  # permanent / freelance_temporary / unknown

    # Workspace
    tmp_cv: OutputSchemaCV
    tmp_cover_letter: OutputSchemaCoverLetter
    
    # Chronological history of the loops
    cv_loop: List[ReflectionIterationCV]
    cover_letter_loop: List[ReflectionIterationCoverLetter]
    
    # Final production copies outputted by this model channel
    final_compiled_cv: OutputSchemaCV
    final_compiled_cv_txt: str
    final_compiled_cover_letter: OutputSchemaCoverLetter

    error_log: Optional[str]