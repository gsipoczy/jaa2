
from typing import Annotated, TypedDict, Dict, List, Optional, Any
from pydantic import BaseModel, Field

from agentic.schemas.input_schema import InputSchema
from agentic.schemas.multi_model import ModelPipelineWorkspace, merge_model_runs
from agentic.schemas.output_schema_cv import OutputSchemaCV
from agentic.schemas.output_schema_cover_letter import OutputSchemaCoverLetter

# Workflow State
class AppState(TypedDict):

    # INPUTS
    input_data: InputSchema
    
    # INDEPENDENT PARALLEL CHANNELS
    # Annotated with the custom reducer to handle concurrent writes safely
    model_runs: Annotated[Dict[str, ModelPipelineWorkspace], merge_model_runs]
    
    # FINAL SELECTION
    selected_best_cv: OutputSchemaCV
    selected_best_cover_letter: OutputSchemaCoverLetter
    judge_justification_notes: str
    labels: List

    # WORKFLOW ERRORS
    global_error: Optional[str]
