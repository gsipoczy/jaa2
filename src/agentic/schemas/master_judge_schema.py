from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional
from agentic.schemas.multi_model import ReflectionIterationCV, ReflectionIterationCoverLetter
from agentic.schemas.output_schema_cv import OutputSchemaCV
from agentic.schemas.output_schema_cover_letter import OutputSchemaCoverLetter

class ModelPipelineWorkspace(TypedDict, total=False):
    target_language: str
    company_research: str
    is_fake_company: bool
    is_headhunter: bool
    employment_type: str  
    
    cv_loop: List[ReflectionIterationCV]
    cover_letter_loop: List[ReflectionIterationCoverLetter]
    
    # 🔍 THE JUDGE EVALUATES THESE TWO TARGETS PER KEY:
    final_compiled_cv: OutputSchemaCV
    final_compiled_cover_letter: OutputSchemaCoverLetter

    error_log: Optional[str]  # Lanes with errors are automatically bypassed by the judge


class CherryPickSelection(BaseModel):
    summary_winning_key: str = Field(
        ..., description="The model lane ID containing the best professional_summary string section."
    )
    achievements_winning_key: str = Field(
        ..., description="The model lane ID containing the best key_achievements string list."
    )
    skills_winning_key: str = Field(
        ..., description="The model lane ID containing the best structured skills dictionary block."
    )
    experience_winning_key: str = Field(
        ..., description="The model lane ID containing the best structured experience chronological block list."
    )
    cl_subject_winning_key: str = Field(
        ..., description="The model lane ID containing the best cover letter subject string. Only lanes that produced a cover letter are eligible; use \"NONE\" if no lane has one."
    )
    cl_text_winning_key: str = Field(
        ..., description="The model lane ID containing the best cover letter body text narrative. Only lanes that produced a cover letter are eligible; use \"NONE\" if no lane has one."
    )
    justification_notes: str = Field(
        ..., description="Granular comparative analysis detailing why each component was picked over the alternatives."
    )
