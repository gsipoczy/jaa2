from pydantic import BaseModel, Field

class SectionSelection(BaseModel):
    winning_model_key: str = Field(
        ..., 
        description="The exact model_key (e.g., 'gpt-4o', 'claude-sonnet') chosen for this specific section."
    )
    justification: str = Field(
        ..., 
        description="A short, clear critique explaining why this model's translation, tone, and constraint adherence won."
    )

class MasterJudgeSelection(BaseModel):
    professional_summary: SectionSelection = Field(..., description="Selection for the top summary paragraph.")
    experience_compiler: SectionSelection = Field(..., description="Selection for the chronological project timeline.")
    skills_compiler: SectionSelection = Field(..., description="Selection for the comma-separated skill blocks.")
    cv_metadata_compiler: SectionSelection = Field(..., description="Selection for achievements, languages, and education.")
