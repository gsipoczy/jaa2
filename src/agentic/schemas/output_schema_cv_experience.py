from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TailoredEmployerBlock(BaseModel):
    """Temporary schema for the LLM output mapping step."""
    employer_company: str = Field(description="The exact legal employer company name copied verbatim from the input shell.")
    dates: str = Field(description="The exact date string copied verbatim from the input shell.")
    main_responsibilities: str = Field(
        description="A comprehensive, multi-paragraph or bulleted summary written entirely in the target language. It must integrate all project context, tailored technical metrics, and tools into a single high-impact block."
    )

class PydanticExperienceSynthesisSchema(BaseModel):
    """Enforced schema for Micro-Agent 5's LLM generation step."""
    tailored_experience_blocks: List[TailoredEmployerBlock] = Field(
        description="List of chronological experience blocks matching the exact sequence, company names, and dates of the input shell."
    )

class ClientProjectItem(BaseModel):
    client_name: str = Field(description="The name of the assignment client project from the nested list (e.g., Lonza, Roche).")
    from_date: str
    to_date: str
    tailored_achievements: List[str] = Field(
        description="A list of bullet points translated and adapted to highlight software engineering value matching the target AI job."
    )

class EmployerExperienceBlock(BaseModel):
    employer_company: str = Field(description="The primary, legal employer name from the top-level parent company key (e.g., Avoras AG).")
    from_date: str
    to_date: str
    main_responsibilities: str = Field(description="High-level structural summary of the core role at this employer.")
    
    # FIX: Use default_factory to allow the model to safely omit this key if there are no sub-projects
    client_projects: List[ClientProjectItem] = Field(
        default_factory=list,
        description="List of nested client projects completed while working for this employer. Leave empty or omit if none exist."
    )