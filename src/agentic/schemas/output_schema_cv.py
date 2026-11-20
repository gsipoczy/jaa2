from pydantic import BaseModel, Field
from typing import List, Dict, Any

from agentic.schemas.output_schema_cv_experience import EmployerExperienceBlock

class PydanticCVSynthesisSchema(BaseModel):
    """Rigid translation and layout generation targets for Agent 3."""
    professional_summary: str = Field(description="2-3 paragraphs heavily framing the profile around target metrics.")
    key_achievements: List[str] = Field(description="List of strings compiled from source achievements.")
    skills: Dict[str, List[str]] = Field(description="Top 20 technical keys grouped by Category names.")
    certificates: List[str] = Field(description="List of strings matching 'Issuer - Course Title'.")
    
    # Enforce this exact variable key name to sync cleanly with the prompt instructions
    tailored_experience_blocks: List[Dict[str, Any]] = Field(
        description="Chronological timeline structured by actual primary employers matching shell indexes."
    )

class OutputSchemaCV(BaseModel):
    personal_info: Dict[str, Any]
    professional_summary: str
    key_achievements: List[str]
    skills: Dict[str, List[str]]
    certificates: List[str]
    experience: List[EmployerExperienceBlock] # Linked directly to our hierarchical layout block
    educations: List[Dict[str, Any]]
    languages: List[Dict[str, Any]]
    other_info: Dict[str, List[str]]
