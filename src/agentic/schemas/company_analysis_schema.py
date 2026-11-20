from typing import Annotated, TypedDict, Dict, List, Optional, Any
from pydantic import BaseModel, Field

class CompanyAnalysisSchema(BaseModel):
    """Structured data returned by the company researcher analysis step."""
    is_fake_company: bool = Field(
        description="True if the web search returned zero meaningful records or if the company name appears entirely fictional."
    )
    is_headhunter: bool = Field(
        description="True if this is a recruitment agency, staff outsourcing company, or headhunter searching on behalf of a third-party client."
    )
    employment_type: str = Field(
        description="Must be strictly one of these three strings: 'permanent', 'freelance_temporary', or 'unknown'."
    )
    culture_summary_10_points: str = Field(
        description="The detailed 10-point markdown bulleted list outlining company culture, goals, and values. Leave empty if is_fake_company is True."
    )