from typing import Annotated, TypedDict, Dict, List, Optional, Any
from pydantic import BaseModel, Field

class OutputSchemaCoverLetter(BaseModel):
    #personal_info: Dict[str, Any]
    recipient: str = Field(description="The person and the name of the Company.")
    subject: str = Field(description="Letter subject, application for the job (with job title)")
    letter_text: str = Field(description="The body text of the cover letter")