from pydantic import BaseModel, Field
from typing import List, Dict, Any

class CVCriticFeedback(BaseModel):
    personal_info: str
    professional_summary: str
    key_achievements: str
    skills: str
    certificates: str
    experience: str
    educations: str
    languages: str
    other_info: str