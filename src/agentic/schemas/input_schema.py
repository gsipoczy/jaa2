from typing import Annotated, TypedDict, Dict, List, Optional, Any
from pydantic import BaseModel, Field

class InputSchema(TypedDict, total=False):
    """Plain dict schema; use new_input_data() to get one pre-filled with defaults."""

    target_company: str
    job_description: str

    personal_info: dict
    projects: List[dict]
    skills: dict
    achievements: List[dict]
    certificates: List[dict]
    educations: List[dict]
    languages: List[dict]
    other_info: dict

def new_input_data() -> InputSchema:
    """Create an InputSchema with empty defaults so it can be filled step by step."""
    return InputSchema(
        target_company="",
        job_description="",
        personal_info={},
        projects=[],
        skills={},
        achievements=[],
        certificates=[],
        educations=[],
        languages=[],
        other_info={},
    )