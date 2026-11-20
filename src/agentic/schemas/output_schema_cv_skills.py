from pydantic import BaseModel
from typing import List

# Represent each dictionary item as an object
class SkillCategory(BaseModel):
    category_name: str
    skills_list: List[str]

class PydanticSkillsSchema(BaseModel):
    skills: List[SkillCategory]
