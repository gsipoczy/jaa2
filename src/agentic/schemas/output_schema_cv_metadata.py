from pydantic import BaseModel, Field


# ----------------------------------------------------
# STRICT TYPED SUB-SCHEMAS TO SATISFY OPENAI API REQS
# ----------------------------------------------------
class LanguageEntry(BaseModel):
    language: str = Field(..., description="The translated name of the language.")
    level: str = Field(..., description="The translated level of competency (e.g., C2, Anyanyelv).")

class SpecializationEntry(BaseModel):
    specialization: str = Field(..., description="The translated name of the specialization field.")

class EducationEntry(BaseModel):
    institute: str = Field(..., description="Name of the educational institution.")
    city: str = Field(..., description="City location.")
    country: str = Field(..., description="Country location.")
    field: str = Field(..., description="Translated field of study (e.g., Matematika).")
    # FIX: Swapped generic list[dict] out for an explicit typed list of Pydantic models
    specializations: list[SpecializationEntry] = Field(
        default=[], 
        description="List of specialization objects, e.g., [{'specialization': 'Számítástechnika'}]."
    )
    degree: str = Field(..., description="Translated degree title (e.g., Master of Sciences).")
    from_date: str = Field(..., description="Start year.")
    to_date: str = Field(..., description="End or graduation year.")

class PersonalInfoEntry(BaseModel):
    name: str = Field(..., description="Candidate name.")
    title: str = Field(..., description="Translated engineering or professional title.")
    field: str = Field(..., description="Translated list or summary string of core technical fields.")
    email: str = Field(..., description="Contact email address.")
    github: str = Field(..., description="GitHub profile URL link.")
    linkedin: str = Field(..., description="LinkedIn profile URL link.")
    phone: str = Field(..., description="Contact phone number.")
    address: str = Field(..., description="Physical residential address string.")
    others: str = Field(..., description="Any extra metadata notes.")

# ----------------------------------------------------
# MAIN STRUCTURAL CONTAINER SCHEMA
# ----------------------------------------------------
class MetadataEnsembleOutput(BaseModel):
    languages: list[LanguageEntry] = Field(..., description="1:1 mapped translated language structures.")
    educations: list[EducationEntry] = Field(..., description="1:1 mapped translated education entries.")
    personal_info: PersonalInfoEntry = Field(..., description="Translated basic personal information block.")

def parse_dynamic_other_info(raw_other: dict) -> str:
    """
    Dynamically flattens any unstructured lists or objects within other_info
    without hardcoding keys like 'hobbies' or 'interests'.
    """
    if not isinstance(raw_other, dict):
        return str(raw_other)
        
    lines = []
    for key, value in raw_other.items():
        formatted_key = key.replace("_", " ").title()
        
        if isinstance(value, list):
            items = []
            for item in value:
                if isinstance(item, dict):
                    items.extend([str(v) for v in item.values() if v])
                else:
                    items.append(str(item))
            lines.append(f"{formatted_key}: {', '.join(items)}")
        elif isinstance(value, dict):
            items = [f"{str(k)}: {str(v)}" for k, v in value.items()]
            lines.append(f"{formatted_key}: {', '.join(items)}")
        else:
            lines.append(f"{formatted_key}: {str(value)}")
            
    return "\n".join(lines)
