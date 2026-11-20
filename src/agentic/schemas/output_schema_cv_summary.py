from pydantic import BaseModel, Field

def create_dynamic_schema(target_lang: str) -> type[BaseModel]:
    """
    Dynamically creates a Pydantic schema forcing the language constraint 
    directly into the JSON Schema schema metadata itself. This prevents 
    local models like Qwen from defaulting to English text fields.
    """
    class ProfessionalSummaryOutput(BaseModel):
        summary_text: str = Field(
            ..., 
            description=f"The tailored 3-sentence professional summary. CRITICAL REQUIREMENT: This string MUST be written 100% in {target_lang}."
        )
        justification: str = Field(
            ..., 
            description="A short explanation of how this summary was optimized for the specific employment_type."
        )
    return ProfessionalSummaryOutput