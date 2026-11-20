from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from agentic.schemas.state import AppState
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from agentic.schemas.company_analysis_schema import CompanyAnalysisSchema
from utilities.logger import *

def run_company_researcher(state: AppState, llm: BaseChatModel, search_tool: BaseTool, model_key: str) -> dict:
    """
    Agent 2: Conducts a web search, processes raw snippets, and evaluates structural
    business classification flags (fake check, recruiter check, contract type check).
    """
    logg_mr(gcv.AGENT_COMPANY_RESEARCHER, model_key)


    if state.get("global_error") or state.get("model_runs", {}).get(model_key, {}).get("error_log"):
        logg_error_exit()
        return {}

    company = state[gcv.INPUT_DATA][gcv.COMPANY].strip()
    job_desc = state[gcv.INPUT_DATA][gcv.JOB].strip()
    
    if not company:
        return log_text_error(gcv.AGENT_COMPANY_RESEARCHER, "Company name missing.", model_key)

    try:
        # 1. Search the web
        query = f"{company} company profile culture mission values 2026"
        search_results = search_tool.invoke({"query": query})

        # 2. Load Prompts
        config = gcv.prompts.get("agent_2_company_researcher", {})
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.get("system", "")),
            ("user", config.get("user", ""))
        ])

        # 3. Bind the Pydantic schema to enforce structured output parsing
        structured_llm = llm.with_structured_output(CompanyAnalysisSchema, method="json_schema")
        chain = prompt | structured_llm

        # 4. Invoke the model
        analysis: CompanyAnalysisSchema = chain.invoke({
            "company": company,
            "job_description": job_desc,
            "raw_data": search_results
        })

        logg(gcv.LOGD, "company_research: " + analysis.culture_summary_10_points[:100])
        logg(gcv.LOGD, "is_fake_company: " + str(analysis.is_fake_company))
        logg(gcv.LOGD, "is_headhunter: " + str(analysis.is_headhunter))
        logg(gcv.LOGD, "employment_type: " + analysis.employment_type)

        # 5. Extract structural data values safely into our pipeline workspace map

        updates = {
            "company_research": analysis.culture_summary_10_points,
            "is_fake_company": analysis.is_fake_company,
            "is_headhunter": analysis.is_headhunter,
            "employment_type": analysis.employment_type,
        }
        return gf.format_cv_agent_return(model_key, state, updates, None, True)

    except Exception as e:
        return gf.format_cv_agent_return(model_key, state, {}, e)
