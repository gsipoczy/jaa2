import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agentic.schemas.state import AppState
from agentic.schemas.output_schema_cv_summary import create_dynamic_schema
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from utilities.cv_feedback import extract_feedback
from utilities.logger import *


def run_cv_writer_summary(state: AppState, llm: BaseChatModel, model_key: str) -> dict:
    """
    Micro-Agent: Generates a tailored, highly optimized professional summary parsing structured candidate data profiles.
    """
    logg_mr(gcv.AGENT_CV_WRITER_SUMMARY, model_key)

    if state.get(gcv.GLOBAL_ERROR) or state.get(gcv.MODEL_RUNS, {}).get(model_key, {}).get(gcv.ERROR_LOG):
        logg_error_exit()
        return {}
    
    try:
        # 1. Extract context from State using absolute schema keys
        inputs = state[gcv.INPUT_DATA]
        target_lang = state[gcv.MODEL_RUNS][model_key].get(gcv.TARGET_LANGUAGE, "English")
        logg(gcv.LOGD, "Language: " + target_lang)
        
        # Extract research data from Agent 2 Company Researcher
        employment_type = state[gcv.MODEL_RUNS][model_key].get(gcv.EMPLOYMENT_TYPE, "")
        logg(gcv.LOGD, "Employement Type: " + employment_type)
        
        # Explicitly extract sub-dictionaries from your InputSchema
        personal_info = inputs.get(gcv.PERSONAL_INFO, {})
        skills = inputs.get(gcv.SKILLS, {})
        projects = inputs.get(gcv.PROJECTS, [])
        achievements = inputs.get(gcv.ACHIEVEMENTS, [])
        
        # 2. Format a structured, hallucination-resistant context block for the LLM
        candidate_profile_context = (
            f"Candidate Core Persona: {personal_info.get('title', '')} with expertise in {personal_info.get('field', '')}\n"
            f"Technical & Soft Skills: {json.dumps(skills, indent=2, ensure_ascii=False)}\n"
            f"Chronological Assignments & Projects: {json.dumps(projects, indent=2, ensure_ascii=False)}\n"
            f"Highlighted Core Achievements: {json.dumps(achievements, indent=2, ensure_ascii=False)}"
        )

        # Extract loop history
        todo, feedback, bypass = extract_feedback(gcv.AGENT_CV_WRITER_SUMMARY, state, model_key)
        if todo == gcv.AGENT_STOP:
            logg(gcv.LOGD, "According to the history, I have nothing to do, exiting.")
            return bypass
        fb = gf.extract_feedback(feedback, gcv.PROFESSIONAL_SUMMARY)
        if fb:
            workspace = state[gcv.MODEL_RUNS][model_key]
            cv_loop = workspace.get(gcv.CV_LOOP, [])
            if cv_loop:
                latest_iteration = dict(cv_loop[-1])   
                latest_cv = latest_iteration.get(gcv.DRAFT_CV, {})
                latest_draft = latest_cv.get(gcv.PROFESSIONAL_SUMMARY, {})
                document = "summary"
                critic_feedback = True
        else:
            critic_feedback = False
            latest_draft = ""
            fb = ""
            document = ""

        # Prompt
        config = gcv.prompts.get("cv_writer_summary", {})
        system = config['system']
        user = config['user']
        revision = config['revision']
        if critic_feedback: user = user + "\n\n" + revision


        ProfessionalSummaryOutput = create_dynamic_schema(target_lang)
        structured_llm = llm.with_structured_output(ProfessionalSummaryOutput, method="json_schema")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", user)
        ])
        
        chain = prompt | structured_llm
        
        # 5. Invoke Chain with mapped values
        response = chain.invoke({
            "target_lang": target_lang,
            "job_description": inputs.get(gcv.JOB, ""),
            "target_company": inputs.get(gcv.COMPANY, ""),
            "candidate_context": candidate_profile_context,
            "employment_type": employment_type,
            "latest_draft": latest_draft,
            "feedback": fb,
            "document": document
        })
        
        # 6. Package payload for the Annotated Dict model_runs Reducer
        node_output = {
            "summary_text": response.summary_text,
            "justification": response.justification
        }

        updates = {
            gcv.PROFESSIONAL_SUMMARY: node_output,
        }

        return gf.format_cv_agent_return(model_key, state, updates)

    except Exception as e:
        return gf.format_cv_agent_return(model_key, state, {}, e)
