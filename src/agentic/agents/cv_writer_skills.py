import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agentic.schemas.state import AppState
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from utilities.logger import *
from utilities.response_formatter import formatted_response
from utilities.cv_feedback import extract_feedback
from agentic.schemas.output_schema_cv_skills import PydanticSkillsSchema

def run_cv_writer_skills(state: AppState, llm: BaseChatModel, model_key: str) -> dict:
    """
    Micro-Agent: Filters, limits, translates and flattens the candidate's skills dictionary into a clean print-ready layout string.
    """
    logg_mr(gcv.AGENT_CV_WRITER_SKILLS, model_key)
    if state.get(gcv.GLOBAL_ERROR) or state.get(gcv.MODEL_RUNS, {}).get(model_key, {}).get(gcv.ERROR_LOG):
        return {}
    
    try:
        # 1. Extract context from State
        inputs = state[gcv.INPUT_DATA]
        target_lang = state[gcv.MODEL_RUNS][model_key].get(gcv.TARGET_LANGUAGE, "English")
        job_description = inputs.get(gcv.JOB)
        
        # Pull raw nested skills block from your InputSchema
        skills = inputs.get(gcv.SKILLS)
        
        # 2. Serialize input skills to clean JSON block for the LLM context window
        skills_context = json.dumps(skills, indent=2, ensure_ascii=False)
        
        # 3. Load Split Prompts
        config = gcv.prompts.get("cv_writer_skills", {})

        # Extract loop history
        todo, feedback, bypass = extract_feedback(gcv.AGENT_CV_WRITER_SKILLS, state, model_key)
        if todo == gcv.AGENT_STOP:
            return bypass

        fb = gf.extract_feedback(feedback, gcv.SKILLS)
        if fb:
            workspace = state[gcv.MODEL_RUNS][model_key]
            cv_loop = workspace.get(gcv.CV_LOOP, [])
            if cv_loop:
                latest_iteration = dict(cv_loop[-1])   
                latest_cv = latest_iteration.get(gcv.DRAFT_CV, {})
                latest_draft = latest_cv.get(gcv.SKILLS, {})
                document = "skills list"
                critic_feedback = True
        else:
            critic_feedback = False
            latest_draft = ""
            fb = ""
            document = ""
            
        system = config['system']
        user = config['user']
        revision = config['revision']
        if critic_feedback: user = user + "\n\n" + revision

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", user)
        ])

        structured_llm = llm.with_structured_output(PydanticSkillsSchema, method="json_schema")
        chain = prompt | structured_llm 
        
        # 5. Invoke Chain
        response = chain.invoke({
            "target_lang": target_lang,
            "job_description": job_description,
            "skills_context": skills_context,
            "latest_draft": latest_draft,
            "feedback": fb,
            "document": document
        })

        final_skills_dict = {
            item.category_name: item.skills_list 
            for item in response.skills
        }


        updates = {
            gcv.SKILLS: final_skills_dict,
        }
        return gf.format_cv_agent_return(model_key, state, updates)

    except Exception as e:
        return gf.format_cv_agent_return(model_key, state, {}, e)
    
