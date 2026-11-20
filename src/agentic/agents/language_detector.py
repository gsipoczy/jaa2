from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from agentic.schemas.state import AppState
import utilities.general_functions as gf
import utilities.global_const_var as gcv
from utilities.logger import *

def agent_language_selector(state: AppState, llm: BaseChatModel, model_key: str) -> dict:
    """
    Agent 1: Analyzes the job description text to detect its primary language.
    Gracefully logs errors to data properties for seamless rendering in Gradio.
    """
    logg_mr(gcv.AGENT_LANGUAGE_DETECTOR, model_key)

    # Guard check: If a global error has already been tripped by a previous node, short-circuit
    if state.get("global_error"):
        logg_error_exit()
        return {}

    job_desc = state[gcv.INPUT_DATA][gcv.JOB].strip()
    if not job_desc:
        return log_text_error(gcv.AGENT_LANGUAGE_DETECTOR, "Input Job Description is completely empty.", model_key)

    config_node = gcv.prompts.get("agent_1_language_selector", {})
    system_tmpl = config_node.get("system", "")
    user_tmpl = config_node.get("user", "")

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_tmpl),
            ("user", user_tmpl)
        ])

        chain = prompt | llm
        response = chain.invoke({"job_desc": job_desc})
        detected_language = response.content.strip().replace(".", "").replace('"', '').replace("'", "")
        if not detected_language or detected_language == "" or len(detected_language) > 20:
            detected_language = 'English'

        logg(gcv.LOGD, "Detected language: " + detected_language)

        return gf.format_cv_agent_return(model_key, state, {gcv.TARGET_LANGUAGE: detected_language}, None, True)

    except Exception as e:
        return gf.format_cv_agent_return(model_key, state, {}, e)
