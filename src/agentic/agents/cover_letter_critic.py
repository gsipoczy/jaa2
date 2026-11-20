from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from agentic.schemas.state import AppState
import utilities.global_const_var as gcv
from utilities.logger import *

def run_cover_letter_critic(state: AppState, llm: BaseChatModel, model_key: str, max_loops: int = 0) -> dict:
    """
    Agent Critically evaluates the Motivation Letter draft, 
    enforcing natural human prose limits.
    """

    logg_mr(gcv.AGENT_COVER_LETTER_CRITIC, model_key)

    if state.get(gcv.GLOBAL_ERROR):
        return {}
    workspace = state.get(gcv.MODEL_RUNS, {}).get(model_key, {})
    if workspace.get(gcv.ERROR_LOG):
        return {}

    cl_loop = list(workspace.get(gcv.COVER_LETTER_LOOP, []))
    if not cl_loop:
        return {gcv.MODEL_RUNS: {model_key: {gcv.ERROR_LOG: "Critic Exception: No Motivation Letter draft available to evaluate."}}}

    try:
        latest_iteration = dict(cl_loop[-1])

        job_desc = state[gcv.INPUT_DATA][gcv.JOB]
        target_lang = workspace.get(gcv.TARGET_LANGUAGE, "English")
        compiled_cv = workspace.get(gcv.FINAL_COMPILED_CV_TXT, "")
        cover_letter_draft = latest_iteration.get(gcv.DRAFT_COVER_LETTER, {}).get(gcv.LETTER_TEXT, {})


        config = gcv.prompts.get("agent_6_cover_letter_critic", {})

        prompt = ChatPromptTemplate.from_messages([
            ("system", config.get("system", "")),
            ("user", config.get("user", ""))
        ])

        chain = prompt | llm
        response = chain.invoke({
            "target_lang": target_lang,
            "job_description": job_desc,
            "cv": compiled_cv,
            "letter_draft": cover_letter_draft
        })

        feedback_text = ""
        rc = response.content
        if isinstance(rc, str):
            feedback_text = response.content.strip()
        elif isinstance(rc, list):
            for resl in rc:
                if isinstance(resl, str):
                    feedback_text = feedback_text + "\n" + resl
                elif isinstance(resl, dict) and 'text' in resl.keys():
                    feedback_text = resl.get('text', "")
        elif isinstance(rc, dict) and 'text' in rc.keys():
            feedback_text = resl.get('text', "")
        else:
            feedback_text = "APPROVED"

        is_approved = "APPROVED" in feedback_text.upper()

        # If the maximum number of iterations has been reached, mark no more iterations
        def_max_loops = gcv.config[gcv.MAX_CL_LOOPS]
        if max_loops > 0: def_max_loops = max_loops
        if len(cl_loop) >= def_max_loops:
            is_approved = True
            latest_iteration[gcv.NO_MORE_ITERATIONS] = True
        else:
            latest_iteration[gcv.NO_MORE_ITERATIONS] = False

        latest_iteration[gcv.FEEDBACK_COVER_LETTER] = feedback_text
        latest_iteration[gcv.APPROVED_COVER_LETTER] = is_approved
        cl_loop[-1] = latest_iteration

        update_data = {
            gcv.COVER_LETTER_LOOP: cl_loop,
            gcv.ERROR_LOG: None
        }
        if is_approved:
            update_data[gcv.FINAL_COMPILED_COVER_LETTER] = latest_iteration[gcv.DRAFT_COVER_LETTER]

        return {gcv.MODEL_RUNS: {model_key: update_data}}

    except Exception as e:
        #return {gcv.MODEL_RUNS: {model_key: {gcv.ERROR_LOG: f"Agent 6 Failure under {model_key}: {str(e)} "}}}
        return log_error(gcv.AGENT_COVER_LETTER_CRITIC, e, model_key)
