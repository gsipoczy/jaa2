import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agentic.schemas.state import AppState
from agentic.schemas.output_schema_cv_critic import CVCriticFeedback
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from utilities.logger import *
from utilities.outputs import *
from rich import print

def run_cv_critic_agent(state: AppState, llm: BaseChatModel, model_key: str, max_loops: int = 0) -> dict:
    """
    Agent CV Critic Node.
    Audits the latest assembled draft dictionary block-by-block using the
    strongly typed CVCriticFeedback schema format to drive targeted micro-agent updates.
    """

    logg_mr(gcv.AGENT_CV_CRITIC, model_key)

    # Short-circuit execution check on upstream thread failures
    if state.get(gcv.GLOBAL_ERROR) or state.get(gcv.MODEL_RUNS, {}).get(model_key, {}).get(gcv.ERROR_LOG):
        return {}

    try:
        workspace = state[gcv.MODEL_RUNS][model_key]

        # There must be an unempty CV loop, otherwise the critic has nothing to do.
        cv_loop = workspace.get(gcv.CV_LOOP, [])
        if not cv_loop:
            message = "CV Critic called but cv_loop is empty."
            log_text_error(gcv.AGENT_CV_CRITIC, message, model_key)
            return {}
            
        # Latest CV iteration
        latest_iteration = dict(cv_loop[-1])        # make sure it's a dictionary

        # CV to criticize
        draft_cv = latest_iteration.get(gcv.DRAFT_CV, {})
        if not draft_cv:
            message = "No draft CV to criticize."
            log_text_error(gcv.AGENT_CV_CRITIC, message, model_key)
            return {}

        # CV sections
        personal_info = draft_cv.get(gcv.PERSONAL_INFO, {})             # it's a dictionary of strings
        summary = draft_cv.get(gcv.PROFESSIONAL_SUMMARY, "")            # it's a string
        achievements = draft_cv.get(gcv.KEY_ACHIEVEMENTS, [])           # it's a list of strings
        skills = draft_cv.get(gcv.SKILLS, {})                           # it's a dictionary of list of strings
        certificates = draft_cv.get(gcv.CERTIFICATES, [])               # it's a list of strings
        experience = draft_cv.get(gcv.EXPERIENCE, [])                   # it's a list of dictionaries of strings
        educations = draft_cv.get(gcv.EDUCATIONS, [])                   # it's a list of dictionaries of strings
        languages = draft_cv.get(gcv.LANGUAGES, [])                     # it's a list of dictionaries of strings
        other_info = draft_cv.get(gcv.OTHER_INFO, {})                   # it's a dictionary of lists of strings

        # Job description
        job_description = state[gcv.INPUT_DATA].get(gcv.JOB, "")

        # Target language
        target_lang = workspace.get(gcv.TARGET_LANGUAGE, "English")

        # PROMPTS
        style_moderation = str(gcv.prompts.get("make_human", ""))
        system_core = str(gcv.prompts.get("cv_critic", {}).get('system_core', ''))
        user_core = str(gcv.prompts.get("cv_critic", {}).get('user_core', ''))

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_core),
            ("user", user_core)
        ])

        prompt = prompt_template.format_messages(

            # Instructive parts
            style_moderation = style_moderation,
            target_lang = target_lang,

            # Input part
            job_description = job_description,

            # CV parts to criticize
            personal_info = personal_info,
            summary = summary,
            achievements = achievements,
            skills = skills,
            certificates = certificates,
            experience = experience,
            educations = educations,
            languages = languages,
            other_info = other_info
        )
        
        # Structured LLM 
        structured_critic = llm.with_structured_output(CVCriticFeedback, method="json_schema")

        # Run LLM
        feedback = structured_critic.invoke(prompt).model_dump()

        # Decide next steps
        approved = True
        no_more_iterations = False
        skip_cover_letter = False

        # If any section is flagged for revision, mark the CV as not approved
        if feedback[gcv.PERSONAL_INFO]: approved = False
        if feedback[gcv.PROFESSIONAL_SUMMARY]: approved = False
        if feedback[gcv.KEY_ACHIEVEMENTS]: approved = False
        if feedback[gcv.SKILLS]: approved = False
        if feedback[gcv.CERTIFICATES]: approved = False
        if feedback[gcv.EXPERIENCE]: approved = False
        if feedback[gcv.EDUCATIONS]: approved = False
        if feedback[gcv.LANGUAGES]: approved = False
        if feedback[gcv.OTHER_INFO]: approved = False

        # If the maximum number of iterations has been reached, mark no more iterations
        def_max_loops = gcv.config[gcv.MAX_CV_LOOPS]
        if max_loops > 0: def_max_loops = max_loops
        if len(cv_loop) >= def_max_loops:
            approved = True
            latest_iteration[gcv.NO_MORE_ITERATIONS] = True
        else:
            latest_iteration[gcv.NO_MORE_ITERATIONS] = False

        if workspace.get(gcv.IS_FAKE_COMPANY, False) or workspace.get(gcv.IS_HEADHUNTER, False):
            skip_cover_letter = True

        latest_iteration[gcv.SKIP_COVER_LETTER] = skip_cover_letter
        latest_iteration[gcv.FEEDBACK_CV] = feedback
        latest_iteration[gcv.APPROVED_CV] = approved
        cv_loop[-1] = latest_iteration
        updated_loop = cv_loop


        updates = {
            gcv.CV_LOOP: updated_loop,
        }  

        # Finish the loop: approved, or max iterations hit without a full approval flag
        if approved or len(cv_loop) >= gcv.config[gcv.MAX_CV_LOOPS]:
            updates[gcv.FINAL_COMPILED_CV] = latest_iteration[gcv.DRAFT_CV]
            cv_txt = convert_struct_to_txt(updates[gcv.FINAL_COMPILED_CV])
            updates[gcv.FINAL_COMPILED_CV_TXT] = cv_txt

        return gf.format_cv_agent_return(model_key, state, updates, error_log=None, upper_level=True)

    except Exception as e:
        return gf.format_cv_agent_return(model_key, state, {}, e)
