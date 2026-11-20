from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from agentic.schemas.state import AppState
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from utilities.logger import *
from agentic.schemas.output_schema_cover_letter import OutputSchemaCoverLetter

def run_cover_letter_writer(state: AppState, llm: BaseChatModel, model_key: str) -> dict:
    """
    Agent Generates a tailored, persuasive Motivation Letter 
    matching the job parameters and contract type.
    """
    logg_mr(gcv.AGENT_COVER_LETTER_WRITER, model_key)

    if state.get(gcv.GLOBAL_ERROR):
        return {}
    workspace = state.get(gcv.MODEL_RUNS, {}).get(model_key, {})
    if workspace.get(gcv.ERROR_LOG):
        return {}

    try:

        cl_loop_history = list(workspace.get(gcv.COVER_LETTER_LOOP, []))
        if cl_loop_history:
            latest_iteration = dict(cl_loop_history[-1])
            fb = latest_iteration.get(gcv.FEEDBACK_COVER_LETTER, "")
            latest_draft = latest_iteration.get(gcv.DRAFT_COVER_LETTER, {})
            document = "cover_letter"
            critic_feedback = True
        else:
            critic_feedback = False
            latest_draft = ""
            fb = ""
            document = ""

        inputs = state[gcv.INPUT_DATA]

        company = inputs[gcv.COMPANY]
        job_desc = inputs[gcv.JOB]
        target_lang = workspace.get(gcv.TARGET_LANGUAGE, "English")
        research = workspace.get(gcv.COMPANY_RESEARCH, "")
        compiled_cv = workspace.get(gcv.FINAL_COMPILED_CV_TXT, "")

        raw_pi = inputs.get(gcv.PERSONAL_INFO, {})
        processed_pi = gf.flatten_dld_to_dl(raw_pi, llm, target_lang, True)

        # Fetch templates directly from your global constants dictionary cache
        config = gcv.prompts.get("agent_5_cover_letter_writer", {})
        system = config['system']
        user = config['user']
        revision = config['revision']
        if critic_feedback: user = user + "\n\n" + revision

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", user)
        ])

        structured_llm = llm.with_structured_output(OutputSchemaCoverLetter, method="json_schema")
        chain = prompt | structured_llm 

        response = chain.invoke({
            "target_language": target_lang,
            "company": company,
            "job_description": job_desc,
            "company_research": research,
            "compiled_cv": compiled_cv,
            "latest_draft": latest_draft,
            "feedback": fb,
            "document": document
        })

        
        cl_loop_history.append({
            gcv.DRAFT_COVER_LETTER: {
                gcv.PERSONAL_INFO: processed_pi,
                gcv.LETTER_TEXT: response.model_dump()
            },
            gcv.FEEDBACK_COVER_LETTER: "",
            gcv.APPROVED_COVER_LETTER: False,
            gcv.NO_MORE_ITERATIONS: False
        })

        return {
            gcv.MODEL_RUNS: {
                model_key: {
                    gcv.COVER_LETTER_LOOP: cl_loop_history,
                    gcv.ERROR_LOG: None
                }
            }
        }

    except Exception as e:
        return log_error(gcv.AGENT_COVER_LETTER_WRITER, e, model_key)
