from langchain_core.language_models.chat_models import BaseChatModel
from agentic.schemas.state import AppState
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from utilities.logger import *
from utilities.translator import *

def run_label_formatter(state: AppState, llm: BaseChatModel, model_key: str) -> dict:
    """
    Simply translates the configured labels to the target language
    """
    logg_mr(gcv.AGENT_LABEL_FORMATTER, model_key)


    if state.get("global_error") or state.get("model_runs", {}).get(model_key, {}).get("error_log"):
        logg_error_exit()
        return {}

    labels = gcv.config.get(gcv.LABELS, [])
    target_lang = state.get(gcv.MODEL_RUNS, {}).get(model_key, {}).get(gcv.TARGET_LANGUAGE, "English")

    try:
        for lbl in labels:
            lbl['text'] = translate_block(llm, lbl['text'], target_lang)

        updates = {
            "labels": labels,
        }
        return gf.format_cv_agent_return(model_key, state, updates, None, True)

    except Exception as e:
        return gf.format_cv_agent_return(model_key, state, {}, e)
