
import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agentic.schemas.state import AppState
from agentic.schemas.output_schema_cv_critic import CVCriticFeedback
import utilities.global_const_var as gcv
from utilities.logger import *

# Stop condition in case of error
def should_continue_pipeline(state: AppState, model_key: str) -> str:
    """
    A conditional edge checker used by LangGraph to steer execution paths.
    """
    # Halt everything if a catastrophic global error occurred
    if state.get("global_error"):
        return "route_to_gradio_error_view"
        
    # Halt this specific model thread if its individual run crashed
    workspace = state.get("model_runs", {}).get(model_key, {})
    if workspace.get("error_log"):
        return "halt_this_model_thread"
        
    return "continue_to_next_agent"

# Decide whether to skip the motivation letter phase based on company type
def route_after_cv_generation(state: AppState, model_key: str) -> str:
    """ LangGraph Router to dynamically bypass or run Cover Letter writing. """
    workspace = state["model_runs"][model_key]
    
    # Rule: If it's a headhunter agency, completely skip the motivation letter phase
    if workspace.get("is_headhunter") is True or workspace.get("is_fake_company") is True:
        return "skip_to_final_merge"
        
    return "continue_to_motivation_letter"

def should_continue_cv_loop(state: AppState, model_key: str) -> str:
    """
    Evaluates the current state of the CV loop to drive execution routing.
    """
    workspace = state.get(gcv.MODEL_RUNS, {}).get(model_key, {})
    cv_loop = workspace.get(gcv.CV_LOOP, [])
    
    if not cv_loop or state.get(gcv.GLOBAL_ERROR) or workspace.get(gcv.ERROR_LOG):
        return gcv.ROUTE_TO_ERROR

    logg(gcv.LOGI, f"Completed CV Loop Iteration: {len(cv_loop)} / {gcv.config[gcv.MAX_CV_LOOPS]}")

    # The critic node sets FINAL_COMPILED_CV when the loop is finished (approved or max iterations).
    # A router cannot write state itself: LangGraph discards any mutation made here.
    if workspace.get(gcv.FINAL_COMPILED_CV):
        # Check if the company was flagged as a headhunter agency by Agent 2
        if workspace.get(gcv.IS_HEADHUNTER) is True:
            return gcv.SKIP_TO_FINAL_MERGE # Bypass Cover Letter completely
        return gcv.CONTINUE_TO_MOTIVATION_LETTER
        
    return gcv.LOOP_BACK_TO_CV_WRITER

def route_cover_letter_reflection_loop(state: AppState, model_key: str) -> str:
    """Routes execution based on Motivation Letter approval status or thresholds."""
    workspace = state.get(gcv.MODEL_RUNS, {}).get(model_key, {})
    cl_loop = workspace.get(gcv.COVER_LETTER_LOOP, [])
    
    if not cl_loop:
        return "route_to_error"
        
    latest = cl_loop[-1]
    MAX_REVIEWS = 3
    
    if latest[gcv.APPROVED] is True or len(cl_loop) >= MAX_REVIEWS:
        if not workspace.get(gcv.FINAL_COMPILED_COVER_LETTER):
            workspace[gcv.FINAL_COMPILED_COVER_LETTER] = latest[gcv.DRAFT]
        return "skip_to_final_merge"
        
    return "loop_back_to_cover_letter_writer"
