
from langchain_core.language_models.chat_models import BaseChatModel

from agentic.schemas.model_registry import ModelLaneConfig
from agentic.schemas.state import AppState
import utilities.global_const_var as gcv
from agentic.schemas.output_schema_cv import OutputSchemaCV
from agentic.schemas.output_schema_cover_letter import OutputSchemaCoverLetter
from utilities.logger import *

from agentic.agents.language_detector import agent_language_selector
from agentic.agents.company_researcher import run_company_researcher
from agentic.agents.cv_writer_summary import run_cv_writer_summary
from agentic.agents.cv_writer_experience import run_cv_writer_experience
from agentic.agents.cv_writer_skills import run_cv_writer_skills
from agentic.agents.cv_writer_metadata import run_cv_writer_metadata
from agentic.agents.cv_draft_compiler import run_cv_draft_compiler
from agentic.agents.cv_critic import run_cv_critic_agent
from agentic.agents.cv_router import run_cv_router
from agentic.agents.cover_letter_writer import run_cover_letter_writer
from agentic.agents.cover_letter_critic import run_cover_letter_critic
from agentic.agents.cover_letter_router import run_cover_letter_router
from agentic.agents.master_judge import run_master_judge
from agentic.agents.label_formatter import run_label_formatter

def marker(text: str = "MARKER"):
    """Return a no-op node function that logs the marker each time the node runs."""
    def _marker_node(state: AppState) -> dict:
        logg_mr("***** " + text + " *****" , "MARKER")
        return {}
    return _marker_node

def language_selector(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["writer_llm"]
    return agent_language_selector(state, llm, model_key)

def company_researcher(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["writer_llm"]
    search_tool = config["search_tool"]
    return run_company_researcher(state, llm, search_tool,model_key)

def label_formatter(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["writer_llm"]
    return run_label_formatter(state, llm, model_key)

def summary_writer(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["writer_llm"]
    return run_cv_writer_summary(state, llm, model_key)

def experience_writer(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["writer_llm"]
    return run_cv_writer_experience(state, llm, model_key)

def skills_writer(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["writer_llm"]
    return run_cv_writer_skills(state, llm, model_key)

def metadata_writer(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["writer_llm"]
    return run_cv_writer_metadata(state, llm, model_key)

def cv_compiler(state: AppState, model_key: str) -> dict:

    return run_cv_draft_compiler(state, model_key)

def cv_critic(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["critic_llm"]
    max_loops = config.get("max_cv_loops", 0)
    return run_cv_critic_agent(state, llm, model_key, max_loops)

def cv_decision_router(state: AppState, model_key: str) -> dict:
    # No routing here: the join edge in the workflow waits for all lanes before the judge runs
    return run_cv_router(state, model_key)

def cover_letter_writer(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["writer_llm"]
    return run_cover_letter_writer(state, llm, model_key)

def cover_letter_critic(state: AppState, model_key: str, registry: dict[str, ModelLaneConfig]) -> dict:

    config = registry[model_key]
    llm = config["critic_llm"]
    max_loops = config.get("max_cover_letter_loops", 0)
    return run_cover_letter_critic(state, llm, model_key, max_loops)

def cover_letter_decision_router(state: AppState, model_key: str) -> dict:
    # No routing here: the join edge in the workflow waits for all lanes before the judge runs
    return run_cover_letter_router(state, model_key)

def master_judge(state: AppState, llm: BaseChatModel) -> dict:
    return run_master_judge(state, llm)

def handle_global_error(state: AppState) -> dict:
    """
    Model-independent terminal node, reached when state['global_error'] is set
    (at workflow start or by the master judge).
    Logs the global error together with any per-lane errors and records a readable
    report in 'judge_justification_notes'. 'global_error' itself is left untouched,
    so the caller can still detect the failure on the returned state.
    """
    logg_mr(gcv.AGENT_GLOBAL_ERROR_HANDLER, "GLOBAL")

    global_error = state.get(gcv.GLOBAL_ERROR) or "Unknown global error"

    # Collect per-lane errors: they often explain why the global step failed
    # (e.g. "No parallel model lane execution completed successfully").
    lane_errors = [
        f"- {model_key}: {workspace.get(gcv.ERROR_LOG)}"
        for model_key, workspace in (state.get(gcv.MODEL_RUNS) or {}).items()
        if isinstance(workspace, dict) and workspace.get(gcv.ERROR_LOG)
    ]

    report = f"WORKFLOW ABORTED - GLOBAL ERROR:\n{global_error}"
    if lane_errors:
        report += "\n\nModel lane errors:\n" + "\n".join(lane_errors)

    logg(1, gcv.LOG_ENDSEP)
    logg(1, report)
    logg(1, gcv.LOG_ENDSEP)

    return {"judge_justification_notes": report}
