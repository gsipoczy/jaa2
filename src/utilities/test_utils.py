from utilities.execution import initialize_global_state, convert_inputs
from agentic.fetch_models import create_workflow_model
from typing import Any
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.chat_models import BaseChatModel

from agentic.agents.company_researcher import run_company_researcher
from agentic.agents.language_detector import agent_language_selector

from agentic.agents.cv_writer_summary import run_cv_writer_summary
from agentic.agents.cv_writer_experience import run_cv_writer_experience
from agentic.agents.cv_writer_skills import run_cv_writer_skills
from agentic.agents.cv_writer_metadata import run_cv_writer_metadata
from agentic.agents.cv_draft_compiler import run_cv_draft_compiler
from agentic.agents.cv_critic import run_cv_critic_agent
from agentic.routing import should_continue_cv_loop
from langchain_core.tools import BaseTool
from agentic.schemas.state import AppState
import utilities.global_const_var as gcv

##########################################################################################################################
# LLM models to choose
##########################################################################################################################
llm_models = {
    "Ollama Small": ["Ollama", "qwen3-coder:30b"],
    "Ollama Big": ["Ollama", "llama3:70b"],
    "OpenAI Small": ["OpenAI", "gpt-4o-mini"],
    "OpenAI Big": ["OpenAI", "gpt-4.1"],
    "Anthropic Small": ["Anthropic", "claude-sonnet-5-5"],
    "Anthropic Big": ["Anthropic", "claude-haiku-5-5"],
}

##########################################################################################################################
# Load file
##########################################################################################################################
def load_jd_file(file_name: str):
    file_name1 = "./" + file_name
    file_name2 = "./tests/" + file_name

    target_company = ""
    job_description = ""

    try:
        with open(file_name1, "r", encoding="utf-8") as file:
            first_line = True
            for line in file:
                if first_line:
                    target_company = line
                    first_line = False
                else:
                    job_description = job_description + "\n" + line
    except:
        with open(file_name2, "r", encoding="utf-8") as file:
                first_line = True
                for line in file:
                    if first_line:
                        target_company = line
                        first_line = False
                    else:
                        job_description = job_description + "\n" + line

    return target_company, job_description

##########################################################################################################################
# Initialize Inputs
##########################################################################################################################
def initialize_inputs(target_company, job_description):

    input_dict = convert_inputs(target_company, job_description,
        "",  "",  "",  "", "",  "",  "",  "",  "",  "")

    state, messages = initialize_global_state(input_dict, True)

    if not messages == "":
        raise(messages)

    return state

##########################################################################################################################
# Get LLM
##########################################################################################################################
def get_llm(llm_choice: str):

    choice = llm_models.get(llm_choice, [])
    provider = choice[0]
    model = choice[1]

    model_key = f"{provider}_{model.replace(':', '-')}_0"
    try:
        llm = create_workflow_model(provider, model)
    except Exception as e:
        print(e)
        raise RuntimeError(f"Failed to create LLM: {e}")
    #print("LLM", llm_choice, "is initialized.")
    return llm, model_key

##########################################################################################################################
# Test CV workflow
##########################################################################################################################
def create_test_cv_workflow(llm: BaseChatModel, critic_llm: BaseChatModel, model_key: str, search_tool: BaseTool) -> Any:
    workflow = StateGraph(AppState)

    # Nodes
    workflow.add_node("language_detector", lambda state: agent_language_selector(state, llm, model_key))
    workflow.add_node("company_researcher", lambda state: run_company_researcher(state, llm, search_tool, model_key))
    workflow.add_node("summary_writer", lambda state: run_cv_writer_summary(state, llm, model_key))
    workflow.add_node("experience_writer", lambda state: run_cv_writer_experience(state, llm, model_key))
    workflow.add_node("skills_writer", lambda state: run_cv_writer_skills(state, llm, model_key))
    workflow.add_node("metadata_writer", lambda state: run_cv_writer_metadata(state, llm, model_key))
    workflow.add_node("cv_compiler", lambda state: run_cv_draft_compiler(state, model_key))
    workflow.add_node("cv_critic", lambda state: run_cv_critic_agent(state, critic_llm, model_key))

    # Edges
    workflow.add_edge(START, "language_detector")
    workflow.add_edge("language_detector", "company_researcher")
    workflow.add_edge("company_researcher", "summary_writer")
    workflow.add_edge("summary_writer", "experience_writer")
    workflow.add_edge("experience_writer", "skills_writer")
    workflow.add_edge("skills_writer", "metadata_writer")
    workflow.add_edge("metadata_writer", "cv_compiler")
    workflow.add_edge("cv_compiler", "cv_critic")

    workflow.add_conditional_edges(
        "cv_critic",
        lambda state: should_continue_cv_loop(state, model_key),
        {
            gcv.LOOP_BACK_TO_CV_WRITER: "summary_writer",
            gcv.CONTINUE_TO_MOTIVATION_LETTER: END,
            gcv.SKIP_TO_FINAL_MERGE: END,
            gcv.ROUTE_TO_ERROR: END
        }
    )


    return workflow.compile()