from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from typing import Any, Tuple, Dict
from agentic.schemas.state import AppState
from agentic.agents.agent_wrappers import *
from agentic.schemas.model_registry import ModelLaneConfig

######################################################################################
# Orchestrator fan-out node
######################################################################################
class OrchestratorRouter:
    def __init__(self, registry: dict[str, ModelLaneConfig]):
        # Store dependency cleanly on the instance
        self.registry = registry

    def __call__(self, state: AppState) -> list[str]:
        """LangGraph calls this directly, passing only the state."""
        if state.get("global_error"):
            return ["handle_global_error_node"]
            
        # Dynamically calculate targets using instance data
        return [f"START_MODEL_RUN_{key}" for key in self.registry.keys()]
    
def route_after_judge(state: AppState) -> str:
    """After the final judge: go to the global error handler if an error is set, otherwise end."""
    return "handle_global_error_node" if state.get("global_error") else END

######################################################################################
# Create the master workflow
######################################################################################
def create_master_workflow(registry: dict[str, ModelLaneConfig], master_judge_llm: BaseChatModel) -> Any:

    master_workflow = StateGraph(AppState)

    ######################################################################
    # NODES
    ######################################################################

    for model_key in registry.keys():

        master_workflow.add_node(f"START_MODEL_RUN_{model_key}", marker("START_MODEL_RUN"))

        master_workflow.add_node(f"language_detector_{model_key}", 
                                 lambda s, mk=model_key: language_selector(s, mk, registry))
        master_workflow.add_node(f"company_researcher_{model_key}", 
                                 lambda s, mk=model_key: company_researcher(s, mk, registry))
        master_workflow.add_node(f"label_formatter_{model_key}", 
                                         lambda s, mk=model_key: label_formatter(s, mk, registry))

        master_workflow.add_node(f"START_CV_LOOP_{model_key}", marker("START_CV_LOOP"))

        master_workflow.add_node(f"summary_writer_{model_key}", 
                                 lambda s, mk=model_key: summary_writer(s, mk, registry))
        master_workflow.add_node(f"experience_writer_{model_key}", 
                                 lambda s, mk=model_key: experience_writer(s, mk, registry))
        master_workflow.add_node(f"skills_writer_{model_key}", 
                                 lambda s, mk=model_key: skills_writer(s, mk, registry))
        master_workflow.add_node(f"metadata_writer_{model_key}", 
                                 lambda s, mk=model_key: metadata_writer(s, mk, registry))
        master_workflow.add_node(f"cv_compiler_{model_key}", 
                                 lambda s, mk=model_key: cv_compiler(s, mk))
        master_workflow.add_node(f"cv_critic_{model_key}", 
                                 lambda s, mk=model_key: cv_critic(s, mk, registry))
        master_workflow.add_node(f"cv_decision_{model_key}", 
                                 lambda s, mk=model_key: cv_decision_router(s, mk))
        
        master_workflow.add_node(f"START_COVER_LETTER_LOOP_{model_key}", marker("START_COVER_LETTER_LOOP"))

        master_workflow.add_node(f"cover_letter_writer_{model_key}", 
                                 lambda s, mk=model_key: cover_letter_writer(s, mk, registry))
        master_workflow.add_node(f"cover_letter_critic_{model_key}", 
                                 lambda s, mk=model_key: cover_letter_critic(s, mk, registry))
        master_workflow.add_node(f"cover_letter_decision_{model_key}", 
                                         lambda s, mk=model_key: cover_letter_decision_router(s, mk))

        master_workflow.add_node(f"END_MODEL_RUN_{model_key}", marker("END_MODEL_RUN"))

    # Add the final Maste Judge
    master_workflow.add_node("final_judge_agent", lambda s: master_judge(s, master_judge_llm))

    # Model-independent global error handler (reachable from START and after the judge)
    master_workflow.add_node("handle_global_error_node", handle_global_error)

    ######################################################################
    # FAN OUT
    ######################################################################
    router = OrchestratorRouter(registry)
    master_workflow.add_conditional_edges(
        START,
        router,
        path_map=[f"START_MODEL_RUN_{mk}" for mk in registry.keys()] + ["handle_global_error_node"]
    )

    ######################################################################
    # EDGES
    ######################################################################

    # Connect linear processing steps within individual isolated model keys
    for model_key in registry.keys():

        master_workflow.add_edge(f"START_MODEL_RUN_{model_key}", f"language_detector_{model_key}")
        master_workflow.add_edge(f"language_detector_{model_key}", f"company_researcher_{model_key}")
        master_workflow.add_edge(f"company_researcher_{model_key}", f"label_formatter_{model_key}")
        master_workflow.add_edge(f"label_formatter_{model_key}", f"START_CV_LOOP_{model_key}")

        master_workflow.add_edge(f"START_CV_LOOP_{model_key}", f"summary_writer_{model_key}")
        master_workflow.add_edge(f"summary_writer_{model_key}", f"experience_writer_{model_key}")
        master_workflow.add_edge(f"experience_writer_{model_key}", f"skills_writer_{model_key}")
        master_workflow.add_edge(f"skills_writer_{model_key}", f"metadata_writer_{model_key}")
        master_workflow.add_edge(f"metadata_writer_{model_key}", f"cv_compiler_{model_key}")
        master_workflow.add_edge(f"cv_compiler_{model_key}", f"cv_critic_{model_key}")
        master_workflow.add_edge(f"cv_critic_{model_key}", f"cv_decision_{model_key}")

        master_workflow.add_edge(f"START_COVER_LETTER_LOOP_{model_key}", f"cover_letter_writer_{model_key}")
        master_workflow.add_edge(f"cover_letter_writer_{model_key}", f"cover_letter_critic_{model_key}")
        master_workflow.add_edge(f"cover_letter_critic_{model_key}", f"cover_letter_decision_{model_key}")

    # Fan-in / join: the judge runs only once ALL lanes have reached their END_MODEL_RUN node.
    # (Separate add_edge calls per lane would trigger the judge as soon as the first lane finishes.)
    master_workflow.add_edge([f"END_MODEL_RUN_{mk}" for mk in registry.keys()], "final_judge_agent")

    # The judge may set global_error itself -> route to the error handler, otherwise finish
    master_workflow.add_conditional_edges(
        "final_judge_agent",
        route_after_judge,
        path_map=["handle_global_error_node", END]
    )
    master_workflow.add_edge("handle_global_error_node", END)

    return master_workflow.compile()

######################################################################################
# Workflow execution entry point
######################################################################################
def execute_master_workflow(
        state: AppState, 
        registry: dict[str, ModelLaneConfig], 
        master_judge_llm: BaseChatModel) -> Tuple[Dict, str]:

    workflow = create_master_workflow(registry, master_judge_llm)
    result = workflow.invoke(state)
    errors = ""

    global_error = state.get(gcv.GLOBAL_ERROR)
    if global_error:
        errors = global_error

    runs = result.get(gcv.MODEL_RUNS)
    if runs and isinstance(runs, dict):
        for model_key, run in runs.items():
            if run and isinstance(run, dict):
                error_log = run.get(gcv.ERROR_LOG)
                if error_log:
                    etxt = model_key + ": " + error_log
                    errors = errors + "\n" + etxt

    if not errors:
        errors = "Workflow executed successfully."

    return result, errors