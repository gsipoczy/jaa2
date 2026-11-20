from typing import TypedDict, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
import utilities.general_functions as gf
from agentic.fetch_models import create_workflow_model
from utilities import general_functions as gf, global_const_var as gcv
from typing import Tuple, List, Dict
import yaml
from rich import print as rprint
from agentic.tools import web_search_tool

class ModelLaneConfig(TypedDict):
    """Configuration mapping for a single isolated LLM channel execution thread."""
    writer_llm: BaseChatModel
    critic_llm: BaseChatModel
    search_tool: BaseTool
    max_cv_loops: int
    max_cover_letter_loops: int

def new_registry() -> Dict[str, ModelLaneConfig]:
    """Initialize the model registry with default configurations."""
    return {}

def add_to_model_registry(
    model_registry: Dict[str, ModelLaneConfig],
    writer_provider: str,
    writer_model: str,
    critic_provider: str,
    critic_model: str,
    search_tool: BaseTool,
    max_cv_loops: int = 0,
    max_cover_letter_loops: int = 0,
    temperature = 0.0,
    base_url = None
) -> None:
    """Add a new model lane configuration to the registry."""

    lane_name = f"{writer_provider}_{writer_model.replace(':', '-')}_{gf.random_string(2)}"

    try:
        writer_llm = create_workflow_model(writer_provider, writer_model, temperature, base_url)
    except Exception as e:
        raise RuntimeError(f"Failed to create writer LLM: {e}")
    try:
        critic_llm = create_workflow_model(critic_provider, critic_model, temperature, base_url)
    except Exception as e:
        raise RuntimeError(f"Failed to create critic LLM: {e}")

    model_registry[lane_name] = ModelLaneConfig(
        writer_llm=writer_llm,
        critic_llm=critic_llm,
        search_tool=search_tool,
        max_cv_loops=max_cv_loops,
        max_cover_letter_loops = max_cover_letter_loops
    )

#________________________________________________________________________________#
# Load the provider/model list from a yaml file
#________________________________________________________________________________#
def parse_models_config(models_yaml: str = None) -> Tuple[Dict[str, ModelLaneConfig], BaseChatModel, str]:

    message = ""

    # Read the models file
    if models_yaml == None or models_yaml == "":
        requested_models = gf.read_yaml_file(gcv.config[gcv.MODELS], gcv.config[gcv.DATA_DIR])
    else:
        requested_models = yaml.safe_load(models_yaml)

    if requested_models is None or not requested_models or not isinstance(requested_models, dict):
        message = f"Error reading models configuration file '{gcv.config[gcv.MODELS]}'. Please ensure it exists and is properly formatted."
        return None, None, "Error reading models configuration file."

    # Workers
    wms = requested_models.get(gcv.MODEL_RUNS, [])
    model_registry = new_registry()

    for wm in wms:
        if isinstance(wm, dict):
            provider = wm.get("provider", "")
            model = wm.get("model", "")
            writer_provider = wm.get("writer_provider", provider)
            writer_model = wm.get("writer_model", model)
            critic_provider = wm.get("critic_provider", provider)
            critic_model = wm.get("critic_model", model)
            temperature = wm.get("temperature", gcv.WORKER_TEMPERATURE)
            base_url = wm.get("base_url", None)
            max_cv_loops = wm.get("max_cv_loops", gcv.config[gcv.MAX_CV_LOOPS])
            max_cover_letter_loops = wm.get("max_cover_letter_loops", gcv.config[gcv.MAX_CL_LOOPS])

            try:
                add_to_model_registry(
                    model_registry, 
                    writer_provider, 
                    writer_model, 
                    critic_provider, 
                    critic_model, 
                    web_search_tool,
                    max_cv_loops,
                    max_cover_letter_loops,
                    temperature,
                    base_url)
            except RuntimeError as e:
                message = message + f"\nError adding to model registry: {repr(e)}"
                model_registry = None

    # Master reviewer
    mr = requested_models.get(gcv.MASTER_REVIEWER, {})
    if not mr: master_reviewer = None
    else:
        provider = mr.get("provider", "")
        model = mr.get("model", "")
        temperature = mr.get("temperature", gcv.MASTER_TEMPERATURE)
        base_url = mr.get("base_url", None)
        try:
            master_reviewer = create_workflow_model(
                provider, 
                model,
                temperature,
                base_url)
        except Exception as e:
            message = message + f"\nError creating master reviewer: {repr(e)}"
            master_reviewer = None

    return model_registry, master_reviewer, message