import os
from anyio import Path
import requests
from openai import OpenAI
from utilities import general_functions as gf, global_const_var as gcv
from typing import Tuple, List, Dict
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
import yaml
from rich import print

#________________________________________________________________________________#
# Get the list of available models from each provider dynamically at runtime
#________________________________________________________________________________#

# Ollama
def get_dynamic_ollama_models(base_url: str) -> list[str]:
    """Dynamically queries your local Ollama instance for pulled models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code == 200:
            return [model["name"] for model in response.json().get("models", [])]
    except Exception:
        pass
    return [] 

# OpenAI
def get_dynamic_openai_models() -> list[str]:
    """Dynamically queries the official OpenAI API for available chat models."""
    if not os.environ.get("OPENAI_API_KEY"):
        return [] # Fallback if no key is set yet
    try:
        client = OpenAI()
        models = client.models.list()
        # Filter for standard production chat/completion models
        return [m.id for m in models if m.id.startswith(("gpt-4", "gpt-3.5", "o1"))]
    except Exception:
        return []

# Anthropic
def get_dynamic_anthropic_models() -> list[str]:
    """Queries Anthropic API directly for active Claude chat models."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return [] # Fallbacks
    try:
        # Anthropic exposes a clean /v1/models endpoint
        url = "https://api.anthropic.com/v1/models"
        headers = {
            "X-API-Key": api_key,
            "anthropic-version": "2023-06-01"
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return [m["id"] for m in response.json().get("data", [])]
    except Exception:
        pass
    return []

# Google
def get_dynamic_google_models() -> list[str]:
    """Queries Google GenAI API for current Gemini chat models."""
    # Works with standard GEMINI_API_KEY or GOOGLE_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return []
    try:
        from google import genai
        client = genai.Client()
        # client.models.list() returns metadata for all Gemini variants
        return [
            m.name.split("/")[-1] for m in client.models.list()
            if "generateContent" in m.supported_generation_methods
        ]
    except Exception:
        pass
    return []

# Groq
def get_dynamic_groq_models() -> list[str]:
    """Queries Groq API for active high-speed open-weights chat models."""
    if not os.environ.get("GROQ_API_KEY"):
        return []
    try:
        from groq import Groq
        client = Groq()
        models = client.models.list()
        # Return only active models, skipping text-embedding variants
        return [m.id for m in models.data if "embed" not in m.id]
    except Exception:
        pass
    return []

# Mistral
def get_dynamic_mistral_models() -> list[str]:
    """Queries Mistral AI API for official open & commercial chat models."""
    if not os.environ.get("MISTRAL_API_KEY"):
        return []
    try:
        from mistralai import Mistral
        client = Mistral()
        models = client.models.list()
        # Filter for models capable of chat completion tasks
        return [m.id for m in models.data if getattr(m.capabilities, "completion_chat", True)]
    except Exception:
        pass
    return []

#________________________________________________________________________________#
# Full models list
#________________________________________________________________________________#
# Get all available models from each provider
def get_all_available_models() -> dict[str, list[str]]:
    """Returns a dictionary of all available models from each provider."""
    return {
        "Ollama": get_dynamic_ollama_models(gf.get_wsl_host_ip()),
        "OpenAI": get_dynamic_openai_models(),
        "Anthropic": get_dynamic_anthropic_models(),
        "Google": get_dynamic_google_models(),
        "Groq": get_dynamic_groq_models(),
        "Mistral": get_dynamic_mistral_models()
    }

#________________________________________________________________________________#
# One model, just to have somethong
#________________________________________________________________________________#
def get_first_available_model() -> str | None:
    """Returns the first available model, or None if unavailable."""
    models = get_all_available_models()
    for provider, provider_models in models.items():
        return provider, provider_models[0]
    return None

#________________________________________________________________________________#
# List or Save the provider/model list into a yaml file
#________________________________________________________________________________#
def save_or_list_models(save_it: bool = False) -> None:
    """Creates a YAML file listing all available models from each provider."""
    providers = get_all_available_models()

    if save_it:
        gf.save_file(providers, gcv.config[gcv.MODELS], gcv.config[gcv.DATA_DIR])
    else:
        print(providers)




#________________________________________________________________________________#
# Load the provider/model list from a yaml file
#________________________________________________________________________________#
def create_workflow_model(
        provider: str, 
        model_name: str, 
        temperature: float = 0.3, 
        base_url: str = None) -> BaseChatModel:
    """
    Instantiates a LangChain chat model wrapper dynamically using `init_chat_model`.
    Explicitly assigns model_provider as a core keyword argument to avoid type inferences.
    """
    # Fail fast: init_chat_model(model="") silently returns a "configurable" placeholder
    # that only crashes at .invoke() time with "_init_chat_model_helper() missing ... 'model'"
    if not provider or not model_name:
        raise ValueError(f"Missing provider or model name (provider='{provider}', model='{model_name}')")

    provider_clean = provider.lower().strip()
    
    # Standardizing provider namespaces for LangChain ecosystem routing
    provider_mapping = {
        "ollama": "ollama",
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google_genai",
        "groq": "groq",
        "mistral": "mistralai"
    }
    
    resolved_provider = provider_mapping.get(provider_clean, provider_clean)
    
    # Pack underlying structural model settings
    extra_kwargs = {}
    
    # Only assign temperature if the provider is NOT Anthropic
    if resolved_provider != "anthropic":
        extra_kwargs["temperature"] = temperature

    # Disable thinking mode to ensure clean .with_structured_output validation
    if resolved_provider == "anthropic":
        if "opus" in model_name.lower():
            # Adaptive reasoning mode with low effort.
            # Effort is NOT part of the thinking block: it goes to output_config.effort
            # (ChatAnthropic exposes it as the `effort` / `reasoning_effort` field)
            extra_kwargs["thinking"] = {"type": "adaptive"}
            extra_kwargs["effort"] = "low"
        else:
            # Standard Sonnet models 
            extra_kwargs["thinking"] = {
                "type": "between_tools"
            }
    
    # Correctly inject custom host endpoints (like WSL network setups)
    if provider_clean == "ollama":
        if not base_url:
            base_url = gf.get_wsl_host_ip()
        extra_kwargs["base_url"] = base_url
        
    # FIX: model_provider MUST be an explicit keyword argument to the function
    return init_chat_model(
        model=model_name,
        model_provider=resolved_provider,
        **extra_kwargs
    )

#________________________________________________________________________________#
# Initialize all requested LLMs
#________________________________________________________________________________#
def initialize_all_workflow_engines() -> Tuple[Dict[str, BaseChatModel], BaseChatModel]:
    """
    Loads the config file, instantiates live LangChain chat engine objects,
    and returns a tuple containing:
      1. A dictionary of active workers mapped by a unique runtime key.
      2. The dedicated Master Reviewer model engine instance.
    """
    # No need, we have it in global_const_var.py already
    #worker_configs, master_config = parse_models_config()

    
    live_workers: Dict[str, BaseChatModel] = {}
    
    # 1. Instantiate the parallel worker models
    for index, config in enumerate(gcv.worker_models):
        provider = config["provider"]
        model_name = config["model"]
        
        # Create a unique key for the worker (e.g., "ollama_qwen3-coder:30b_0")
        # to handle cases where someone might use variants of the same provider
        unique_key = f"{provider}_{model_name.replace(':', '-')}_{index}"
        
        # Instantiate through our dynamic runtime manager
        live_workers[unique_key] = create_workflow_model(
            provider=provider,
            model_name=model_name,
            temperature=gcv.WORKER_TEMPERATURE
        )
        
    # 2. Instantiate the Master Reviewer model
    live_master = create_workflow_model(
        provider=gcv.master_reviewer["provider"],
        model_name=gcv.master_reviewer["model"],
        temperature=gcv.MASTER_TEMPERATURE
    )
    
    return live_workers, live_master

