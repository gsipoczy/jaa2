import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import subprocess
import traceback
from agentic.schemas.state import AppState
import yaml
import utilities.global_const_var as gcv
from rich import print
from langchain_core.language_models.chat_models import BaseChatModel
from utilities.translator import translate_block
import sys
import random
import string
from utilities.logger import *

def get_full_path(file_name: str, subdir: str = None) -> str:
    """
    Returns the absolute path of a file
    """
    if subdir is None:
        return os.path.join(os.path.dirname(__file__), '..', '..', file_name)       
    return os.path.join(os.path.dirname(__file__), '..', '..', subdir, file_name)

def init_defaults():
    gcv.config[gcv.DATA_DIR] = "data"
    gcv.config[gcv.MODELS] = "models"
    gcv.config[gcv.MAX_MODELS] = 5
    gcv.config[gcv.PROMPTS] = "prompts"
    gcv.config[gcv.PERSONAL_INFO] = "personal_info"
    gcv.config[gcv.PROJECTS] = "projects"
    gcv.config[gcv.SKILLS] = "skills"
    gcv.config[gcv.ACHIEVEMENTS] = "achievements"
    gcv.config[gcv.CERTIFICATES] = "certificates"
    gcv.config[gcv.EDUCATIONS] = "educations"
    gcv.config[gcv.LANGUAGES] = "languages"
    gcv.config[gcv.OTHER_INFO] = "other_info"
    
    gcv.config[gcv.TEMPLATES_DIR] = "templates"
    gcv.config[gcv.CV_TEMPLATE] = "cv_template.fodt"
    gcv.config[gcv.COVLET_TEMPLATE] = "cover_letter_template.fodt"

    gcv.config[gcv.OUTPUT_DIR] = "output"
    gcv.config[gcv.LOG_MODE] = gcv.LOG_PRINT
    gcv.config[gcv.LOG_LEVEL] = gcv.NO_COMMENTS
    gcv.config[gcv.MAX_CV_LOOPS] = 3
    gcv.config[gcv.MAX_CL_LOOPS] = 3
    gcv.config[gcv.LABELS] = []



def read_config(msgs, supress_logging: bool = False):

    init_defaults()

    user_config = read_yaml_file(gcv.CONFIG_FILE_NAME)
    if user_config is None or not isinstance(user_config, dict) or not user_config:
        msgs = msgs + f"\nWARNING: Configuration file '{gcv.CONFIG_FILE_NAME}' not found. Using default settings."

    for key, value in user_config.items():
        gcv.config[key] = value

    if not supress_logging:
        logg(2, "Configuration =\n")
        for key, value in gcv.config.items():
            if isinstance(value, str):
                logg(2, key + " = " + value)
        logg(2, "\n\n")

    return msgs

def read_yaml_file(file_name: str, subdir: str = None) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    """

    res = {}
    try:
        file_path = get_full_path(file_name, subdir)
        with open(file_path, 'r', encoding="utf-8") as f:
            res = yaml.safe_load(f)
            return res
    except: pass

    try:
        fyaml = file_name + ".yaml"
        file_path = get_full_path(fyaml, subdir)
        with open(file_path, 'r', encoding="utf-8") as f:
           res = yaml.safe_load(f)
           return res
    except: pass

    try:
        fyml = file_name + ".yml"
        file_path = get_full_path(fyml, subdir)
        with open(file_path, 'r', encoding="utf-8") as f:
            res = yaml.safe_load(f)
            return res
    except: pass

    return res

def save_file(data: any, file_name: str, subdir: str = None) -> None:
    """
    A completely generic utility function that serializes any standard nested 
    Python data structure (dicts, lists, scalars) directly into clean YAML format.
    """
    file_path = get_full_path(file_name, subdir)
    dir_path = os.path.dirname(file_path)
    path = Path(dir_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    
def get_wsl_host_ip() -> str:
    """
    Executes a shell pipeline to find the default network bridge IP 
    connecting the WSL Linux instance to the Windows host.
    """

    ollama_port = gcv.config.get("ollama_port") if gcv.config else "11434"
    if not ollama_port:
        ollama_port = "11434"  # Default port if not specified in config

    try:
        # Runs: ip route | grep default | awk '{print $3}'
        cmd = "ip route | grep default | awk '{print $3}'"
        host_ip = subprocess.check_output(cmd, shell=True, text=True).strip()
        if host_ip:
            return f"http://{host_ip}:{ollama_port}"
    except Exception as e:
        print(f"Warning: Could not automatically detect WSL host IP ({e}). Falling back to localhost.")
    
    return f"http://localhost:{ollama_port}"

def general_read(yaml_str: str, file_name: str, empty_value = None):
    """
    Returns the parsed YAML string, or the parsed file if yaml_str is None.
    Returns empty_value (default: {}) if nothing could be read.
    """
    if yaml_str is None or yaml_str == "":
        target = read_yaml_file(file_name, gcv.config[gcv.DATA_DIR])

        # also need to check if there are files in a subdirectory
        # file: xyz.yaml -> subdir: xyz
        subdir, ext = os.path.splitext(file_name)
        abs_subdir = get_full_path(subdir, gcv.config[gcv.DATA_DIR])
            
        if os.path.exists(abs_subdir):
            onlyfiles = [f for f in listdir(abs_subdir) if isfile(join(abs_subdir, f))]
            if onlyfiles:
                for single_file in sorted(onlyfiles):
                    addition = read_yaml_file(single_file, abs_subdir)
                    if addition and not addition == {}:

                        # Handle Dict and List type YAMLS
                        if isinstance(addition, list):

                            # Check the upper level file type
                            if isinstance(target, list):
                                target.extend(addition)
                            else:
                                target = addition
                        else: # it can be only dict
                            if isinstance(target, dict):
                                target.update(addition)
                            else:
                                target = addition

    else:
        target = yaml.safe_load(yaml_str)
    if target == None or not target:
        target = {} if empty_value is None else empty_value
    return target

def general_read_text(file_path: str):
    try:
        with open(file_path, "r") as file:
            text = file.read()
        return text
    except:
        return ""


def flatten_dld_to_dl(indict: dict, llm: BaseChatModel = None, target_lang: str = "", fix_key: bool = False) -> dict:

    if isinstance(indict, dict):

        outdict = {}

        for key, value in indict.items():
            main_key = key
            outkey = key.replace("_", " ").title()
            if llm:
                outkey = translate_block(llm, outkey, target_lang)

            if isinstance(value, list):
                outlist = []

                for element in value:
                    if isinstance(element, dict):
                        for evalue in element.values():
                            if llm:
                                evalue = translate_block(llm, evalue, target_lang)
                            outlist.append(evalue)
                    elif isinstance(element, list):
                        for evalue in element:
                            if llm:
                                evalue = translate_block(llm, evalue, target_lang)
                            outlist.append(evalue)
                    elif isinstance(element, str):
                        if llm:
                            element = translate_block(llm, element, target_lang)
                        outlist.append(element)
            else:
                if llm:
                    value = translate_block(llm, value, target_lang)
                outlist = [value]

            if fix_key: outdict[main_key] = [outkey, outlist]
            else: outdict[outkey] = outlist
        return outdict
    
    else:
        return indict

def extract_feedback(feedback, which_one):

    if isinstance(feedback, dict):
        return feedback.get(which_one, "")

    match which_one:
        case gcv.PERSONAL_INFO: return feedback.personal_info
        case gcv.PROFESSIONAL_SUMMARY: return feedback.professional_summary
        case gcv.KEY_ACHIEVEMENTS: return feedback.key_achievements
        case gcv.SKILLS: return feedback.skills
        case gcv.CERTIFICATES: return feedback.certificates
        case gcv.EXPERIENCE: return feedback.experience
        case gcv.EDUCATIONS: return feedback.educations
        case gcv.LANGUAGES: return feedback.languages
        case gcv.OTHER_INFO: return feedback.other_info
        case _: return ""

    return ""

def format_error(e: Exception) -> str:

    # 1. Capture the exact line number and file name where the crash occurred
    exc_type, exc_obj, exc_tb = sys.exc_info()
    line_number = exc_tb.tb_lineno if exc_tb else "Unknown"
    
    # 2. Extract the complete, multi-line traceback stack history
    full_stack_trace = traceback.format_exc()
    
    # 3. Construct a highly informative, detailed error payload
    detailed_error = (
        f"Error Type: {exc_type.__name__ if exc_type else 'Exception'}\n"
        f"Core Message: {str(e)}\n"
        f"Crash Location: Line {line_number}\n"
        f"--- FULL STACK TRACE ---\n"
        f"{full_stack_trace}"
    )

    return detailed_error

def format_cv_agent_return(model_key: str, state: AppState, updates: dict, error_log = None, upper_level = False) -> dict:
    """
    Formats the return value of a CV agent run into a standardized dictionary structure.
    """

    model_key_data = state.get(gcv.MODEL_RUNS, {}).get(model_key, {})
    tmp_cv = model_key_data.get(gcv.TMP_CV, {}) if isinstance(model_key_data, dict) else {}
    tmp_cover_letter = model_key_data.get(gcv.TMP_COVER_LETTER, {}) if isinstance(model_key_data, dict) else {}

    # Special handling: If upper_level is True, we return the updates directly without merging into TMP_CV
    if upper_level:
        if gcv.CV_LOOP in updates:
            model_key_update = {
                **updates,
                gcv.ERROR_LOG: format_error(error_log) if error_log else None,
                gcv.TMP_CV: {}
            }
        else:
            model_key_update = {
                gcv.ERROR_LOG: format_error(error_log) if error_log else None,
                **updates
            }
        formatted_return = {
            gcv.MODEL_RUNS: {
                model_key: {
                    **model_key_data,
                    **model_key_update
                }
            }
        }
        #return formatted_return

    # General case: Merge updates into TMP_CV
    else:
        formatted_return = {
            gcv.MODEL_RUNS: {
                model_key: {
                    **model_key_data,
                    gcv.ERROR_LOG: format_error(error_log) if error_log else None,
                    gcv.TMP_CV: {
                        **tmp_cv,
                        **updates
                    }
                }
            }
        }

    return formatted_return

def random_string(length: int = 8) -> str:
    """
    Generates a random alphanumeric string of the specified length.
    """
    
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))
