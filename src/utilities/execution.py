
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from typing import Tuple, Dict
from utilities.outputs import *

import agentic.fetch_models as fm
import agentic.schemas.model_registry as mr
from agentic.schemas.input_schema import new_input_data
from agentic.schemas.state import AppState
from agentic.schemas.output_schema_cv import OutputSchemaCV
from agentic.schemas.output_schema_cover_letter import OutputSchemaCoverLetter
from agentic.workflow import execute_master_workflow


#________________________________________________________________________________#
# MAIN EXECUTION FUNCTION
#________________________________________________________________________________#
def execute(
        target_company: str = None,
        job_description: str = None,
        last_minute: str = None,

        personal_info_yaml: str = None,
        projects_yaml: str = None,
        skills_yaml: str = None,
        achievements_yaml: str = None,
        certificates_yaml: str = None,
        educations_yaml: str = None,
        languages_yaml: str = None,
        other_info_yaml: str = None,

        models_yaml: str = None,
        prompts_yaml: str = None,

        cv_template: str = None,
        cover_letter_template: str = None,

        save_files: bool = True
        
) -> Tuple[str, Dict]:

    # Job description is obligatory
    if not job_description:
        return "INPUT ERROR: Job Description is missing.", None

    # If no target company, try to pick from the job description
    if not target_company:
        try:
            for line in job_description.splitlines():
                if line != '':
                    target_company = line
                    break
        except:
            return "INPUT ERROR: Target Company is missing.", None
    if not target_company:
        return "INPUT ERROR: Target Company is missing.", None

    input_dict = convert_inputs(
        target_company, job_description, last_minute, 
        personal_info_yaml, projects_yaml, skills_yaml, achievements_yaml,
        certificates_yaml, educations_yaml, languages_yaml, other_info_yaml,
        models_yaml, prompts_yaml
    )

    state, messages = initialize_global_state(input_dict)

    read_templates(cv_template, cover_letter_template)

    #________________________________________________________________________________#
    # In case of error do not continue
    #________________________________________________________________________________#
    if not messages == "":
        print(messages)
        return messages, None
    
    #________________________________________________________________________________#
    # EXECUTE WORKFLOW
    #________________________________________________________________________________#
    messages = messages + "\n***** WORKFLOW EXECUTION STARTS *****\n"
    final_state, errors = execute_master_workflow(state, gcv.model_registry, gcv.master_reviewer)
    if errors:
        messages = "\nWorkflow Errors:\n\n" + messages
    messages = messages + "\n***** WORKFLOW EXECUTION FINISHED *****\n"

    #________________________________________________________________________________#
    # OUTPUT GENERATION
    #________________________________________________________________________________#
    outputs, outdir, timestamp = produce_output(final_state, save_files)  

    if save_files:
        msg = save_output_to_files(outputs, outdir, timestamp)
        messages = messages + "\n" + msg

    #________________________________________________________________________________#
    # Return results
    #________________________________________________________________________________#
    messages = "Execution Log:\n\n" + messages
    if save_files:
        print("\n\n", messages)
        return messages, None
    else:
        return messages, outputs

#________________________________________________________________________________#
# INITIALIZER
#________________________________________________________________________________#
def initialize_global_state(id: dict, supress_logging: bool = False) -> Tuple[AppState, str]:

    # AppState is a TypedDict (a plain dict at runtime): access its keys as state["key"].
    # input_data is also a TypedDict: access its fields as state[gcv.INPUT_DATA]["key"].
    state = AppState(
        input_data=new_input_data(),
        model_runs={},
        selected_best_cv = OutputSchemaCV.model_construct().model_dump() ,
        selected_best_cover_letter = OutputSchemaCoverLetter.model_construct().model_dump(),
        judge_justification_notes = "",
        global_error = ""
    )
    
    messages = ""

    #________________________________________________________________________________#
    # LOAD CONFIGURATION
    #________________________________________________________________________________#
    messages = gf.read_config(messages, supress_logging)

    # In testing we may do not need logging
    save_log_level = gcv.config[gcv.LOG_LEVEL]
    if supress_logging: gcv.config[gcv.LOG_LEVEL] = gcv.NO_COMMENTS

    #________________________________________________________________________________#
    # CHECK OBLIGATORY INPUTS
    #________________________________________________________________________________#
    logg(1, "- Target Company -")
    if id[gcv.COMPANY] == None or not id[gcv.COMPANY]:
        messages = log_error(messages, "Target Company is missing.")
    else:
        state[gcv.INPUT_DATA][gcv.COMPANY] = id[gcv.COMPANY]
        logg(2, "=\n" + state[gcv.INPUT_DATA][gcv.COMPANY] + "\n\n")

    logg(1, "- Job Description -")
    if id[gcv.JOB] == None or not id[gcv.JOB]:
        messages = log_error(messages, "Job Description is missing.")
    else:
        state[gcv.INPUT_DATA][gcv.JOB] = id[gcv.JOB]
        logg(2, "=\n" + state[gcv.INPUT_DATA]["job_description"][:100] + "..." + "\n\n")

    

    #________________________________________________________________________________#
    # LOAD ALL INFORMATION FILES/DATA
    #________________________________________________________________________________#

    for key in state[gcv.INPUT_DATA].keys():

        # Company and Job description was handled separate
        if not key == gcv.COMPANY and not key == gcv.JOB and not key == gcv.LAST_MINUTE:
            state[gcv.INPUT_DATA][key] = fill_state_part(key, id[key], gcv.config[key])
            if not state[gcv.INPUT_DATA][key]:
                messages = log_error(messages, key + "Job Description is missing.")

    #________________________________________________________________________________#
    # LOAD PROMPTS
    #________________________________________________________________________________#
    last_minute = id.get(gcv.LAST_MINUTE, "")
    logg(1, "- Loading Prompts -")
    gcv.prompts = gf.general_read(id[gcv.PROMPTS], gcv.config[gcv.PROMPTS])
    if gcv.prompts == None or not gcv.prompts:
        messages = log_error(messages, "No Prompts.")
    else:

        process_last_minute_prompts(last_minute)

    

    #________________________________________________________________________________#
    # LOAD MODELS
    #________________________________________________________________________________#
    logg(1, "- Loading LLM Models -")
    gcv.model_registry, gcv.master_reviewer, error_message = mr.parse_models_config(id[gcv.MODELS])
    if error_message:
        messages = log_error(messages, error_message)
    if gcv.model_registry == None or not gcv.model_registry:
        messages = log_error(messages, "No working LLM models.")
    if gcv.master_reviewer == None or not gcv.master_reviewer:
        messages = log_error(messages, "No master LLM model.")

    #________________________________________________________________________________#
    # Return results
    #________________________________________________________________________________#
    logg(1, "Initial Workflow State and Global Variables initialized.")

    # Reset log level
    if supress_logging: gcv.config[gcv.LOG_LEVEL] = save_log_level

    return state, messages

def log_error(msgs, text):
    msgs = msgs + "\nERROR: " + text
    logg(0, "ERROR:", text)
    return msgs

def fill_state_part(prheader, inyaml, fname):
    logg(1, "- " + prheader + " -")
    stvar = gf.general_read(inyaml, fname)
    if isinstance(stvar, str):
        logg(2, "=\n" + stvar + "\n\n")
    return stvar

def convert_inputs(
        target_company: str = None,
        job_description: str = None,
        last_minute: str = None,

        personal_info_yaml: str = None,
        projects_yaml: str = None,
        skills_yaml: str = None,
        achievements_yaml: str = None,
        certificates_yaml: str = None,
        educations_yaml: str = None,
        languages_yaml: str = None,
        other_info_yaml: str = None,

        models_yaml: str = None,
        prompts_yaml: str = None,

) -> Dict[str, any]:

    res = dict()

    res[gcv.COMPANY] = target_company
    res[gcv.JOB] = job_description
    res[gcv.LAST_MINUTE] = last_minute
    res[gcv.PERSONAL_INFO] = personal_info_yaml
    res[gcv.PROJECTS] = projects_yaml
    res[gcv.SKILLS] = skills_yaml
    res[gcv.ACHIEVEMENTS] = achievements_yaml
    res[gcv.CERTIFICATES] = certificates_yaml
    res[gcv.EDUCATIONS] = educations_yaml
    res[gcv.LANGUAGES] = languages_yaml
    res[gcv.OTHER_INFO] = other_info_yaml
    res[gcv.MODELS] = models_yaml
    res[gcv.PROMPTS] = prompts_yaml

    return res

def process_last_minute_prompts(lm: str = ""):

    if not lm: 
        return

    # Prompts are loaded in gcv.prompts 
    if not gcv.prompts:
        return

    pks = ['system', 'system_core']

    for agent_key, agent_value in gcv.prompts.items():

        if isinstance(agent_value, dict):       # do not bother single text agent instructions
            for prompt_key, prompt_value in agent_value.items():

                if prompt_key in pks:
                    prompt_value = prompt_value + "\n\nIMPORTANT INSTRUCTIONS:\n\n" + lm
                    gcv.prompts[agent_key][prompt_key] = prompt_value
