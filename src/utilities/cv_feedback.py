from agentic.schemas.output_schema_cv_critic import CVCriticFeedback
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from agentic.schemas.state import AppState
from utilities.logger import *

BYPASS_TEXT = "Bypassed LLM call: Copied unchanged section forward from previous loop iteration."
INITIAL_DRAFT_COMMENT = "Initial draft pass. No feedback yet."

def extract_feedback(agent: str, state: AppState, model_key: str):

    # Get the last critic
    workspace = state[gcv.MODEL_RUNS][model_key]
    cv_loop = workspace.get("cv_loop", [])

    # No feedback yet
    if not cv_loop:
        logg(gcv.LOGD, "Feedback: no history")
        return gcv.AGENT_CONTINUE, initial_cv_feedback(), {}

    # Have a loop already
    latest_loop_run = cv_loop[-1]
    latest_feedback = latest_loop_run.get(gcv.FEEDBACK_CV, {})
    latest_draft = latest_loop_run.get(gcv.DRAFT_CV, {})

    # log the feedback
    logg_object(latest_feedback, "Full feedback content")

    # Check fully approved: if yes, clear the critics
    if latest_loop_run.get(gcv.APPROVED_CV, False):
        latest_feedback = initial_cv_feedback()

    # agent processing
    match agent:
        case gcv.AGENT_CV_WRITER_SUMMARY:
            todo, feedback, updates = process_summary(latest_feedback, latest_draft)
        case gcv.AGENT_CV_WRITER_EXPERIENCE:
            todo, feedback, updates = process_experience(latest_feedback, latest_draft)
        case gcv.AGENT_CV_WRITER_METADATA:
            todo, feedback, updates = process_metadata(latest_feedback, latest_draft)
        case gcv.AGENT_CV_WRITER_SKILLS:
            todo, feedback, updates = process_skills(latest_feedback, latest_draft)
        case _:
            return gcv.AGENT_CONTINUE, initial_cv_feedback(), {}

    # Bypassed result must land in TMP_CV like a normal agent result, so the compiler finds it
    if todo == gcv.AGENT_STOP:
        return todo, feedback, gf.format_cv_agent_return(model_key, state, updates)
    return todo, feedback, {}

#################################################################################################################
# CV SUMMARY AGENT
#################################################################################################################
def process_summary(latest_feedback, latest_draft):
    # Get the previous work
    latest_result = latest_draft.get(gcv.PROFESSIONAL_SUMMARY, "")
    # There is a critic for this agent
    if not gf.extract_feedback(latest_feedback, gcv.PROFESSIONAL_SUMMARY) == "":
        logg(gcv.LOGD, "Feedback: critic FOUND for the agent, let it rework.")
        return gcv.AGENT_CONTINUE, latest_feedback, {}

    # No critic, but also no previous written version
    if not latest_result:
        latest_feedback[gcv.PROFESSIONAL_SUMMARY] = INITIAL_DRAFT_COMMENT
        logg(gcv.LOGD, "Feedback: no complains, and no previous version, let it write new")
        return gcv.AGENT_CONTINUE, latest_feedback, {}

    # No critic, have latest version
    updates = {
        gcv.PROFESSIONAL_SUMMARY: {
            gcv.SUMMARY_TEXT: latest_result,
            gcv.JUSTIFICATION: BYPASS_TEXT
        },
    }
    logg(gcv.LOGD, "Feedback: no complains, no rework is necessary, use the previous version.")
    return gcv.AGENT_STOP, latest_feedback, updates

#################################################################################################################
# CV EXPERIENCE AGENT
#################################################################################################################
def process_experience(latest_feedback, latest_draft):
    # Get the previous work
    latest_result = latest_draft.get(gcv.EXPERIENCE, [])
    # There is a critic for this agent
    if not gf.extract_feedback(latest_feedback, gcv.EXPERIENCE) == "":
        logg(gcv.LOGD, "Feedback: critic FOUND for the agent, let it rework.")
        return gcv.AGENT_CONTINUE, latest_feedback, {}

    # No critic, but also no previous written version
    if not latest_result:
        latest_feedback[gcv.EXPERIENCE] = INITIAL_DRAFT_COMMENT
        logg(gcv.LOGD, "Feedback: no complains, and no previous version, let it write new")
        return gcv.AGENT_CONTINUE, latest_feedback, {}

    # No critic, have latest version
    updates = {
        gcv.EXPERIENCE: latest_result,
    }
    logg(gcv.LOGD, "Feedback: no complains, no rework is necessary, use the previous version.")
    return gcv.AGENT_STOP, latest_feedback, updates

#################################################################################################################
# CV METADATA AGENT
#################################################################################################################
def process_metadata(latest_feedback, latest_draft):
    # Get the previous work
    logg_object(latest_draft, "Metadata Latest Draft")

    latest_result = {}
    latest_result[gcv.PERSONAL_INFO] = latest_draft.get(gcv.PERSONAL_INFO, {})
    latest_result[gcv.KEY_ACHIEVEMENTS] = latest_draft.get(gcv.KEY_ACHIEVEMENTS, [])
    latest_result[gcv.CERTIFICATES] = latest_draft.get(gcv.CERTIFICATES, [])
    latest_result[gcv.EDUCATIONS] = latest_draft.get(gcv.EDUCATIONS, [])
    latest_result[gcv.LANGUAGES] = latest_draft.get(gcv.LANGUAGES, [])
    latest_result[gcv.OTHER_INFO] = latest_draft.get(gcv.OTHER_INFO, {}) 

    # There is a critic for this agent
    if ( not gf.extract_feedback(latest_feedback, gcv.PERSONAL_INFO) == "" ) or \
        (  not gf.extract_feedback(latest_feedback, gcv.KEY_ACHIEVEMENTS) == "" ) or \
        (  not gf.extract_feedback(latest_feedback, gcv.CERTIFICATES) == "" ) or \
        (  not gf.extract_feedback(latest_feedback, gcv.EDUCATIONS) == "" ) or \
        (  not gf.extract_feedback(latest_feedback, gcv.LANGUAGES) == "" ) or \
        (  not gf.extract_feedback(latest_feedback, gcv.OTHER_INFO)== "" ):
        logg(gcv.LOGD, "Feedback: critic FOUND for the agent, let it rework.")
        return gcv.AGENT_CONTINUE, latest_feedback, {}

    # No critic, but also no previous written version
    if not latest_result:
        latest_feedback[gcv.PERSONAL_INFO] = INITIAL_DRAFT_COMMENT
        logg(gcv.LOGD, "Feedback: no complains, and no previous version, let it write new")
        return gcv.AGENT_CONTINUE, latest_feedback, {}

    # No critic, have latest version
    updates = {
        gcv.CV_METADATA_COMPILER: latest_result,
    }
    logg(gcv.LOGD, "Feedback: no complains, no rework is necessary, use the previous version.")
    return gcv.AGENT_STOP, latest_feedback, updates

#################################################################################################################
# CV SKILLS AGENT
#################################################################################################################
def process_skills(latest_feedback, latest_draft):
    # Get the previous work
    latest_result = latest_draft.get(gcv.SKILLS, {})
    # There is a critic for this agent
    if not gf.extract_feedback(latest_feedback, gcv.SKILLS) == "":
        logg(gcv.LOGD, "Feedback: critic FOUND for the agent, let it rework.")
        return gcv.AGENT_CONTINUE, latest_feedback, {}

    # No critic, but also no previous written version
    if not latest_result:
        latest_feedback[gcv.SKILLS] = INITIAL_DRAFT_COMMENT
        logg(gcv.LOGD, "Feedback: no complains, and no previous version, let it write new")
        return gcv.AGENT_CONTINUE, latest_feedback, {}

    # No critic, have latest version
    updates = {
        gcv.SKILLS: latest_result,
    }
    logg(gcv.LOGD, "Feedback: no complains, no rework is necessary, use the previous version.")
    return gcv.AGENT_STOP, latest_feedback, updates

#################################################################################################################
# Empty Feedback
#################################################################################################################        


def initial_cv_feedback() -> CVCriticFeedback:
    return CVCriticFeedback(
        personal_info = "",
        professional_summary = "",
        key_achievements = "",
        skills = "",
        certificates = "",
        experience = "",
        educations = "",
        languages = "",
        other_info = ""
    )

#################################################################################################################
# Create a Metadata Feedback
#################################################################################################################

def compile_metadata_feedback(full: CVCriticFeedback) -> str:

    logg_object(full, "CV_FEEDBACK: Feedback returned from cv_writer_metadata()")

    # if 'process_metadata()' decided, that no critic, but also no previous written version,
    # it writes this comment.
    if gf.extract_feedback(full, gcv.PERSONAL_INFO) == INITIAL_DRAFT_COMMENT:
        return INITIAL_DRAFT_COMMENT

    cf = ""

    # Theoretically there are several sections made by the metadata agent, but only
    # the Key Achievements is 'creative', the others are just a copy-paste + translation of the input 

    critic_found = False

    if gf.extract_feedback(full, gcv.PERSONAL_INFO)  :
        cf = cf + "\n\nIn Section Personal Information:\n" + gf.extract_feedback(full, gcv.PERSONAL_INFO) 
        critic_found = True
    if gf.extract_feedback(full, gcv.KEY_ACHIEVEMENTS)  : 
        cf = cf + "\n\nIn Section Key Achievements:\n" + gf.extract_feedback(full, gcv.KEY_ACHIEVEMENTS) 
        critic_found = True
    if gf.extract_feedback(full, gcv.CERTIFICATES) :
        cf = cf + "\n\nIn Section Certificates:\n" + gf.extract_feedback(full, gcv.CERTIFICATES) 
        critic_found = True
    if gf.extract_feedback(full, gcv.EDUCATIONS) :
        cf = cf + "\n\nIn Section Educations:\n" + gf.extract_feedback(full, gcv.EDUCATIONS) 
        critic_found = True
    if gf.extract_feedback(full, gcv.LANGUAGES) :
        cf = cf + "\n\nIn Section Languages:\n" + gf.extract_feedback(full, gcv.LANGUAGES) 
        critic_found = True
    if gf.extract_feedback(full, gcv.OTHER_INFO) :
        cf = cf + "\n\nIn Section Other Information:\n" + gf.extract_feedback(full, gcv.OTHER_INFO) 
        critic_found = True

    # Not found any critic
    if not critic_found:
        return ""

    text = """
    --- REVISION CONTROL PROTOCOL ---
    You are performing a targeted revision. Below is the master feedback from the Compliance Inspector.
    Review it carefully. If the section 
    'personal_info' or
    'key_achievements' or
    'certificates' or
    'educations' or
    'languages' or
    'other_info'
    is explicitly criticized below, apply the corrections ruthlessly.

    Master Inspector Feedback:\n
    """

    cf = text + cf

    logg_object(cf, "Compiled Metadata Feedback")
    return cf
    