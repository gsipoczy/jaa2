
# CONSTANTS
CONFIG_FILE_NAME = "config.yaml"

INPUT_DATA = "input_data"

COMPANY = "target_company"
JOB = "job_description"
LAST_MINUTE = "last_minute"
DATA_DIR = "data_dir"
TEMPLATES_DIR = "templates_dir"
OUTPUT_DIR = "output_dir"
MODELS = "models"
MAX_MODELS = "max_models"
PROMPTS = "prompts"
PERSONAL_INFO = "personal_info"
PROJECTS = "projects"
SKILLS = "skills"
SKILLS_LIST = "skills_list"
ACHIEVEMENTS = "achievements"
CERTIFICATES = "certificates"
EDUCATIONS = "educations"
LANGUAGES = "languages"
OTHER_INFO = "other_info"
CV_TEMPLATE = "cv_template"
COVLET_TEMPLATE = "cover_letter_template"
STYLE = "style"
OUTPUT_CV_NAME = "output_cv_name"
OUTPUT_COVER_LETTER_NAME = "output_cover_letter_name"
LABELS = "labels"


WORKER_TEMPERATURE = 0.3
MASTER_TEMPERATURE = 0.2

# ROUTING
MAX_CV_LOOPS = "max_cv_loops"
MAX_CL_LOOPS = "max_cover_letter_loops"
ROUTE_TO_ERROR = "route_to_error"
SKIP_TO_FINAL_MERGE= "skip_to_final_merge"
CONTINUE_TO_MOTIVATION_LETTER = "continue_to_motivation_letter"
LOOP_BACK_TO_CV_WRITER = "loop_back_to_cv_writer"
MASTER_REVIEWER = "master_reviewer"

# WORKFLOW CHANNELS & CORE LAYOUT
MODEL_RUNS = "model_runs"
GLOBAL_ERROR = "global_error"
ERROR_LOG = "error_log"

# PIPELINE TRACKING KEYS
TARGET_LANGUAGE = "target_language"
COMPANY_RESEARCH = "company_research"
IS_FAKE_COMPANY = "is_fake_company"
IS_HEADHUNTER = "is_headhunter"
EMPLOYMENT_TYPE = "employment_type"

# REFLECTION LOOP STORAGE
CV_LOOP = "cv_loop"
COVER_LETTER_LOOP = "cover_letter_loop"
DRAFT = "draft"
FEEDBACK = "feedback"
APPROVED = "approved"

# WORKSPACES
FINAL_COMPILED_CV = "final_compiled_cv"
FINAL_COMPILED_CV_TXT = "final_compiled_cv_txt"
FINAL_COMPILED_COVER_LETTER = "final_compiled_cover_letter"
TMP_CV = "tmp_cv"
TMP_COVER_LETTER = "tmp_cover_letter"


# FINAL CHAMPION VERDICTS (JUDGE SELECTIONS)
SELECTED_BEST_CV = "selected_best_cv"
SELECTED_BEST_COVER_LETTER = "selected_best_cover_letter"
JUDGE_JUSTIFICATION_NOTES = "judge_justification_notes"

# OUTPUT STRUCTURE SCHEMAS (FOR PARSING TEXT / WEASYPRINT TEMPLATES)
HEADER = "header"
BODY = "body"

# AGENTS
AGENT_LANGUAGE_DETECTOR = "language_detector"
AGENT_COMPANY_RESEARCHER = "company_researcher"
AGENT_LABEL_FORMATTER = "label_formatter"

AGENT_CV_WRITER_SUMMARY = "cv_writer_summary"
AGENT_CV_WRITER_SKILLS = "cv_writer_skills"
AGENT_CV_WRITER_EXPERIENCE = "cv_writer_experience"
AGENT_CV_WRITER_METADATA = "cv_writer_metadata"
AGENT_CV_ASSEMBLER = "cv_assembler"
AGENT_CV_CRITIC = "cv_critic"
AGENT_CV_DECISION_ROUTER = "cv_decision_router"

AGENT_COVER_LETTER_CRITIC = "cover_letter_critic"
AGENT_COVER_LETTER_WRITER = "cover_letter_writer"
AGENT_COVER_LETTER_DECISION_ROUTER = "cl_decision_router"

AGENT_MASTER_JUDGE = "master_judge"
AGENT_GLOBAL_ERROR_HANDLER = "global_error_handler"

AGENT_CONTINUE = "continue"
AGENT_STOP = "stop"

# LLMs
WRITER_LLM = "writer_llm"
CRITIC_LLM = "critic_llm"

# SECTIONS
FEEDBACK_CV = "feedback_cv"
DRAFT_CV = "draft_cv"
APPROVED_CV = "approved_cv"
NO_MORE_ITERATIONS = "no_more_iterations"
FEEDBACK_COVER_LETTER = "feedback_cover_letter"
DRAFT_COVER_LETTER = "draft_cover_letter"
APPROVED_COVER_LETTER = "approved_cover_letter"
SKIP_COVER_LETTER = "skip_cover_letter"

PROFESSIONAL_SUMMARY = "professional_summary"
SUMMARY_TEXT = "summary_text"
JUSTIFICATION = "justification"
EXPERIENCE = "experience"
CV_METADATA_COMPILER = "cv_metadata_compiler"
KEY_ACHIEVEMENTS = "key_achievements"
LETTER_TEXT = "letter_text"

# LOGGING
LOG_PRINT = "log_print"
LOG_FILE = "log_file"
LOG_BOTH = "log_both"
LOG_MODE = "log_mode"

LOG_LEVEL = "log_level"

LOG_SEPARATOR = 80*"#"
LOG_ENDSEP = 80*"-"
LOG_NEW_LINE = "\n"
LOG_DOUBLE = "\n\n"
LOG_TRIPLE = "\n\n\n"
LOG_DOUBLE_SEP = LOG_SEPARATOR + LOG_NEW_LINE + LOG_SEPARATOR

LOGI = 1                # Important high level log
LOGD = 2                # Details log
NO_COMMENTS = 0
IMPORTANT_COMMENTS = 1
ALL_COMMENTS = 2

# VARIABLES
config = dict()
prompts = None

cv_template = ""
cl_template = ""

model_registry = None
master_reviewer = None


