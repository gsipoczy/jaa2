
import traceback
import sys
import utilities.global_const_var as gcv
from rich import print
import json


def logg(required_level: int = 0, text: str = "", new_lines = False):
    if text == None: return

    if gcv.config[gcv.LOG_LEVEL] == gcv.NO_COMMENTS:
        return
    if required_level > gcv.config[gcv.LOG_LEVEL] or required_level == gcv.NO_COMMENTS:
        return
    
    if gcv.config[gcv.LOG_MODE] == gcv.LOG_PRINT or gcv.config[gcv.LOG_MODE] == gcv.LOG_BOTH:
        if new_lines:
            formatted_json = json.dumps(text, indent=4, ensure_ascii=False)
            text = formatted_json.replace("\\n", "\n")
        print(text)
        
    if gcv.config[gcv.LOG_MODE] == gcv.LOG_FILE or gcv.config[gcv.LOG_MODE] == gcv.LOG_BOTH:
        pass

def logg_mr(text: str = "", model_key: str = ""):
    required_level = gcv.LOGI
    logg(required_level, f"### AGENT START: [{model_key}] -> {text.upper()}", False)

def logg_mc(text: str = "", model_key: str = ""):
    required_level = gcv.LOGI
    logg(required_level, f"--- COMMENT: [{model_key}] -> {text.upper()}", False)

def logg_object(content: any, header: str = ""):
    required_level = gcv.LOGD
    logg(required_level, gcv.LOG_ENDSEP)
    logg(required_level, header + ":\n")
    try:
        logg(required_level, content)
    except:
        logg(required_level, "--- object is not printable ---")
    logg(required_level, gcv.LOG_ENDSEP)

def log_text_error(agent: str, msg: str, model_key: str) -> dict:
    message = agent + ": " + msg
    logg(1, agent + ": " + message)
    if model_key == None:
        return {gcv.GLOBAL_ERROR: message}
    else:
        return {
            gcv.MODEL_RUNS: {
                model_key: {
                    gcv.ERROR_LOG: message
                }
            }
        }

def log_error(agent: str, e: Exception, model_key: str) -> dict:

    # 1. Capture the exact line number and file name where the crash occurred
    exc_type, exc_obj, exc_tb = sys.exc_info()
    line_number = exc_tb.tb_lineno if exc_tb else "Unknown"
    
    # 2. Extract the complete, multi-line traceback stack history
    full_stack_trace = traceback.format_exc()
    
    # 3. Construct a highly informative, detailed error payload
    detailed_error = (
        f"🚨 {agent}\n"
        f"Error Type: {exc_type.__name__ if exc_type else 'Exception'}\n"
        f"Core Message: {str(e)}\n"
        f"Crash Location: Line {line_number}\n"
        f"--- FULL STACK TRACE ---\n"
        f"{full_stack_trace}"
    )

    logg(1, agent + ": " + detailed_error)

    if model_key == None:
        return {gcv.GLOBAL_ERROR: detailed_error}
    else:
        return {
            gcv.MODEL_RUNS: {
                model_key: {
                    gcv.ERROR_LOG: detailed_error
                }
            }
        }


def logg_error_exit():
    logg(gcv.LOGI, "Exiting because of previous error.")


