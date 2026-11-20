from agentic.schemas.output_schema_cv import OutputSchemaCV
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from utilities.logger import *
from typing import Dict, Tuple
import os
import datetime 
import yaml
import json
import html

NL = "\n"

# PATTERNS
DOCTYPE_CV = "cv"
DOCTYPE_CL = "cl"
DOCTYPE_LB = "lb"
TOP_KEY_CV = "selected_best_cv"
TOP_KEY_CL = "selected_best_cover_letter"
TEST_KEY_CV = "professional_summary"
TEST_KEY_CL = "letter_text"
TOP_KEY_LB = "labels"

# OUTPUT TYPES
OF = "out_"
CV = "_CV."
CL = "_Cover_Letter."
FODT = "fodt"
JSON = "json"
YAML = "yml"
TXT = "txt"

# patterns
CV_PI = "cv_pi_"
CL_PI = "cl_pi_"

def copy_of(a):

    if isinstance(a, str):
        return (a + '.')[:-1]

    return None

def convert_struct_to_txt(ob, first = True):
    stopper = "ERROR converting object to text"
    txt = ""

    try: 
        if first: txt = NL

        if isinstance(ob, str):
            if first: ob = ob.upper()
            txt = ob + NL
        elif isinstance(ob, int):
            txt = str(ob) + NL
        elif isinstance(ob, list):
            for litem in ob:
                sub = convert_struct_to_txt(litem, False)
                if sub == stopper: return stopper
                txt = txt + NL + sub
        elif isinstance(ob, dict):
            for key, value in ob.items():
                txt = txt + key + ":"
                sub = convert_struct_to_txt(value, False)
                if sub == stopper: return stopper
                txt = txt + NL + sub
        return txt
    except Exception as e:
        return stopper

def read_templates(cvt = None, clt = None):

    if cvt: gcv.cv_template = cvt
    else: 
        cv_file = gf.get_full_path(gcv.config[gcv.CV_TEMPLATE], gcv.config[gcv.TEMPLATES_DIR])
        gcv.cv_template = gf.general_read_text(cv_file)

    if clt: gcv.cl_template = clt
    else:
        cl_file = gf.get_full_path(gcv.config[gcv.COVLET_TEMPLATE], gcv.config[gcv.TEMPLATES_DIR])
        gcv.cl_template = gf.general_read_text(cl_file)

def get_document(obj: dict, doctype: str) -> dict:

    # This function can get the state object in 2 different formats:
    # - complete workflow state (directly after workflow execution).
    #   In this case obj has keys like "selected_best_cv" and "selected_best_cover_letter"
    #
    # - if no, then it must be already the individual cv or cl object (if it's from a correction upload).
    #   In this case we check for keys like "professional_summary" for cv or "letter_text" for cover letter

    match doctype:
        case dt if dt == DOCTYPE_CV:
            doc = obj.get(TOP_KEY_CV)
            while TOP_KEY_CV in doc.keys(): doc = doc.get(TOP_KEY_CV)
            if doc: 
                return doc
            doc = obj.get(TEST_KEY_CV)
            if doc: 
                return obj
            return {}
        case dt if dt == DOCTYPE_CL:
            doc = obj.get(TOP_KEY_CL)
            while TOP_KEY_CL in doc.keys(): doc = doc.get(TOP_KEY_CL)
            if doc: 
                return doc
            doc = obj.get(TEST_KEY_CL)
            if doc: 
                return obj
            return {}
        case dt if dt == DOCTYPE_LB:
            doc = {}                    # It's list, but must package into a dict
            lbl = obj.get(TOP_KEY_LB)
            if lbl:
                doc[TOP_KEY_LB] = lbl
                return doc

        case _: return {}
    return {}

def fodt_text(text: str) -> str:
    if text and isinstance(text, str):
        text = html.escape(text, quote=False)
        text = text.replace("\n", "<text:line-break/>")
        text = text.replace(' &', ',')
        text = text.replace('&', ',')
        return text
    else:
        return ""
    

def get_doc_key(doctype: str, key: str) -> str:
    return doctype + '_' + key

def get_key_achievements_fodt_parts(obj: dict) -> dict:

    cv_struct = obj
    if not cv_struct: return {}
    outdict = {}

    key_achievements = cv_struct.get(gcv.KEY_ACHIEVEMENTS, [])
    counter = 0
    for achievement in key_achievements:
        counter += 1
        key = f"{{achievement_{counter:02d}}}"
        value = fodt_text(achievement)
        outdict[key] = value

    for remainder in range(counter + 1, 100):
        key = f"{{achievement_{remainder:02d}}}"
        value = ""
        outdict[key] = value

    return outdict

def list_to_text(ll: list) -> str:
    text = ""
    if not isinstance(ll, list) or len(ll) == 0: return text

    for tx in ll: text = text + tx + ', '
    text = text[:len(text)-2]
    return text


def get_skills_fodt_parts(obj: dict) -> dict:

    cv_struct = obj
    if not cv_struct: return {}
    outdict = {}


    skills = cv_struct.get(gcv.SKILLS, {})
    counter = 0
    for topic, skill in skills.items():
        counter += 1
        key = f"{{skill_{counter:02d}}}"
        value = fodt_text(topic)
        outdict[key] = value
        key = f"{{skill_list_{counter:02d}}}"
        value = fodt_text(list_to_text(skill))
        outdict[key] = value

    for remainder in range(counter + 1, 100):
        key = f"{{skill_{remainder:02d}}}"
        value = ""
        outdict[key] = value
        key = f"{{skill_list_{remainder:02d}}}"
        value = ""
        outdict[key] = value


    return outdict

def get_certificates_fodt_parts(obj: dict) -> dict:

    cv_struct = obj
    if not cv_struct: return {}
    outdict = {}


    certs = cv_struct.get(gcv.CERTIFICATES, [])
    counter = 0
    for cert in certs:
        counter += 1
        key = f"{{certificate_{counter:02d}}}"
        value = fodt_text(cert)
        outdict[key] = value

    for remainder in range(counter + 1, 100):
        key = f"{{certificate_{remainder:02d}}}"
        value = ""
        outdict[key] = value

    return outdict

def get_languages_fodt_parts(obj: dict) -> dict:

    cv_struct = obj
    if not cv_struct: return {}
    outdict = {}


    langs = cv_struct.get(gcv.LANGUAGES, [])
    counter = 0
    for lang in langs:
        counter += 1
        key = f"{{language_{counter:02d}}}"
        value = fodt_text(lang.get('language', ""))
        outdict[key] = value
        key = f"{{level_{counter:02d}}}"
        value = fodt_text(lang.get('level', ""))
        outdict[key] = value

    for remainder in range(counter + 1, 100):
        key = f"{{language_{remainder:02d}}}"
        value = ""
        outdict[key] = value
        key = f"{{level_{remainder:02d}}}"
        value = ""
        outdict[key] = value

    return outdict

def get_experience_fodt_parts(obj: dict) -> dict:

    cv_struct = obj
    if not cv_struct: return {}
    outdict = {}


    exps = cv_struct.get(gcv.EXPERIENCE, [])
    counter = 0
    for exp in exps:
        counter += 1
        key = f"{{employer_company_{counter:02d}}}"
        value = fodt_text(exp.get('employer_company', ""))
        outdict[key] = value
        key = f"{{from_date_{counter:02d}}}"
        value = fodt_text(exp.get('from_date', ""))
        outdict[key] = value
        key = f"{{to_date_{counter:02d}}}"
        value = fodt_text(exp.get('to_date', ""))
        outdict[key] = value
        key = f"{{main_responsibilities_{counter:02d}}}"
        value = fodt_text(exp.get('main_responsibilities', ""))
        outdict[key] = value


    for remainder in range(counter + 1, 100):
        key = f"{{employer_company_{remainder:02d}}}"
        value = ""
        outdict[key] = value
        key = f"{{from_date_{remainder:02d}}}"
        value = ""
        outdict[key] = value
        key = f"{{to_date_{remainder:02d}}}"
        value = ""
        outdict[key] = value
        key = f"{{main_responsibilities_{remainder:02d}}}"
        value = ""
        outdict[key] = value


    return outdict

def listdict_to_list(lin: list) -> list:
    res = []
    if not lin or not isinstance(lin, list): return res
    for lelem in lin:
        if lelem and isinstance(lelem, dict):
            for value in lelem.values():
                res.append(value)
    return res

def get_education_fodt_parts(obj: dict) -> dict:

    cv_struct = obj
    if not cv_struct: return {}
    outdict = {}


    edus = cv_struct.get(gcv.EDUCATIONS, [])
    counter = 0
    for edu in edus:
        counter += 1
        key = f"{{edu_degree_{counter:02d}}}"
        value = fodt_text(edu.get('degree', ""))
        outdict[key] = value
        key = f"{{edu_field_{counter:02d}}}"
        value = fodt_text(edu.get('field', ""))
        outdict[key] = value
        key = f"{{edu_spec_{counter:02d}}}"
        value = fodt_text(list_to_text(listdict_to_list(edu.get('specializations', ""))))
        outdict[key] = value
        key = f"{{edu_institute_{counter:02d}}}"
        value = fodt_text(edu.get('institute', ""))
        outdict[key] = value
        key = f"{{edu_city_{counter:02d}}}"
        value = fodt_text(edu.get('city', ""))
        outdict[key] = value
        key = f"{{edu_country_{counter:02d}}}"
        value = fodt_text(edu.get('country', ""))
        outdict[key] = value
        key = f"{{edu_from_date_{counter:02d}}}"
        value = fodt_text(edu.get('from_date', ""))
        outdict[key] = value
        key = f"{{edu_to_date_{counter:02d}}}"
        value = fodt_text(edu.get('to_date', ""))
        outdict[key] = value

    value = ""
    for remainder in range(counter + 1, 100):
        key = f"{{edu_degree_{remainder:02d}}}"
        outdict[key] = value
        key = f"{{edu_field_{remainder:02d}}}"
        outdict[key] = value
        key = f"{{edu_spec_{remainder:02d}}}"
        outdict[key] = value
        key = f"{{edu_institute_{remainder:02d}}}"
        outdict[key] = value
        key = f"{{edu_city_{remainder:02d}}}"
        outdict[key] = value
        key = f"{{edu_country_{remainder:02d}}}"
        outdict[key] = value
        key = f"{{edu_from_date_{remainder:02d}}}"
        outdict[key] = value
        key = f"{{edu_to_date_{remainder:02d}}}"
        outdict[key] = value


    return outdict

def get_personal_info_fodt_parts(obj: dict) -> dict:

    struct = obj
    if not struct: return {}

    outdict = {}

    # Personal Info
    try:
        personal_info = struct.get(gcv.PERSONAL_INFO, {})
        if personal_info:
            for key, value in personal_info.items():
                if isinstance(value, list): 
                    key_value = value[0]
                    value_value = value[1]
                    while isinstance(value_value, list): value_value = value_value[0]

                    # Example: outer 
                    # key = 'email'
                    # value = ['E-Mail', ['gabor@sipoczy.ch']]
                    # ->
                    # key_value = 'E-Mail'
                    # value_value = 'gabor@sipoczy.ch'
                    # ->
                    # outdict['{email_label}'] = 'E-Mail'
                    # outdict['{email}'] = 'gabor@sipoczy.ch'

                    outdict['{' + key + '_label}'] = key_value
                    outdict['{' + key + '}'] = value_value

 

    except Exception as e:
        return {}
    
    return outdict

def get_cv_fodt_parts(obj: dict, labels: list) -> dict:

    cv_struct = obj
    if not cv_struct: return {}

    outdict = {}

    # labels
    for lbl in labels:
        outdict['{' + lbl['label'] + '}'] = lbl['text']

    # Personal Info
    outdict.update(get_personal_info_fodt_parts(obj))

    # Summary
    key = '{' + gcv.PROFESSIONAL_SUMMARY + '}'
    value = fodt_text(cv_struct.get(gcv.PROFESSIONAL_SUMMARY, ""))
    outdict[key] = value

    # Key Achievements
    outdict.update(get_key_achievements_fodt_parts(cv_struct))

    # Skills
    outdict.update(get_skills_fodt_parts(cv_struct))

    # Certificates
    outdict.update(get_certificates_fodt_parts(cv_struct))

    # Experience
    outdict.update(get_experience_fodt_parts(cv_struct))

    # Education
    outdict.update(get_education_fodt_parts(cv_struct))

    # Languages
    outdict.update(get_languages_fodt_parts(cv_struct))



    return outdict

def get_cl_fodt_parts(obj: dict, labels: list) -> dict:

    cl_struct = obj
    if not cl_struct: return {}

    outdict = {}

    # labels
    for lbl in labels:
        outdict['{' + lbl['label'] + '}'] = lbl['text']

    # Personal Info
    outdict.update(get_personal_info_fodt_parts(obj))

    # Letter parts
    letter = cl_struct.get(gcv.LETTER_TEXT, {})
    if not letter: return outdict

    outdict['{recipient}'] = fodt_text(letter.get('recipient', 'No recipient found'))
    outdict['{subject}'] = fodt_text(letter.get('subject', 'No subject found'))
    outdict['{letter_text}'] = fodt_text(letter.get('letter_text', 'No letter text found'))

    outdict['{date}']  = datetime.datetime.now().strftime('%d.%m.%Y')


    return outdict

def create_cv_fodt(obj: dict, labels: list) -> str:

    cv_fodt = gcv.cv_template
    texts = get_cv_fodt_parts(obj, labels)

    for key, value in texts.items():
        cv_fodt = cv_fodt.replace(key, value, -1)

    return cv_fodt

def create_cl_fodt(obj: dict, labels: list) -> str:

    cl_fodt = gcv.cl_template
    texts = get_cl_fodt_parts(obj, labels)

    for key, value in texts.items():
        cl_fodt = cl_fodt.replace(key, value, -1)

    return cl_fodt

def produce_output(obj: Dict, prepare_target: bool = False) -> Tuple[Dict, str, str]:

    outdir = gcv.config[gcv.OUTPUT_DIR]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outdir = outdir + "_" + timestamp

    if prepare_target:
        fodt_path = gf.get_full_path(gcv.config[gcv.TEMPLATES_DIR])
        output_path = gf.get_full_path(outdir)
        if not os.path.exists(output_path): os.makedirs(output_path)

    outputs = {}

    ### Labels ###
    labels = get_document(obj, DOCTYPE_LB).get(TOP_KEY_LB, [])

    ### CV ###
    cv_struct = get_document(obj, DOCTYPE_CV)       # Reduce state to cv
    cv_fodt = create_cv_fodt(cv_struct, labels)     # FODT

    outputs[CV+FODT] = cv_fodt
    outputs[CV+YAML] = yaml.dump(cv_struct)         # YAML
    outputs[CV+JSON] = json.dumps(cv_struct)        # JSON

    ### Cover Letter ###
    cl_struct = get_document(obj, DOCTYPE_CL)       # Reduce state to cl
    cl_fodt = create_cl_fodt(cl_struct, labels)     # FODT

    outputs[CL+FODT] = cl_fodt
    outputs[CL+YAML] = yaml.dump(cl_struct)         # YAML
    outputs[CL+JSON] = json.dumps(cl_struct)        # JSON


    return outputs, outdir, timestamp


def save_output_to_files(outputs: Dict, outdir: str, timestamp: str) -> str:

    msg = NL + "Files created in the " + outdir + " directory:"

    output_path = gf.get_full_path(outdir)
    
    # save cv and cl files:
    # .txt, .json, .yml, .fodt
    if not outputs or not isinstance(outputs, dict):
        msg = msg + "\nNo output produced."
        return msg

    # Create file name template: company name without space, etc + _YYYYMMDD_hhmmss"
    filename = OF + timestamp

    for key, value in outputs.items():
        target_file = filename + key
        target_path = gf.get_full_path(target_file, output_path)

        # Binary or text output
        if FODT in key or JSON in key or YAML in key or TXT in key:
            write_mode = "w"
        else:
            write_mode = "wb"

        try:
            with open(target_path, write_mode) as out_file:
                out_file.write(value)
            msg = msg + NL + target_file
        except Exception as e:
            msg = msg + NL + repr(e)

    return msg
