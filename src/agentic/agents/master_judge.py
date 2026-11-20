import yaml
from copy import deepcopy
from typing import Dict, List
from langchain_core.language_models.chat_models import BaseChatModel

from agentic.schemas.state import AppState
from agentic.schemas.output_schema_cv import OutputSchemaCV
from agentic.schemas.output_schema_cover_letter import OutputSchemaCoverLetter
from agentic.schemas.master_judge_schema import CherryPickSelection, ModelPipelineWorkspace
import utilities.global_const_var as gcv
from utilities.logger import *

def run_master_judge(state: AppState, master_llm: BaseChatModel) -> dict:
    """
    Executes at the final collection bottleneck of the workflow graph.
    Scans state['model_runs'] and cherry-picks the best individual parts 
    independently across different lanes while preserving structural boilerplate assets.
    """
    logg_mr(gcv.AGENT_CV_WRITER_SKILLS, "MASTER")

    if state.get("global_error"):
        return {}

    model_runs = state.get("model_runs", {})

    # 1. Filter and keep only successful lanes.
    #    A lane is valid if it produced a CV. A cover letter is optional: some lanes skip it
    #    (e.g. fake company / headhunter), so cover letter lanes are tracked separately.
    valid_packages: Dict[str, ModelPipelineWorkspace] = {}
    cl_packages: Dict[str, ModelPipelineWorkspace] = {}
    for key, workspace in model_runs.items():
        if workspace.get("error_log"):
            continue
        if not workspace.get("final_compiled_cv"):
            continue
        valid_packages[key] = workspace
        if workspace.get("final_compiled_cover_letter"):
            cl_packages[key] = workspace

    if not valid_packages:
        return {
            "global_error": "Master Judge Failure: No parallel model lane execution completed successfully."
        }

    # 2. Extract and format individual granular components for comparative visibility
    component_blocks: List[str] = []
    for model_key, workspace in valid_packages.items():
        professional_summary = ""
        key_achievements = ""
        skills = ""
        experience = ""
        cl_subject = ""
        cl_text = ""

        cv_data: OutputSchemaCV = workspace.get("final_compiled_cv")
        if cv_data:
            professional_summary = cv_data.get(gcv.PROFESSIONAL_SUMMARY, "")
            key_achievements = cv_data.get(gcv.KEY_ACHIEVEMENTS, "")
            skills = cv_data.get(gcv.SKILLS, "")
            experience = cv_data.get(gcv.EXPERIENCE, "")
            
        cl_data: OutputSchemaCoverLetter = workspace.get("final_compiled_cover_letter")
        if cl_data:
            cl_subject = cl_data.get(gcv.LETTER_TEXT, {}).get("subject", "")
            cl_text = cl_data.get(gcv.LETTER_TEXT, {}).get("letter_text", "")
        else:
            cl_subject = cl_text = "NOT AVAILABLE (this lane produced no cover letter - do not select it for cover letter components)"

        block = (
            f"=== WORKER LANE ID: {model_key} ===\n"
            f"[PROFESSIONAL SUMMARY]:\n{professional_summary}\n\n"
            f"[KEY ACHIEVEMENTS]:\n{key_achievements}\n"
            f"[SKILLS]:\n{skills}\n"
            f"[EXPERIENCE]:\n{experience}\n"
            f"[COVER LETTER SUBJECT]:\n{cl_subject}\n\n"
            f"[COVER LETTER TEXT]:\n{cl_text}\n"
            f"================================================\n"
        )
        component_blocks.append(block)
    
    formatted_components = "\n".join(component_blocks)
    if cl_packages:
        formatted_components += (
            "\nLanes eligible for cover_letter_subject and cover_letter_text selection: "
            + ", ".join(cl_packages.keys()) + "\n"
        )
    else:
        formatted_components += (
            "\nNo lane produced a cover letter: return \"NONE\" for the cover letter winning keys.\n"
        )

    input_data = state.get("input_data", {})
    company_name = input_data.get("company", "Target Company")
    job_desc = input_data.get("job", "Target Job Description")

    # 3. Read prompt configurations
    config = gcv.prompts.get("master_judge", {})
    system = config['system']
    user = config['user']
    user_prompt = user.format(
        company_name=company_name,
        job_description=job_desc,
        formatted_candidate_components=formatted_components
    )

    # 4. Invoke LLM structured selector
    structured_llm = master_llm.with_structured_output(CherryPickSelection, method="json_schema")
    
    try:
        decision: CherryPickSelection = structured_llm.invoke([
            {"role": "system", "content": system },
            {"role": "user", "content": user_prompt}
        ])

        # 5. Safe Fallback mapping logic to guarantee valid model keys
        first_valid_key = next(iter(valid_packages))

        first_cl_key = next(iter(cl_packages), None)

        def get_valid_key(chosen_key: str) -> str:
            return chosen_key if chosen_key in valid_packages else first_valid_key

        def get_valid_cl_key(chosen_key: str) -> str | None:
            # Cover letter parts may only come from lanes that actually produced a cover letter
            return chosen_key if chosen_key in cl_packages else first_cl_key

        s_key = get_valid_key(decision.summary_winning_key)
        a_key = get_valid_key(decision.achievements_winning_key)
        sk_key = get_valid_key(decision.skills_winning_key)
        e_key = get_valid_key(decision.experience_winning_key)
        cl_sub_key = get_valid_cl_key(decision.cl_subject_winning_key)
        cl_txt_key = get_valid_cl_key(decision.cl_text_winning_key)

        # 6. Extract baseline elements from the first successful workspace lane
        #    (for the cover letter: the first lane that has one)
        base_cv = valid_packages[first_valid_key]["final_compiled_cv"]
        base_cl = cl_packages[first_cl_key]["final_compiled_cover_letter"] if first_cl_key else None

        # 7. Reconstruct the hybrid CV. The lane outputs are stored as plain dicts,
        #    so start from the baseline lane (technical boilerplate: personal info,
        #    certificates, educations, languages, other info) and override the winners.
        def cv_part(lane_key: str, field: str):
            return deepcopy(valid_packages[lane_key]["final_compiled_cv"].get(field))

        final_cv = deepcopy(base_cv)
        final_cv[gcv.PROFESSIONAL_SUMMARY] = cv_part(s_key, gcv.PROFESSIONAL_SUMMARY)  # Winner summary
        final_cv[gcv.KEY_ACHIEVEMENTS] = cv_part(a_key, gcv.KEY_ACHIEVEMENTS)          # Winner achievements
        final_cv[gcv.SKILLS] = cv_part(sk_key, gcv.SKILLS)                              # Winner skills
        final_cv[gcv.EXPERIENCE] = cv_part(e_key, gcv.EXPERIENCE)                      # Winner experience

        # 8. Reconstruct the Cover Letter: {personal_info, letter_text: {recipient, subject, letter_text}}
        #    If no lane produced a cover letter, the final cover letter is None.
        def cl_part(lane_key: str, field: str):
            return cl_packages[lane_key]["final_compiled_cover_letter"].get(gcv.LETTER_TEXT, {}).get(field, "")

        final_cover_letter = None
        if base_cl is not None:
            final_cover_letter = deepcopy(base_cl)
            final_cover_letter.setdefault(gcv.LETTER_TEXT, {})
            final_cover_letter[gcv.LETTER_TEXT]["subject"] = cl_part(cl_sub_key, "subject")          # Winner subject
            final_cover_letter[gcv.LETTER_TEXT]["letter_text"] = cl_part(cl_txt_key, "letter_text")  # Winner body text

        # 9. Format audit trail tracking metrics
        tracking_footer = (
            f"\n\n[Cherry-Picked Composition Mapping]:"
            f"\n- Summary Source: {s_key}"
            f"\n- Achievements Source: {a_key}"
            f"\n- Skills Source: {sk_key}"
            f"\n- Experience Source: {e_key}"
            f"\n- Cover Letter Subject Source: {cl_sub_key or 'NONE (no lane produced a cover letter)'}"
            f"\n- Cover Letter Text Source: {cl_txt_key or 'NONE (no lane produced a cover letter)'}"
            f"\n- Technical Boilerplate Source: {first_valid_key}"
            f"\n- Cover Letter Boilerplate Source: {first_cl_key or 'NONE'}"
        )

        # Labels: just take any.
        labels = []
        for value in model_runs.values():
            labels = value.get('labels', [])
            if labels:
                break

        return {
            "selected_best_cv": final_cv,
            "selected_best_cover_letter": final_cover_letter,
            "judge_justification_notes": decision.justification_notes + tracking_footer,
            "labels": labels
        }

    except Exception as e:
        return {
            "global_error": f"Critical exception during Cherry-Picking Judge execution: {str(e)}"
        }
