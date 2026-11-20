from agentic.schemas.state import AppState

import utilities.global_const_var as gcv
import utilities.general_functions as gf
from agentic.schemas.multi_model import ReflectionIterationCV
from agentic.schemas.output_schema_cv import OutputSchemaCV
from utilities.logger import logg_mr


def run_cv_draft_compiler(state: AppState, model_key: str) -> dict:
    if state.get(gcv.GLOBAL_ERROR) or state.get(gcv.MODEL_RUNS, {}).get(model_key, {}).get(gcv.ERROR_LOG):
        return {}

    logg_mr(gcv.AGENT_CV_ASSEMBLER, model_key)

    # Get the actual model run
    workspace = state.get(gcv.MODEL_RUNS, {}).get(model_key, {})

    # The CV writer agents store their results in the TMP_CV workspace
    tmp_cv = workspace.get(gcv.TMP_CV, {})
    metadata = tmp_cv.get(gcv.CV_METADATA_COMPILER, {})

    # Draft CV
    draft_cv = OutputSchemaCV(
        personal_info=metadata.get(gcv.PERSONAL_INFO, {}),
        
        # Safely fallback to a string for professional summary
        professional_summary=tmp_cv.get(gcv.PROFESSIONAL_SUMMARY, {}).get(gcv.SUMMARY_TEXT, ""),
        
        key_achievements=metadata.get(gcv.KEY_ACHIEVEMENTS, []),
        
        #  RIGHT: Default to an empty dictionary {} to satisfy Dict[str, List[str]]
        skills=tmp_cv.get(gcv.SKILLS, {}),
        
        certificates=metadata.get(gcv.CERTIFICATES, []),
        experience=tmp_cv.get(gcv.EXPERIENCE, []),
        educations=metadata.get(gcv.EDUCATIONS, []),
        languages=metadata.get(gcv.LANGUAGES, []),
        other_info=metadata.get(gcv.OTHER_INFO, {}),
    )



    # In the end create a new cv_loop entry
    # Copy the list, so the existing state is not mutated in place
    cv_loop = list(workspace.get(gcv.CV_LOOP, []))

    iteration: ReflectionIterationCV = {
        gcv.DRAFT_CV: draft_cv.model_dump()
    }
    cv_loop.append(iteration)

    updates = {
        gcv.CV_LOOP: cv_loop,
    }
    return gf.format_cv_agent_return(model_key, state, updates, error_log=None, upper_level=True)

