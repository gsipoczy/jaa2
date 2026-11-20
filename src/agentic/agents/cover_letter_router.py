from agentic.schemas.state import AppState
from langgraph.types import Command
from utilities.logger import *

def run_cover_letter_router(state: AppState, model_key: str):

    logg_mr(gcv.AGENT_COVER_LETTER_DECISION_ROUTER, model_key)

    # Short-circuit execution check on upstream thread failures
    if state.get(gcv.GLOBAL_ERROR) or state.get(gcv.MODEL_RUNS, {}).get(model_key, {}).get(gcv.ERROR_LOG):
        logg_mc(f"COVER_LETTER_ROUTER: ERROR in previous agents", model_key)
        return Command(
            update={},
            goto=f"END_MODEL_RUN_{model_key}" # Short-circuit this lane out to the reduce stage
        )

    # Get the latest CL iteration
    workspace = state[gcv.MODEL_RUNS][model_key]
    cl_loop = workspace.get(gcv.COVER_LETTER_LOOP, [])
    if not cl_loop:
        logg_mc(f"COVER_LETTER_ROUTER: ERROR no Cover Letter loop", model_key)
        return Command(
            update={},
            goto=f"END_MODEL_RUN_{model_key}" # Short-circuit this lane out to the reduce stage
        )

    latest_iteration = dict(cl_loop[-1])  # make sure it's a dictionary

    # Check if the CV is approved or if the maximum number of iterations has been reached
    approved = latest_iteration.get(gcv.APPROVED_COVER_LETTER, False)
    no_more_iterations = latest_iteration.get(gcv.NO_MORE_ITERATIONS, False)

    if approved or no_more_iterations:
        logg_mc(f"COVER_LETTER_ROUTER: PROCEEDING TO final judgement.", model_key)
        return Command(
            update={},
            goto=f"END_MODEL_RUN_{model_key}"  # Proceed to the cover letter writing stage
        )
    else:
        # If not approved and more iterations are allowed, continue the loop
        logg_mc(f"\n\nNEW ITERATION START: COVER LETTER", model_key)
        return Command(
            update={},
            goto=f"START_COVER_LETTER_LOOP_{model_key}"  # Continue to the next iteration of the CV writing process
        )