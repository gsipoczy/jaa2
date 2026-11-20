from agentic.schemas.state import AppState
from langgraph.types import Command
from utilities.logger import *

def run_cv_router(state: AppState, model_key: str):

    logg_mr(gcv.AGENT_CV_DECISION_ROUTER, model_key)

    # Short-circuit execution check on upstream thread failures
    if state.get(gcv.GLOBAL_ERROR) or state.get(gcv.MODEL_RUNS, {}).get(model_key, {}).get(gcv.ERROR_LOG):
        logg_mc(f"CV_ROUTER: ERROR in previous agents", model_key)

        return Command(
            update={},
            goto=f"END_MODEL_RUN_{model_key}" # Short-circuit this lane out to the reduce stage
        )

    # Get the latest CV iteration
    workspace = state[gcv.MODEL_RUNS][model_key]
    cv_loop = workspace.get(gcv.CV_LOOP, [])
    if not cv_loop:
        logg_mc(f"CV_ROUTER: ERROR no CV loop", model_key)
        return Command(
            update={},
            goto=f"END_MODEL_RUN_{model_key}" # Short-circuit this lane out to the reduce stage
        )

    latest_iteration = dict(cv_loop[-1])  # make sure it's a dictionary

    # Check if the CV is approved or if the maximum number of iterations has been reached
    approved = latest_iteration.get(gcv.APPROVED_CV, False)
    no_more_iterations = latest_iteration.get(gcv.NO_MORE_ITERATIONS, False)
    skip_cover_letter = latest_iteration.get("skip_cover_letter", False)

    if approved or no_more_iterations:
        # If the CV is approved or no more iterations are allowed, proceed to the next stage (e.g., cover letter writing)   
        if skip_cover_letter:
            logg_mc(f"CV_ROUTER: However should not, we still create a cover letter", model_key)
            return Command(
                update={},
                goto=f"START_COVER_LETTER_LOOP_{model_key}"   # Skip to the final judge agent if cover letter is skipped
            )
        else:
            logg_mc(f"CV_ROUTER: PROCEEDING TO cover letter generation.", model_key)
            return Command(
                update={},
                goto=f"START_COVER_LETTER_LOOP_{model_key}"  # Proceed to the cover letter writing stage
            )
    else:
        # If not approved and more iterations are allowed, continue the loop
        logg_mc(f"\n\nNEW ITERATION START: Curriculum Vitae", model_key)
        return Command(
            update={},
            goto=f"START_CV_LOOP_{model_key}"  # Continue to the next iteration of the CV writing process
        )