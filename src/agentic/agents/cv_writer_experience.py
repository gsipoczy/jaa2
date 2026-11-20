import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from agentic.schemas.state import AppState
from agentic.schemas.output_schema_cv_experience import PydanticExperienceSynthesisSchema, EmployerExperienceBlock, ClientProjectItem
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from utilities.logger import *
from utilities.cv_feedback import extract_feedback

def run_cv_writer_experience(state: AppState, llm: BaseChatModel, model_key: str) -> dict:
    """
    Micro-Agent 5: Programmatically enforces chronological order and parent-child employment hierarchies,
    then uses an isolated LLM call specifically to tailor and translate historical text.
    """
    logg_mr(gcv.AGENT_CV_WRITER_EXPERIENCE, model_key)

    if state.get(gcv.GLOBAL_ERROR) or state.get(gcv.MODEL_RUNS, {}).get(model_key, {}).get(gcv.ERROR_LOG):
        return {}

    try:
        inputs = state[gcv.INPUT_DATA]
        workspace = state.get(gcv.MODEL_RUNS, {}).get(model_key, {})
        target_lang = workspace.get(gcv.TARGET_LANGUAGE, "English")
        research = workspace.get(gcv.COMPANY_RESEARCH, "")
        
        raw_projects_list = inputs.get(gcv.PROJECTS, [])

        # --- STEP 1: PROGRAMMATIC SKELETON EXTRACTION (TRUTH LAYER) ---
        experience_shell = []
        for index, item in enumerate(raw_projects_list):
            employer_name = item.get("company", "Unknown Employer").strip()
            from_dt = item.get("from_date", "").strip()
            to_dt = item.get("to_date", "").strip()
            
            # Condense parent responsibilities and nested client data into a single target payload text
            aggregated_context = f"Core Role Responsibilities:\n{item.get('main_responsibilities', '').strip()}\n\n"
            for sub_proj in item.get("projects", []):
                client_name = sub_proj.get("company", "Client").strip()
                client_text = sub_proj.get("main_responsibilities", "").strip()
                aggregated_context += f"-[Client Assignment Project at {client_name}]:\n{client_text}\n"

            experience_shell.append({
                "shell_index": index,
                "employer_company": employer_name,
                "dates": f"{from_dt} - {to_dt}",
                "raw_source_text_to_tailor": aggregated_context.strip()
            })

        str_timeline_shell = json.dumps(experience_shell, indent=2, ensure_ascii=False)

        # Extract loop history
        todo, feedback, bypass = extract_feedback(gcv.AGENT_CV_WRITER_EXPERIENCE, state, model_key)
        if todo == gcv.AGENT_STOP:
            return bypass

        fb = gf.extract_feedback(feedback, gcv.SKILLS)
        if fb:
            workspace = state[gcv.MODEL_RUNS][model_key]
            cv_loop = workspace.get(gcv.CV_LOOP, [])
            if cv_loop:
                latest_iteration = dict(cv_loop[-1])   
                latest_cv = latest_iteration.get(gcv.DRAFT_CV, {})
                latest_draft = latest_cv.get(gcv.EXPERIENCE, [])
                document = "job experience"
                critic_feedback = True
        else:
            critic_feedback = False
            latest_draft = ""
            fb = ""
            document = ""


        config = gcv.prompts.get("cv_writer_experience", {})
        system = config['system']
        user = config['user']
        revision = config['revision']
        if critic_feedback: user = user + "\n\n" + revision
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", user)
        ])

        structured_llm = llm.with_structured_output(PydanticExperienceSynthesisSchema, method="json_schema")
        chain = prompt | structured_llm

        ai_output: PydanticExperienceSynthesisSchema = chain.invoke({
            "target_language": target_lang,
            "target_company": inputs[gcv.COMPANY],
            "job_description": inputs[gcv.JOB],
            "company_research": research,
            "json_timeline_shell": str_timeline_shell,
            "latest_draft": latest_draft,
            "feedback": fb,
            "document": document
        })

        # --- STEP 3: STRICT RE-ASSEMBLY DISCIPLINE ---
        final_experience_timeline = []
        for index, shell_item in enumerate(experience_shell):
            ai_blocks = ai_output.tailored_experience_blocks
            ai_block = ai_blocks[index] if index < len(ai_blocks) else None
            
            # Map structural text block output from the LLM execution channel
            tailored_text = ai_block.get("main_responsibilities", "Text translation processing error.") if isinstance(ai_block, dict) else getattr(ai_block, "main_responsibilities", "Text processing error.")

            # Process nested client tracking objects safely if they exist
            raw_ai_projects = ai_block.get("client_projects", []) if isinstance(ai_block, dict) else getattr(ai_block, "client_projects", [])
            
            projects_list = []
            if raw_ai_projects:
                for p in raw_ai_projects:
                    # Explicit mapping check to protect string list conversion rules
                    raw_achievements = p.get("tailored_achievements", []) if isinstance(p, dict) else getattr(p, "tailored_achievements", [])
                    achievements_list = [raw_achievements] if isinstance(raw_achievements, str) else list(raw_achievements)

                    projects_list.append(
                        ClientProjectItem(
                            client_name=p.get("client_name", "Client") if isinstance(p, dict) else getattr(p, "client_name", "Client"),
                            from_date=p.get("from_date", "") if isinstance(p, dict) else getattr(p, "from_date", ""),
                            to_date=p.get("to_date", "") if isinstance(p, dict) else getattr(p, "to_date", ""),
                            tailored_achievements=achievements_list
                        )
                    )

            # Reconstruct the final primary timeline item
            # Extract date values safely by splitting your shell's composite string element
            dates_split = shell_item["dates"].split(" - ")
            from_date_val = dates_split[0] if len(dates_split) > 0 else ""
            to_date_val = dates_split[1] if len(dates_split) > 1 else ""

            # 1. Create the Pydantic block
            block = EmployerExperienceBlock(
                employer_company=shell_item["employer_company"],
                from_date=from_date_val,
                to_date=to_date_val,
                main_responsibilities=tailored_text,
                client_projects=projects_list 
            )

            # 2. Dump it to a standard Python dictionary before appending
            final_experience_timeline.append(block.model_dump())

        updates = {
            gcv.EXPERIENCE: final_experience_timeline,
        }
        return gf.format_cv_agent_return(model_key, state, updates)

    except Exception as e:
        #return {gcv.MODEL_RUNS: {model_key: {gcv.ERROR_LOG: f"Micro-Agent 5 Experience Compiler Failure: {str(e)} "}}}
        return gf.format_cv_agent_return(model_key, state, {}, e)
