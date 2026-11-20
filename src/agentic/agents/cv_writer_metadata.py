import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agentic.schemas.state import AppState
from utilities.translator import translate_block
from agentic.schemas.output_schema_cv_metadata import MetadataEnsembleOutput, parse_dynamic_other_info
import utilities.global_const_var as gcv
import utilities.general_functions as gf
from utilities.logger import *
from utilities.cv_feedback import extract_feedback, compile_metadata_feedback


def run_cv_writer_metadata(state: AppState, llm: BaseChatModel, model_key: str) -> dict:
    """
    Ensemble Metadata Agent: Restructures, transforms, and translates 
    all remaining mechanical sections of the CV state matrix dynamically.
    """
    logg_mr(gcv.AGENT_CV_WRITER_METADATA, model_key)

    if state.get(gcv.GLOBAL_ERROR) or state.get(gcv.MODEL_RUNS, {}).get(model_key, {}).get(gcv.ERROR_LOG):
        return {}


    try:
        inputs = state[gcv.INPUT_DATA]
        target_lang = state[gcv.MODEL_RUNS][model_key].get(gcv.TARGET_LANGUAGE, "English")

        todo, feedback, bypass = extract_feedback(gcv.AGENT_CV_WRITER_METADATA, state, model_key)
        if todo == gcv.AGENT_STOP:
            return bypass

        critic_feedback = compile_metadata_feedback(feedback)
        
        # 1. PROCESS STEP 1: KEY ACHIEVEMENTS (With Typo Resilience Optimization)
        raw_achievements = inputs.get(gcv.ACHIEVEMENTS, [])
        processed_achievements = []
        for item in raw_achievements:
            base_template = (
                f"As {item.get('role', '')} {item.get('achievement', '')} "
                f"Business benefit: {item.get('result', '')} Link: {item.get('source_code', '')}"
            )
            fb = gf.extract_feedback(feedback, gcv.SKILLS)
            translated_item = translate_block(llm, base_template, target_lang, enforce_cv_style=True, instructions=fb)
            processed_achievements.append(translated_item)

        # 2. PROCESS STEP 2: CERTIFICATES (Mechanical Restructure Only - NO Translation)
        raw_certs = inputs.get(gcv.CERTIFICATES, [])
        processed_certs = [
            f"{c.get('institute', '')} {c.get('certificate', '')} ({c.get('year', '')})"
            for c in raw_certs if c
        ]

        raw_other = inputs.get(gcv.OTHER_INFO, {})
        processed_other = gf.flatten_dld_to_dl(raw_other, llm, target_lang)

        raw_pi = inputs.get(gcv.PERSONAL_INFO, {})
        processed_pi = gf.flatten_dld_to_dl(raw_pi, llm, target_lang, True)

        # 4. PROCESS STEPS 4, 5, 6: STRUCTURED DATA TRANSLATION
        config = gcv.prompts.get("cv_writer_metadata_restructurer", {})
        system_base = str(config.get('system_core', '')) 

        review_prompt = """
        --- REVISION CONTROL PROTOCOL ---
        You are performing a targeted revision. Below is the master feedback from the Compliance Inspector.
        Review it carefully. If the section 'professional_summary' is explicitly criticized below, apply the corrections ruthlessly.

        Master Inspector Feedback:
        """
        review_prompt = review_prompt + "\n" + critic_feedback

        
        user_payload = {
            "languages": inputs.get(gcv.LANGUAGES, []),
            "educations": inputs.get(gcv.EDUCATIONS, []),
            #"personal_info": inputs.get(gcv.PERSONAL_INFO, {})
        }

        user_template = """
        Target Language Requirement: {target_lang}
        Data Payload to Map: {payload}
        {review_prompt}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_base),
            ("user", user_template)
        ])
        
        structured_llm = llm.with_structured_output(MetadataEnsembleOutput, method="json_schema")
        chain = prompt | structured_llm
        
        response = chain.invoke({
            "target_lang": target_lang,
            "payload": json.dumps(user_payload, ensure_ascii=False),
            "review_prompt": review_prompt
        })

        # 5. COMPILE AGENT PAYLOAD FOR STATE REDUCER (.dict() parses models back to raw target states)
        # Deep dictionary conversion included to cleanly serialize the inner specializations array
        node_output = {
            "key_achievements": processed_achievements,
            "certificates": processed_certs,
            "other_info": processed_other,
            "languages": [lang.dict() for lang in response.languages],
            "educations": [
                {**edu.dict(), "specializations": [spec.dict() for spec in edu.specializations]}
                for edu in response.educations
            ],
            "personal_info": processed_pi
        }


        updates = {
            gcv.CV_METADATA_COMPILER: node_output,
        }

        return gf.format_cv_agent_return(model_key, state, updates)

    except Exception as e:
        return gf.format_cv_agent_return(model_key, state, {}, e)
    