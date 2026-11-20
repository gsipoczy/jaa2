from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
import traceback
import sys
import utilities.global_const_var as gcv

def translate_block(
        llm: BaseChatModel, 
        source_text: str, 
        target_lang: str, 
        enforce_cv_style: bool = False,
        instructions: str = "") -> str:
    """
    Utility helper: Translates raw content blocks cleanly using disk-loaded YAML templates.
    """
    if not target_lang or target_lang.lower() == "english":
        return source_text.strip()

    if not source_text or source_text == "": return ""

    try:
        # 1. Load prompts out of global configuration registry
        config = gcv.prompts.get("translator", {})
        
        raw_system_prompt = str(config.get('system_prompt', ''))
        raw_style_rules = str(config.get('style_rules', ''))
        
        # 2. Build out placeholder fragments programmatically
        style_injection = raw_style_rules if enforce_cv_style else ""

        if instructions:
            instructions_injection = """
            
            IMPORTANT:
            If the following instructions contain something translation related directive, apply it.
            Instructions:
            """
            instructions_injection = instructions_injection + "\n" + instructions
        else:
            instructions_injection = ""
        
        # 3. Perform programmatic string replacement BEFORE LangChain template compilation
        # This completely visualizes the template text, avoiding parsing variable bugs
        final_system_string = (
            raw_system_prompt
            .replace("{target_lang}", target_lang)
            .replace("{style_rules_placeholder}", style_injection)
            .replace("{instructions_placeholder}", instructions_injection)
        )
        
        # 4. Map to ChatPromptTemplate cleanly. 
        # Only '{text}' remains as a variable for LangChain to resolve.
        prompt = ChatPromptTemplate.from_messages([
            ("system", final_system_string),
            ("user", "{text}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"text": source_text})
        
        # 5. Extract structural text values based on model output type arrays
        content_payload = response.content
        if isinstance(content_payload, list):
            extracted_chunks = []
            for block in content_payload:
                if isinstance(block, dict):
                    if block.get("type") in ["thinking", "signature", "redacted_thinking"]:
                        continue
                    if "text" in block:
                        extracted_chunks.append(block["text"])
                elif hasattr(block, "type") and getattr(block, "type") in ["thinking", "signature"]:
                    continue
                elif hasattr(block, "text"):
                    extracted_chunks.append(block.text)
                else:
                    extracted_chunks.append(str(block))
            raw_text = "".join(extracted_chunks)
        else:
            raw_text = str(content_payload)
            
        return raw_text.strip()

    except Exception as e:
        full_trace = traceback.format_exc()
        raise RuntimeError(
            f"Translation Utility Failure!\n"
            f"Target Lang: {target_lang}\n"
            f"Raw Message: {str(e)}\n"
            f"{full_trace}"
        ) from e
