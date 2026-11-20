
from langchain_core.messages import AIMessage

def formatted_response(response: AIMessage) -> str:
    # --- CROSS-COMPATIBILITY PATCH FOR ANTHROPIC CLAUDE MODELS ---
    # Extract structural text strings safely whether content returns as a string or list object
    content_payload = response.content
    
    if isinstance(content_payload, list):
        # Flatten content block arrays down into unified text chunks
        extracted_chunks = []
        for block in content_payload:
            # Check for standard dictionary blocks from Anthropic
            if isinstance(block, dict):
                # STRICT GUARD: Skip reasoning, signatures, and thinking content entirely
                if block.get("type") in ["thinking", "signature", "redacted_thinking"]:
                    continue
                if "text" in block:
                    extracted_chunks.append(block["text"])
            # Check for custom LangChain MessageBlock objects
            elif hasattr(block, "type") and getattr(block, "type") in ["thinking", "signature"]:
                continue
            elif hasattr(block, "text"):
                extracted_chunks.append(block.text)
            else:
                extracted_chunks.append(str(block))
        final_text = "".join(extracted_chunks)
    else:
        final_text = str(content_payload)

    return final_text