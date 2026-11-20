

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool("web_search")
def web_search_tool(query: str) -> str:
    """
    Executes a general-purpose internet search query using DuckDuckGo.
    Use this to look up current facts, company details, code syntax, or industry trends.
    
    Args:
        query (str): The free-form search query string.
        
    Returns:
        str: Raw summarized textual search results from the web.
    """
    search = DuckDuckGoSearchRun()
    try:
        # Run search query safely
        return search.run(query)
    except Exception as e:
        return f"Error executing web search: {str(e)}"
