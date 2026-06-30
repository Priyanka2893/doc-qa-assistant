from app.tools.vector_search import vector_search, filtered_vector_search
from app.tools.web_search import web_search

ALL_TOOLS = [vector_search, filtered_vector_search, web_search]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def get_tools(include_web: bool = True) -> list:
    tools = [vector_search, filtered_vector_search]
    if include_web:
        tools.append(web_search)
    return tools


def get_tool_descriptions() -> str:
    return "\n".join(
        f"- {t.name}: {t.description.strip().splitlines()[0]}"
        for t in ALL_TOOLS
    )
