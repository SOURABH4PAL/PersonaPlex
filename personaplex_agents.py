import os
from dotenv import load_dotenv
from typing import TypedDict, List

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph

from pypdf import PdfReader
import pandas as pd

load_dotenv()

# -------------------------------
# LLM
# -------------------------------

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# -------------------------------
# STATE SHARED BY ALL AGENTS
# -------------------------------

class AgentState(TypedDict):
    task: str
    messages: List[BaseMessage]
    file_content: str
    file_name: str
    file_path: str


# -------------------------------
# FILE READER
# -------------------------------

def read_file(file_path: str) -> str:
    ext = file_path.split(".")[-1].lower()

    try:
        if ext == "pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text

        if ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        if ext == "csv":
            df = pd.read_csv(file_path)
            return df.to_string()

        return "Unsupported file format."

    except Exception as e:
        return f"File reading error: {str(e)}"


# ===============================
# AGENTS
# ===============================

def manager_agent(state: AgentState):
    return {
        "task": state["task"],
        "messages": [],
        "file_content": "",
        "file_name": state["file_name"],
        "file_path": state["file_path"]
    }


def file_upload_agent(state: AgentState):
    content = read_file(state["file_path"])

    return {
        **state,
        "file_content": content[:12000]
    }


def analyst_agent(state: AgentState):
    response = llm.invoke(
        f"""
You are a document analysis AI.

Document:
{state['file_name']}

Content:
{state['file_content']}

Task:
{state['task']}

Give a clear structured answer.
"""
    )

    return {
        **state,
        "messages": [response]
    }


def reviewer_agent(state: AgentState):
    response = llm.invoke(
        f"""
Improve and clean the answer below.
Remove markdown symbols like *, **, ###.
Make it professional plain text.

{state['messages'][-1].content}
"""
    )

    return {
        **state,
        "messages": state["messages"] + [response]
    }


# ===============================
# GRAPH
# ===============================

graph = StateGraph(AgentState)

graph.add_node("manager", manager_agent)
graph.add_node("file_upload", file_upload_agent)
graph.add_node("analyst", analyst_agent)
graph.add_node("reviewer", reviewer_agent)

graph.set_entry_point("manager")

graph.add_edge("manager", "file_upload")
graph.add_edge("file_upload", "analyst")
graph.add_edge("analyst", "reviewer")

agent_graph = graph.compile()
