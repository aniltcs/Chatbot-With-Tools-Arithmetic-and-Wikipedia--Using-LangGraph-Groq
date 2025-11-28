# app.py

import streamlit as st
from langchain.tools import tool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# -----------------------------
# Step 1: Load environment and define model
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = groq_api_key

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# -----------------------------
# Step 2: Define tools
# -----------------------------
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [add, multiply, divide, wiki_tool]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# -----------------------------
# Step 3: Define state
# -----------------------------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add] # operator.add annotation instructs LangGraph: Whenever a node returns new messages, append them to the existing list.
    llm_calls: int

# -----------------------------
# Step 4: Nodes
# -----------------------------
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or answer directly."""

    # Full history is fine; LLM handles it
    messages = state["messages"]

    # Build LLM input
    llm_input = [
        SystemMessage(content="You are a helpful assistant that can use tools for arithmetic or Wikipedia queries.")
    ] + messages

    response = model_with_tools.invoke(llm_input)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def tool_node(state: MessagesState):
    """Executes a tool and returns a ToolMessage."""
    last_msg = state["messages"][-1]
    results = []

    for tc in last_msg.tool_calls:
        tool_fn = tools_by_name[tc["name"]]
        output = tool_fn.invoke(tc["args"])

        results.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tc["id"]
            )
        )

    # Add a follow-up signal so LLM continues properly
    results.append(SystemMessage(content="[Tool execution complete. Continue reasoning.]"))

    return {"messages": results}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """If last LLM output has tool calls, run tool."""
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"

    return END


# -----------------------------
# Step 5: Build agent graph
# -----------------------------
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# -----------------------------
# Step 6: Streamlit UI
# -----------------------------
st.title("🧮 Arithmetic + Wikipedia Assistant (Groq + LangGraph)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_calls" not in st.session_state:
    st.session_state.llm_calls = 0

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage) and msg.content.strip():
        st.chat_message("user").write(msg.content)
    # elif isinstance(msg, ToolMessage):
    #     st.chat_message("assistant").write(f"🔧 **Tool Output:** {msg.content}")
    elif isinstance(msg, AIMessage) and msg.content.strip():
        st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    result_state = agent.invoke(
        {
            "messages": st.session_state.messages,
            "llm_calls": st.session_state.llm_calls
        },
        config={"recursion_limit": 50}
    )

    st.session_state.messages = result_state["messages"]
    st.session_state.llm_calls = result_state.get("llm_calls", 0)
    with st.chat_message("assistant"):
        st.write(result_state["messages"][-1].content)
        st.subheader("LLM Calls Made")
        st.write(st.session_state.llm_calls)
