import os
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def add(a: int, b: int) -> int:
    """
    Add a and b.

    Args:
        a: first int
        b: secont int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """
    Multiplies a and b.

    Args:
        a: first int
        b: secont int
    """
    return a * b

def divide(a: int, b: int) -> int:
    """
    Divide a and b.

    Args:
        a: first int
        b: secont int
    """
    return a / b

tools = [add, multiply, divide]

# define llm with bind tools
llm = ChatOpenAI(
    model = "gpt-5-nano",
    temperature = 0.2,
    max_tokens = 250,
    timeout = None,
    max_retries = 2
)
llm_with_tools = llm.bind_tools(tools)

sys_message = SystemMessage(content="You are a helpful assistant taskes with writing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}

# build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)
builder.add_edge("tools", "assistant")

# compile graph
graph = builder.compile()