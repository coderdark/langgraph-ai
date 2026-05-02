import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Load OPENAI_API_KEY and MODEL from a .env file into os.environ (if present).
load_dotenv()


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together."""
    return str(a * b) + " is the result of the multiplication of " + str(a) + " and " + str(b)


# Tools the LLM may call (OpenAI function-calling format is applied in bind_tools, not in ChatOpenAI(...)).
TOOLS = [multiply]


def call_model(state: MessagesState) -> dict:
    """Graph node: send messages to the LLM (with tools bound) and append its reply.

    Use bind_tools(...) instead of ChatOpenAI(tools=...). Constructor kwargs that are not
    first-class fields end up in model_kwargs and break the OpenAI client serializer.
    """
    llm = ChatOpenAI(
        model=os.environ.get("MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    model = llm.bind_tools(TOOLS, tool_choice="auto")
    reply = model.invoke(state["messages"])

    return {"messages": [reply]}


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file.")

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "model")
    # After the LLM: run tools if the last AIMessage has tool_calls, else finish.
    graph.add_conditional_edges(
        "model",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    # Tool messages go back to the model for a natural-language answer.
    graph.add_edge("tools", "model")

    app = graph.compile()

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="I am learning about LangGraph; what is 3.13 * 8.2?",
                )
            ],
        },
    )

    for message in result["messages"]:
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()
