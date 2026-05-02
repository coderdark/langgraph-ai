import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

# Load OPENAI_API_KEY and MODEL from a .env file into os.environ (if present).
load_dotenv()


def call_model(state: MessagesState) -> dict:
    """Graph node: send the current conversation to the LLM and append its reply.

    MessagesState keeps a `messages` list; LangGraph merges returned messages
    into that list (so the human turn stays, and the assistant reply is added).
    """
    model = ChatOpenAI(
        model=os.environ.get("MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    # Runs one completion for the full message list (here: one user message).
    reply = model.invoke(state["messages"])

    # Return only the new assistant message; the graph merges it into state.
    return {"messages": [reply]}


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file.")

    # Define a graph whose state shape is the standard chat "messages" channel.
    graph = StateGraph(MessagesState)
    # Register our LLM step as a named node.
    graph.add_node("model", call_model)
    # Wire the flow: entry → call the model once → exit.
    graph.add_edge(START, "model")
    graph.add_edge("model", END)

    # Build a runnable app (checks the graph is valid, prepares execution).
    app = graph.compile()

    # Run the graph once: initial state is a single user message.
    result = app.invoke({"messages": [HumanMessage(content="I am learning about LangGraph")]})

    # After the run, `messages` contains the user message plus the model reply.
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()
