# langgraph-ai

Minimal [LangGraph](https://langchain-ai.github.io/langgraph/) example: a single-node graph that sends a user message through OpenAI’s chat API and prints the conversation.

## Requirements

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) (recommended) or another way to install dependencies from `pyproject.toml`

## Setup

From the repository root:

```bash
uv sync
```

Create a `.env` file in the project root (see `.gitignore`; do not commit secrets):

```bash
OPENAI_API_KEY=your-key-here
MODEL=gpt-4o-mini
```

`MODEL` is optional and defaults to `gpt-4o-mini` if unset.

## Run

```bash
uv run python main.py
```

The script builds a graph with `START → model → END`, invokes it with one `HumanMessage`, then prints each message in the final state (`human` and `ai`).

## Project layout

| Path | Role |
|------|------|
| `main.py` | Graph definition, `ChatOpenAI` node, `main()` entrypoint |
| `pyproject.toml` | Dependencies: `langgraph`, `langchain-openai`, `python-dotenv` |

## Dependencies

- **langgraph** — stateful graph and `MessagesState`
- **langchain-openai** — `ChatOpenAI` for OpenAI-compatible chat
- **python-dotenv** — load `.env` before reading `os.environ`
