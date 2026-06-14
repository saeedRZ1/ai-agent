"""
AI Agent with Tool Use — Core agent logic.
Tools: web search, Python REPL, calculator, Wikipedia lookup.
"""

import os
import math
import subprocess
import tempfile
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import wikipedia


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use for recent events, facts, or anything that needs up-to-date data."""
    try:
        return search.run(query)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def python_repl(code: str) -> str:
    """
    Execute Python code and return the output.
    Use for calculations, data processing, or any computational task.
    Always print() the result you want to return.
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output if output else "Code executed successfully (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (10s limit)."
    except Exception as e:
        return f"Error: {e}"


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, log, sin, cos, etc.
    Example: '2 ** 10', 'sqrt(144)', 'log(100, 10)'
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def wikipedia_search(query: str) -> str:
    """Look up a topic on Wikipedia. Use for definitions, history, or general knowledge."""
    try:
        summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
        return summary
    except wikipedia.DisambiguationError as e:
        return f"Ambiguous query. Did you mean: {', '.join(e.options[:5])}?"
    except wikipedia.PageError:
        return f"No Wikipedia page found for '{query}'."
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


@tool
def get_current_datetime(_: str = "") -> str:
    """Returns the current date and time. Use when the user asks about today's date or current time."""
    now = datetime.now()
    return now.strftime("Today is %A, %B %d, %Y. Current time: %H:%M:%S")


# ------------------------------------------------------------------
# Agent builder
# ------------------------------------------------------------------

TOOLS = [web_search, python_repl, calculator, wikipedia_search, get_current_datetime]

SYSTEM_PROMPT = """You are a capable AI assistant with access to tools.
Use tools whenever they help you give accurate, up-to-date answers.

Guidelines:
- Use web_search for recent events or facts you're unsure about.
- Use python_repl for any computation, data analysis, or code tasks.
- Use calculator for quick math.
- Use wikipedia_search for definitions, history, and general knowledge.
- Use get_current_datetime when asked about today's date or time.
- Always explain what tool you used and why.
- If a tool fails, try a different approach or explain the limitation.

Be concise, helpful, and transparent about your reasoning.
"""


def build_agent(model: str = "gpt-4o", memory_window: int = 10) -> AgentExecutor:
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=memory_window,
    )

    agent = create_openai_tools_agent(llm=llm, tools=TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )
