# AI Agent with Tool Use

An autonomous AI agent powered by **GPT-4o + LangChain** that decides which tools to use to answer your questions.

Built with **LangChain · OpenAI Function Calling · FastAPI · Streamlit**.

---

## Available Tools

| Tool | Description |
|------|-------------|
| 🔍 Web Search | Real-time search via DuckDuckGo |
| 🐍 Python REPL | Execute Python code dynamically |
| 🧮 Calculator | Safe math expression evaluator |
| 📖 Wikipedia | Knowledge base lookup |
| 🕐 Date & Time | Current date and time |

---

## Project Structure

```
ai-agent/
├── src/
│   ├── agent.py    # Core agent + tool definitions
│   ├── api.py      # FastAPI REST endpoints
│   └── app.py      # Streamlit chat interface
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/your-username/ai-agent.git
cd ai-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## Usage

### Option A — Streamlit Chat UI

```bash
streamlit run src/app.py
```

Open `http://localhost:8501`

### Option B — FastAPI REST API

```bash
uvicorn src.api:app --reload
```

**Send a message:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current Bitcoin price?"}'
```

**Response:**
```json
{
  "response": "The current Bitcoin price is...",
  "tools_used": ["web_search"]
}
```

### Option C — Python directly

```python
from src.agent import build_agent

agent = build_agent()
result = agent.invoke({"input": "What is 2 to the power of 32?"})
print(result["output"])
```

---

## How It Works

```
User Query
    ↓
GPT-4o decides which tool(s) to call
    ↓
Tool executes (search / code / calc / wiki)
    ↓
Result fed back to GPT-4o
    ↓
Final answer with tool citations
```

The agent can chain multiple tools in a single response — e.g., search for data, then run Python to analyze it.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o |
| Agent framework | LangChain |
| Web search | DuckDuckGo Search |
| Code execution | Python subprocess (sandboxed) |
| API | FastAPI |
| UI | Streamlit |

---

## License

MIT
