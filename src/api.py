"""
FastAPI REST API for the AI Agent.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import build_agent


executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    executor = build_agent(model=model)
    print(f"Agent ready with model: {model}")
    yield


app = FastAPI(
    title="AI Agent API",
    description="Agentic AI with tool use: web search, Python REPL, calculator, Wikipedia.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str
    tools_used: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("OPENAI_MODEL", "gpt-4o")}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not executor:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        result = executor.invoke({"input": request.message})
        # Extract tool names from intermediate steps
        tools_used = list({
            step[0].tool
            for step in result.get("intermediate_steps", [])
            if hasattr(step[0], "tool")
        })
        return QueryResponse(response=result["output"], tools_used=tools_used)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-memory")
def clear_memory():
    if not executor:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    executor.memory.clear()
    return {"status": "memory cleared"}


@app.get("/tools")
def list_tools():
    return {"tools": [t.name for t in executor.tools] if executor else []}


@app.get("/")
def root():
    return {
        "message": "AI Agent API is running.",
        "docs": "/docs",
        "endpoints": ["/chat", "/tools", "/clear-memory", "/health"],
    }
