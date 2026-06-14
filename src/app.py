"""
Streamlit Chat UI for the AI Agent.

Run with:
    streamlit run src/app.py
"""

import os
import streamlit as st
from agent import build_agent

st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="centered")

st.title("🤖 AI Agent")
st.caption("Powered by GPT-4o with web search, Python REPL, calculator & Wikipedia.")

with st.sidebar:
    st.header("Configuration")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])

    st.markdown("---")
    st.markdown("**Available tools:**")
    st.markdown("🔍 Web Search (DuckDuckGo)")
    st.markdown("🐍 Python REPL")
    st.markdown("🧮 Calculator")
    st.markdown("📖 Wikipedia")
    st.markdown("🕐 Date & Time")

    if st.button("🔄 Clear conversation"):
        st.session_state.messages = []
        if "agent" in st.session_state:
            st.session_state.agent.memory.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Example questions:**")
    st.markdown("- What is the latest news about AI?")
    st.markdown("- Write Python code to sort a list")
    st.markdown("- What is 2 to the power of 32?")
    st.markdown("- Who invented the internet?")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("Loading agent..."):
        try:
            st.session_state.agent = build_agent(model=model)
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            st.caption(f"🛠 Tools used: {', '.join(msg['tools_used'])}")

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.agent.invoke({"input": prompt})
                response = result["output"]
                tools_used = list({
                    step[0].tool
                    for step in result.get("intermediate_steps", [])
                    if hasattr(step[0], "tool")
                })
                st.markdown(response)
                if tools_used:
                    st.caption(f"🛠 Tools used: {', '.join(tools_used)}")
            except Exception as e:
                response = f"Error: {e}"
                tools_used = []
                st.error(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "tools_used": tools_used,
    })
