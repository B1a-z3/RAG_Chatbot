"""
app.py
Streamlit chat UI over the RAG pipeline.

Run: streamlit run src/app.py
"""

import streamlit as st
from rag import answer_question

st.set_page_config(page_title="Product Docs Assistant", page_icon="📄")
st.title("📄 Product Docs Assistant")
st.caption("RAG over your product documentation — ChromaDB + Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("Sources & metrics"):
                for s in msg["meta"]["sources"]:
                    st.write(f"- **{s['source']}** (relevance: {s['score']})")
                st.write(
                    f"Latency: {msg['meta']['latency_seconds']}s | "
                    f"Tokens: {msg['meta']['input_tokens']} in / {msg['meta']['output_tokens']} out | "
                    f"Cost: ${msg['meta']['cost_usd']}"
                )

if query := st.chat_input("Ask a question about the docs..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = answer_question(query)
        st.markdown(result["answer"])
        with st.expander("Sources & metrics"):
            for s in result["sources"]:
                st.write(f"- **{s['source']}** (relevance: {s['score']})")
            st.write(
                f"Latency: {result['latency_seconds']}s | "
                f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out | "
                f"Cost: ${result['cost_usd']}"
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "meta": result}
    )
