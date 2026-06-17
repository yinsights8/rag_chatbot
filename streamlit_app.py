import requests
import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


BASE_URL = os.getenv("ST_BASE_URL")

st.set_page_config(page_title="AWS Agreement Q&A", layout="centered")


def ask(query: str):
    try:
        response = requests.post(f"{BASE_URL}/ask", json={"query": query}, timeout=120)
    except requests.exceptions.RequestException:
        return None, "Could not reach the backend. Is `uvicorn api:app --reload` running?"
    if response.status_code != 200:
        return None, response.json().get("detail", "Request failed.")
    return response.json(), None


def analytics():
    try:
        response = requests.get(f"{BASE_URL}/analytics", timeout=30)
    except requests.exceptions.RequestException:
        return None, "Could not reach the backend. Is `uvicorn api:app --reload` running?"
    if response.status_code != 200:
        return None, response.json().get("detail", "Request failed.")
    return response.json(), None


view = st.radio("View", ["Chat", "Analytics"], horizontal=True)

if view == "Chat":
    st.title("Ask the AWS Customer Agreement")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Ask a question about the document")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            result, error = ask(query)
            if error:
                st.error(error)
            else:
                st.write(result["answer"])
                if result["source_chunks"]:
                    with st.expander("Sources"):
                        for chunk in result["source_chunks"]:
                            st.markdown(f"**Page {chunk['page']}** (score: {chunk['score']:.2f})")
                            st.text(chunk["text"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

else:
    st.title("Analytics")
    data, error = analytics()
    if error:
        st.error(error)
    else:
        st.metric("Average response latency (ms)", f"{data['average_latency_ms']:.0f}")

        st.subheader("Most frequently asked questions")
        if data["most_frequent_questions"]:
            frequent_df = pd.DataFrame(data["most_frequent_questions"])
            st.dataframe(frequent_df, use_container_width=True)
            st.bar_chart(frequent_df.set_index("query")["count"])
        else:
            st.write("No queries logged yet.")

        st.subheader("Queries with no answer found")
        if data["no_answer_queries"]:
            no_answer_df = pd.DataFrame(data["no_answer_queries"])
            st.dataframe(no_answer_df, use_container_width=True)
            st.bar_chart(no_answer_df.set_index("query")["count"])
        else:
            st.write("No unanswered queries logged yet.")
