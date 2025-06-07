import streamlit as st
from chatbot_engine import load_documents, get_vectorstore, get_qa_chain

st.set_page_config(page_title="Document Chatbot - Zipp.ai", layout="wide")

st.title("📄 Zipp.ai Regulation Document Chatbot")
st.markdown("Ask questions about GCP guidelines (ICH E6(R2), E6(R3))")

# Document selection
doc_options = {
    "ICH E6(R2)": "data/Text_v1.txt",
    "ICH E6(R3)": "data/Text_v2.txt"
}

selected_docs = st.multiselect("Select documents:", list(doc_options.keys()))

# Show selected docs for debugging
if selected_docs:
    st.write("Selected Docs:", selected_docs)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Start chat if documents are selected
if selected_docs:
    doc_paths = [doc_options[doc] for doc in selected_docs]
    documents = load_documents(doc_paths)
    vectorstore = get_vectorstore(documents)
    qa_chain = get_qa_chain(vectorstore)

    user_query = st.text_input("Ask your question here...")

    if user_query:
        response = qa_chain.run({
            "question": user_query,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append((user_query, response))

        # Show chat history
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
else:
    st.info("Please select at least one document to start chatting.")
